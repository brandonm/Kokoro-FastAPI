"""Tests for model TTL and subprocess worker lifecycle."""

import asyncio
import multiprocessing as mp
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.src.inference.model_manager import ModelManager


# ── Fake workers ───────────────────────────────────────────────────────

def _recv_request(conn):
    """Block until the next parent message; returns None on shutdown/close."""
    while True:
        try:
            if not conn.poll(0.5):
                continue
            return conn.recv()
        except (EOFError, OSError):
            return None


def _fake_worker(conn, cancel_id=None, *args):
    """Minimal worker: responds to generate with one chunk."""
    conn.send(("ready",))
    while True:
        msg = _recv_request(conn)
        if msg is None:
            break
        if msg[0] == "generate":
            req_id = msg[1]
            conn.send(("chunk", req_id, np.zeros(100, dtype=np.float32), None))
            conn.send(("done", req_id))
    conn.close()


def _slow_fake_worker(conn, cancel_id=None, *args):
    """Worker that takes 300ms per generate for concurrency tests."""
    import time

    conn.send(("ready",))
    while True:
        msg = _recv_request(conn)
        if msg is None:
            break
        if msg[0] == "generate":
            req_id = msg[1]
            conn.send(("chunk", req_id, np.zeros(100, dtype=np.float32), None))
            time.sleep(0.3)
            conn.send(("chunk", req_id, np.zeros(100, dtype=np.float32), None))
            conn.send(("done", req_id))
    conn.close()


def _crash_after_one_chunk_worker(conn, cancel_id=None, *args):
    """Worker that sends one chunk then crashes (simulates OOM/segfault)."""
    conn.send(("ready",))
    try:
        if not conn.poll(5):
            return
        msg = conn.recv()
    except (EOFError, OSError):
        return
    if msg is not None and msg[0] == "generate":
        conn.send(("chunk", msg[1], np.zeros(100, dtype=np.float32), None))
    # Crash: close pipe without sending "done"
    conn.close()


def _error_worker(conn, cancel_id=None, *args):
    """Worker that returns an error for every generate request."""
    conn.send(("ready",))
    while True:
        msg = _recv_request(conn)
        if msg is None:
            break
        if msg[0] == "generate":
            conn.send(("error", msg[1], "Simulated generation failure"))
    conn.close()


CHUNKS_PER_REQUEST = 5


def _multi_chunk_worker(conn, cancel_id=None, wrote_everything=None, *args):
    """Streams several chunks per request, each stamped with its req_id.

    Stamping the audio with the request id is what lets a test prove that audio
    from an abandoned request never reaches the next one. Honours cancellation
    so an abandoned request can be drained without waiting for the whole stream.

    `wrote_everything` is set once this request's last byte is in the pipe, so a
    test can assert on the pipe's contents without racing the worker thread.
    """
    import time

    conn.send(("ready",))
    while True:
        msg = _recv_request(conn)
        if msg is None:
            break
        if msg[0] == "generate":
            req_id = msg[1]
            if wrote_everything is not None:
                wrote_everything.clear()
            for _ in range(CHUNKS_PER_REQUEST):
                if cancel_id is not None and cancel_id.value == req_id:
                    break
                conn.send(
                    (
                        "chunk",
                        req_id,
                        np.full(100, float(req_id), dtype=np.float32),
                        None,
                    )
                )
                time.sleep(0.02)
            conn.send(("done", req_id))
            if wrote_everything is not None:
                wrote_everything.set()
    conn.close()


def _stale_prefix_worker(conn, cancel_id=None, *args):
    """Replies to every request with a *previous* request's traffic first.

    Simulates a pipe that is already desynchronised — the parent must recognise
    the mismatched req_id and discard it rather than play it as this request's
    audio.
    """
    conn.send(("ready",))
    while True:
        msg = _recv_request(conn)
        if msg is None:
            break
        if msg[0] == "generate":
            req_id = msg[1]
            stale = req_id - 1
            conn.send(("chunk", stale, np.full(100, 99.0, dtype=np.float32), None))
            conn.send(("done", stale))
            conn.send(
                (
                    "chunk",
                    req_id,
                    np.full(100, float(req_id), dtype=np.float32),
                    None,
                )
            )
            conn.send(("done", req_id))
    conn.close()


def _never_finishes_worker(conn, cancel_id=None, *args):
    """Streams chunks but never sends a terminator — an unresponsive worker."""
    conn.send(("ready",))
    while True:
        msg = _recv_request(conn)
        if msg is None:
            break
        if msg[0] == "generate":
            req_id = msg[1]
            for _ in range(3):
                conn.send(
                    ("chunk", req_id, np.zeros(100, dtype=np.float32), None)
                )
            # No ("done", req_id) — the parent must not wait forever.
    conn.close()


# ── Fixtures ───────────────────────────────────────────────────────────

def _start_fake_worker(mgr, worker_fn):
    """Start a fake worker in a thread and wire it into the manager."""
    parent_conn, child_conn = mp.Pipe()
    cancel_id = mp.Value("q", 0)
    wrote_everything = threading.Event()
    t = threading.Thread(
        target=worker_fn,
        args=(child_conn, cancel_id, wrote_everything),
        daemon=True,
    )
    t.start()
    # The real worker is a subprocess, which gets its own copy of child_conn —
    # so production closes the parent's copy. These doubles are threads sharing
    # this very object, so closing it (or dropping the last reference to it and
    # letting the GC close it) would shut the worker's end of the pipe and break
    # every send. Hold it open for the lifetime of the test instead.
    assert parent_conn.poll(5), "Fake worker did not send ready"
    msg = parent_conn.recv()
    assert msg[0] == "ready"

    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = True
    mock_proc.pid = 99999
    mock_proc.exitcode = None
    # After join(), mark as no longer alive
    def join_side_effect(timeout=None):
        mock_proc.is_alive.return_value = False
    mock_proc.join.side_effect = join_side_effect

    mgr._process = mock_proc
    mgr._conn = parent_conn
    mgr._cancel_id = cancel_id
    mgr._worker_thread = t  # keep reference
    mgr._fake_child_conn = child_conn  # keep the worker's end open (see above)
    mgr._fake_wrote_everything = wrote_everything


@pytest.fixture
def manager():
    """Create a ModelManager with fake worker spawner."""
    ModelManager._instance = None
    mgr = ModelManager()
    mgr._device = "cuda"

    async def fake_spawn():
        _start_fake_worker(mgr, _fake_worker)

    mgr._spawn_worker = fake_spawn

    yield mgr

    # Cleanup
    if mgr._conn:
        try:
            mgr._conn.send(None)
            mgr._conn.close()
        except Exception:
            pass
    if mgr._ttl_task and not mgr._ttl_task.done():
        mgr._ttl_task.cancel()
    ModelManager._instance = None


# ── Helpers ────────────────────────────────────────────────────────────

async def _generate_one(mgr):
    """Run one generate request and consume all chunks."""
    chunks = []
    async for chunk in mgr.generate("test", ("af_heart", "/fake/voice.pt")):
        chunks.append(chunk)
    return chunks


# ── TTL tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_kills_worker_after_idle(mock_settings, manager):
    """Worker should be killed after TTL seconds of inactivity."""
    mock_settings.model_ttl = 0.2
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    await _generate_one(manager)

    await asyncio.sleep(0.5)
    assert manager._conn is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_resets_on_generate(mock_settings, manager):
    """Each generate call should reset the TTL timer."""
    mock_settings.model_ttl = 0.5
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    await _generate_one(manager)

    await asyncio.sleep(0.3)
    await _generate_one(manager)

    # 300ms after second request — still within TTL
    await asyncio.sleep(0.3)
    assert manager._conn is not None

    # Wait for full TTL
    await asyncio.sleep(0.4)
    assert manager._conn is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_negative_never_kills(mock_settings, manager):
    """TTL of -1 should never kill the worker."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    await _generate_one(manager)

    await asyncio.sleep(0.2)
    assert manager._conn is not None
    assert manager._ttl_task is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_zero_kills_immediately(mock_settings, manager):
    """TTL of 0 should kill worker right after request completes."""
    mock_settings.model_ttl = 0
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    await _generate_one(manager)

    await asyncio.sleep(0.05)
    assert manager._conn is None


# ── Respawn tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_respawns_after_kill(mock_settings, manager):
    """After TTL kills the worker, next generate should spawn a new one."""
    mock_settings.model_ttl = 0
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    await _generate_one(manager)
    await asyncio.sleep(0.05)
    assert manager._conn is None

    # Next generate should respawn (fake_spawn will be called)
    chunks = await _generate_one(manager)
    assert len(chunks) > 0


# ── Serialization test ─────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_generate_serialized_under_lock(mock_settings, manager):
    """Lock should prevent concurrent pipe access."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    # Use slow worker
    async def slow_spawn():
        _start_fake_worker(manager, _slow_fake_worker)

    manager._spawn_worker = slow_spawn
    await manager._spawn_worker()

    results = [[], []]

    async def consume(idx):
        async for chunk in manager.generate("test", ("af_heart", "/fake/voice.pt")):
            results[idx].append(chunk)

    task1 = asyncio.create_task(consume(0))
    task2 = asyncio.create_task(consume(1))
    await task1
    await task2

    # Both should complete successfully (serialized, no interleaving)
    assert len(results[0]) > 0
    assert len(results[1]) > 0


# ── Error handling tests ───────────────────────────────────────────────

@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_worker_crash_mid_generate(mock_settings, manager):
    """Worker dying mid-generate should raise RuntimeError, not hang."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    async def crash_spawn():
        _start_fake_worker(manager, _crash_after_one_chunk_worker)

    manager._spawn_worker = crash_spawn
    await manager._spawn_worker()

    with pytest.raises(RuntimeError, match="Worker died during generation"):
        async for _ in manager.generate("test", ("af_heart", "/fake/voice.pt")):
            pass


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_worker_generation_error(mock_settings, manager):
    """Worker returning an error should propagate as RuntimeError."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    async def error_spawn():
        _start_fake_worker(manager, _error_worker)

    manager._spawn_worker = error_spawn
    await manager._spawn_worker()

    with pytest.raises(RuntimeError, match="Simulated generation failure"):
        async for _ in manager.generate("test", ("af_heart", "/fake/voice.pt")):
            pass


# ── Active request tracking ───────────────────────────────────────────

@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_active_requests_tracked(mock_settings, manager):
    """Active request count should increment and decrement properly."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    assert manager._active_requests == 0

    await _generate_one(manager)
    assert manager._active_requests == 0  # back to 0 after completion


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_active_requests_decremented_on_error(mock_settings, manager):
    """Active request count should decrement even when worker errors."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    async def error_spawn():
        _start_fake_worker(manager, _error_worker)

    manager._spawn_worker = error_spawn
    await manager._spawn_worker()

    with pytest.raises(RuntimeError):
        async for _ in manager.generate("test", ("af_heart", "/fake/voice.pt")):
            pass

    assert manager._active_requests == 0


# ── Cleanup tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unload_all_cleans_up(manager):
    """unload_all() should close conn and kill process."""
    await manager._spawn_worker()
    assert manager._conn is not None

    manager.unload_all()

    assert manager._conn is None
    assert manager._process is None


# ── Manual unload + ensure_backend shims ───────────────────────────────

@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_unload_kills_worker(mock_settings, manager):
    """unload() should terminate the worker subprocess to fully release VRAM."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    assert manager._conn is not None

    await manager.unload()

    assert manager._conn is None
    assert manager._process is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_unload_cancels_pending_ttl_task(mock_settings, manager):
    """unload() called mid-TTL should cancel the pending timer."""
    mock_settings.model_ttl = 60
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    await _generate_one(manager)
    assert manager._ttl_task is not None and not manager._ttl_task.done()

    await manager.unload()

    assert manager._ttl_task is None or manager._ttl_task.done()
    assert manager._conn is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ensure_backend_spawns_when_dead(mock_settings, manager):
    """ensure_backend() should spawn a worker if none is alive."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    assert manager._conn is None
    await manager.ensure_backend()
    assert manager._conn is not None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ensure_backend_noop_when_alive(mock_settings, manager):
    """ensure_backend() should not respawn if the worker is already alive."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await manager._spawn_worker()
    original_conn = manager._conn
    original_process = manager._process

    await manager.ensure_backend()

    assert manager._conn is original_conn
    assert manager._process is original_process


# ── Abandoned generation / barge-in ───────────────────────────────────
#
# A generation streams many chunks before its terminator. If the consumer walks
# away early — which happens on every client barge-in — those chunks stay queued
# in the pipe. Releasing the lock at that point hands the next request a
# desynchronised pipe: it reads the previous utterance's audio, stops at the
# stale terminator, and leaves its own chunks behind for the request after that.
# Nothing self-corrects, and only a restart clears it.


async def _spawn_with(manager, worker_fn):
    async def spawn():
        _start_fake_worker(manager, worker_fn)

    manager._spawn_worker = spawn
    await manager._spawn_worker()


async def _take_one_then_abandon(manager, text="first"):
    """Consume a single chunk, then walk away mid-stream."""
    stream = manager.generate(text, ("af_heart", "/fake/voice.pt"))
    got = []
    async for chunk in stream:
        got.append(chunk)
        break
    await stream.aclose()
    return got


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_abandoned_generation_does_not_leak_into_next_request(
    mock_settings, manager
):
    """Audio from an abandoned request must never play as the next request's."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await _spawn_with(manager, _multi_chunk_worker)

    first = await _take_one_then_abandon(manager)
    assert len(first) == 1
    assert first[0].audio[0] == 1.0  # request 1 stamps its audio with id 1

    second = [
        chunk
        async for chunk in manager.generate("second", ("af_heart", "/fake/voice.pt"))
    ]

    # Every chunk belongs to request 2, and none of request 2's chunks went missing.
    assert len(second) == CHUNKS_PER_REQUEST
    assert all(chunk.audio[0] == 2.0 for chunk in second), (
        "request 2 played audio left over from the abandoned request 1"
    )


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_abandoned_generation_drains_the_pipe(mock_settings, manager):
    """Nothing may be left queued once an abandoned request is cleaned up."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await _spawn_with(manager, _multi_chunk_worker)

    await _take_one_then_abandon(manager)

    # Barrier, not a timing guess: without it the assertion below could poll the
    # pipe before the worker had even written the leftovers it is meant to catch.
    assert manager._fake_wrote_everything.wait(5), "worker never finished the request"

    assert manager._conn is not None, "worker should survive an ordinary barge-in"
    assert not manager._conn.poll(0), "abandoned request left replies in the pipe"
    assert manager._cancel_id.value == 0, "cancel flag not cleared after drain"
    assert manager._active_requests == 0


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_stale_replies_are_discarded(mock_settings, manager):
    """A desynchronised pipe heals itself instead of playing the wrong audio."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await _spawn_with(manager, _stale_prefix_worker)

    chunks = [
        chunk
        async for chunk in manager.generate("hello", ("af_heart", "/fake/voice.pt"))
    ]

    assert len(chunks) == 1, "stale replies were played instead of discarded"
    assert chunks[0].audio[0] == 1.0  # the reply tagged with *our* req_id


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.DRAIN_TIMEOUT_S", 0.5)
@patch("api.src.inference.model_manager.settings")
async def test_undrainable_worker_is_killed(mock_settings, manager):
    """If the pipe can't be cleaned, kill the worker rather than poison the next request."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await _spawn_with(manager, _never_finishes_worker)

    await _take_one_then_abandon(manager)

    # A respawn costs a model load; a poisoned pipe costs every later request.
    assert manager._conn is None
    assert manager._process is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_repeated_barge_in_stays_aligned(mock_settings, manager):
    """The desync was self-perpetuating — prove it doesn't survive repetition."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.get_device.return_value = "cuda"

    await _spawn_with(manager, _multi_chunk_worker)

    for _ in range(5):
        await _take_one_then_abandon(manager)

    expected_id = float(manager._req_seq + 1)
    final = [
        chunk
        async for chunk in manager.generate("final", ("af_heart", "/fake/voice.pt"))
    ]

    assert len(final) == CHUNKS_PER_REQUEST
    assert all(chunk.audio[0] == expected_id for chunk in final)
