"""Tests for model TTL and subprocess worker lifecycle."""

import asyncio
import multiprocessing as mp
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.src.inference.model_manager import ModelManager


# ── Fake workers ───────────────────────────────────────────────────────

def _fake_worker(conn, *args):
    """Minimal worker: responds to generate with one chunk."""
    conn.send(("ready",))
    while True:
        try:
            if not conn.poll(0.5):
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        if msg[0] == "generate":
            conn.send(("chunk", np.zeros(100, dtype=np.float32), None))
            conn.send(("done",))
    conn.close()


def _slow_fake_worker(conn, *args):
    """Worker that takes 300ms per generate for concurrency tests."""
    import time

    conn.send(("ready",))
    while True:
        try:
            if not conn.poll(0.5):
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        if msg[0] == "generate":
            conn.send(("chunk", np.zeros(100, dtype=np.float32), None))
            time.sleep(0.3)
            conn.send(("chunk", np.zeros(100, dtype=np.float32), None))
            conn.send(("done",))
    conn.close()


def _crash_after_one_chunk_worker(conn, *args):
    """Worker that sends one chunk then crashes (simulates OOM/segfault)."""
    conn.send(("ready",))
    try:
        if not conn.poll(5):
            return
        msg = conn.recv()
    except (EOFError, OSError):
        return
    if msg is not None and msg[0] == "generate":
        conn.send(("chunk", np.zeros(100, dtype=np.float32), None))
    # Crash: close pipe without sending "done"
    conn.close()


def _error_worker(conn, *args):
    """Worker that returns an error for every generate request."""
    conn.send(("ready",))
    while True:
        try:
            if not conn.poll(0.5):
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        if msg[0] == "generate":
            conn.send(("error", "Simulated generation failure"))
    conn.close()


# ── Fixtures ───────────────────────────────────────────────────────────

def _start_fake_worker(mgr, worker_fn):
    """Start a fake worker in a thread and wire it into the manager."""
    parent_conn, child_conn = mp.Pipe()
    t = threading.Thread(target=worker_fn, args=(child_conn,), daemon=True)
    t.start()
    child_conn.close()
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
    mgr._worker_thread = t  # keep reference


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
