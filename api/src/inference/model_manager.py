"""Kokoro V1 model management with subprocess isolation for clean GPU memory release."""

import asyncio
import multiprocessing as mp
import os
from typing import Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import AudioChunk, BaseModelBackend
from .kokoro_v1 import KokoroV1
from .model_worker import worker_entry

# Use spawn so the child gets a clean CUDA context (no inherited state)
_ctx = mp.get_context("spawn")

# Timeout waiting for the worker to load the model and signal ready
WORKER_READY_TIMEOUT_S = 120


class ModelManager:
    """Manages a TTS inference subprocess.

    Instead of loading the model in-process (which permanently allocates
    a CUDA context), inference runs in a child process.  Killing that
    process releases *all* GPU memory — matching Ollama-style behaviour.
    """

    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        self._config = config or model_config
        self._device: Optional[str] = None
        # Subprocess state
        self._process: Optional[mp.Process] = None
        self._conn: Optional[mp.connection.Connection] = None
        # TTL
        self._ttl_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._active_requests = 0
        # Lightweight backend instance for isinstance() checks in TTSService
        self._backend_stub: Optional[KokoroV1] = None

    # ── Device / backend info ──────────────────────────────────────────

    def _determine_device(self) -> str:
        return settings.get_device()

    def get_backend(self) -> BaseModelBackend:
        """Return a stub KokoroV1 so TTSService isinstance checks pass."""
        if not self._backend_stub:
            self._backend_stub = KokoroV1()
        return self._backend_stub

    @property
    def current_backend(self) -> str:
        return "kokoro_v1"

    # ── Subprocess lifecycle ───────────────────────────────────────────

    async def _spawn_worker(self) -> None:
        """Spawn the inference subprocess and wait until the model is loaded."""
        model_path = await paths.get_model_path(
            self._config.pytorch_kokoro_v1_file
        )
        config_path = os.path.join(os.path.dirname(model_path), "config.json")

        parent_conn, child_conn = _ctx.Pipe()
        proc = None

        try:
            proc = _ctx.Process(
                target=worker_entry,
                args=(child_conn, model_path, config_path, self._device),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent doesn't use the child end

            # Wait for "ready" signal
            ready = await asyncio.to_thread(
                parent_conn.poll, WORKER_READY_TIMEOUT_S
            )
            if not ready:
                raise RuntimeError(
                    f"Worker did not become ready within {WORKER_READY_TIMEOUT_S}s"
                )

            msg = await asyncio.to_thread(parent_conn.recv)
            if msg[0] == "error":
                raise RuntimeError(f"Worker failed to start: {msg[1]}")

            self._process = proc
            self._conn = parent_conn
            logger.info(f"Worker subprocess started (PID {proc.pid})")
        except Exception:
            # Clean up on any failure
            parent_conn.close()
            if proc is not None:
                proc.kill()
                proc.join(timeout=5)
            raise

    def _worker_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    async def _ensure_worker(self) -> None:
        """Spawn the worker if it's not running."""
        if self._worker_alive():
            return

        # Clean up dead process
        if self._process is not None:
            await asyncio.to_thread(self._process.join, 1)
            self._process = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                logger.debug(f"Cleanup: closing dead worker conn: {e}")
            self._conn = None

        import time
        start = time.perf_counter()
        await self._spawn_worker()
        ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"Worker ready in {ms}ms")

    async def _kill_worker(self) -> None:
        """Kill the worker subprocess, freeing all GPU memory."""
        pid = self._process.pid if self._process else None

        if self._conn is not None:
            try:
                self._conn.send(None)  # graceful shutdown signal
            except Exception as e:
                logger.debug(f"Shutdown signal send failed: {e}")
            try:
                self._conn.close()
            except Exception as e:
                logger.debug(f"Conn close failed: {e}")
            self._conn = None

        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._process = None

        logger.info(f"Worker (PID {pid}) terminated, GPU memory freed")

    # ── Startup ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        self._device = self._determine_device()

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        import time

        start = time.perf_counter()

        await self.initialize()
        await self._ensure_worker()

        voices = await paths.list_voices()
        voice_path = await paths.get_voice_path(settings.default_voice)

        # Warmup — run a short generate through the subprocess
        async for _ in self._generate_via_worker(
            "Warmup text for initialization.",
            settings.default_voice,
            voice_path,
            speed=1.0,
            lang_code=None,
            return_timestamps=False,
        ):
            pass

        ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"Warmup completed in {ms}ms")

        self._reset_ttl_timer()
        return self._device, "kokoro_v1", len(voices)

    # ── Model loading (no-op stub for compatibility) ───────────────────

    async def load_model(self, path: str) -> None:
        """No-op — model loading happens inside the worker subprocess."""
        pass

    # ── Generation ─────────────────────────────────────────────────────

    async def _generate_via_worker(
        self,
        text: str,
        voice_name: str,
        voice_path: str,
        speed: float,
        lang_code: Optional[str],
        return_timestamps: bool,
    ):
        """Send a generate request to the worker and yield AudioChunks."""
        from ..structures.schemas import WordTimestamp

        lang = lang_code if lang_code else voice_name[0].lower()
        conn = self._conn  # capture local ref

        if conn is None:
            raise RuntimeError("Worker connection is not available")

        try:
            await asyncio.to_thread(
                conn.send,
                (
                    "generate",
                    text,
                    voice_name,
                    voice_path,
                    speed,
                    lang,
                    return_timestamps,
                ),
            )
        except (EOFError, BrokenPipeError, OSError) as e:
            raise RuntimeError(
                "Worker died before generation could start"
            ) from e

        while True:
            try:
                msg = await asyncio.to_thread(conn.recv)
            except (EOFError, BrokenPipeError, OSError) as e:
                exit_code = (
                    self._process.exitcode if self._process else "unknown"
                )
                raise RuntimeError(
                    f"Worker died during generation (exit code: {exit_code}). "
                    "This may indicate GPU OOM or a model crash."
                ) from e

            kind = msg[0]

            if kind == "chunk":
                _, audio, timestamps = msg
                word_ts = None
                if timestamps:
                    word_ts = [
                        WordTimestamp(word=w, start_time=s, end_time=e)
                        for w, s, e in timestamps
                    ]
                yield AudioChunk(audio, word_timestamps=word_ts)

            elif kind == "done":
                return

            elif kind == "error":
                raise RuntimeError(f"Worker generation error: {msg[1]}")

    async def generate(self, *args, **kwargs):
        """Generate audio.

        Holds the lock for the entire request to prevent concurrent pipe I/O.
        This serializes generation — fine for TTS workloads.
        """
        async with self._lock:
            await self._ensure_worker()
            self._active_requests += 1
            if self._ttl_task and not self._ttl_task.done():
                self._ttl_task.cancel()
                self._ttl_task = None

            try:
                # Unpack the same signature TTSService uses:
                # generate(text, (voice_name, voice_path), speed=, lang_code=, ...)
                text = args[0]
                voice_name, voice_path = args[1]
                speed = kwargs.get("speed", args[2] if len(args) > 2 else 1.0)
                lang_code = kwargs.get(
                    "lang_code", args[3] if len(args) > 3 else None
                )
                return_timestamps = kwargs.get(
                    "return_timestamps", args[4] if len(args) > 4 else False
                )

                async for chunk in self._generate_via_worker(
                    text,
                    voice_name,
                    voice_path,
                    speed,
                    lang_code,
                    return_timestamps,
                ):
                    if settings.default_volume_multiplier != 1.0:
                        chunk.audio *= settings.default_volume_multiplier
                    yield chunk
            finally:
                self._active_requests -= 1
                if self._active_requests == 0:
                    if settings.model_ttl == 0:
                        await self._kill_worker()
                    else:
                        self._reset_ttl_timer()

    # ── TTL ────────────────────────────────────────────────────────────

    def _reset_ttl_timer(self) -> None:
        if self._ttl_task and not self._ttl_task.done():
            self._ttl_task.cancel()
            self._ttl_task = None

        if settings.model_ttl <= 0:
            return

        self._ttl_task = asyncio.create_task(self._ttl_countdown())

    async def _ttl_countdown(self) -> None:
        try:
            await asyncio.sleep(settings.model_ttl)
            async with self._lock:
                if self._active_requests == 0 and self._worker_alive():
                    logger.info(
                        f"Model idle for {settings.model_ttl}s, killing worker..."
                    )
                    await self._kill_worker()
        except asyncio.CancelledError:
            pass

    def unload_all(self) -> None:
        """Synchronous teardown for app shutdown."""
        if self._ttl_task and not self._ttl_task.done():
            self._ttl_task.cancel()
        if self._conn is not None:
            try:
                self._conn.send(None)
                self._conn.close()
            except Exception as e:
                logger.debug(f"unload_all conn cleanup: {e}")
            self._conn = None
        if self._process is not None:
            logger.info(f"unload_all: terminating worker PID {self._process.pid}")
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._process = None


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
