"""Kokoro V1 model management with idle TTL support."""

import asyncio
from typing import Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None  # Explicitly type as KokoroV1
        self._device: Optional[str] = None
        self._ttl_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._active_requests = 0

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize Kokoro V1 backend."""
        try:
            self._device = self._determine_device()
            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = KokoroV1()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Load model
            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                # Warm up with short text — call backend directly to avoid
                # starting the TTL timer prematurely during warmup
                warmup_text = "Warmup text for initialization."
                voice_name = settings.default_voice
                logger.debug(f"Using default voice '{voice_name}' for warmup")
                async for _ in self._backend.generate(
                    warmup_text, (voice_name, voice_path)
                ):
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            self._reset_ttl_timer()

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def _ensure_loaded(self) -> None:
        """Reload model if it was unloaded due to TTL."""
        if self._backend and self._backend.is_loaded:
            return

        logger.info("Model was unloaded, reloading...")
        import time
        start = time.perf_counter()

        if not self._backend:
            await self.initialize()

        model_path = self._config.pytorch_kokoro_v1_file
        await self.load_model(model_path)

        ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"Model reloaded in {ms}ms")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        async with self._lock:
            await self._ensure_loaded()
            self._active_requests += 1
            # Cancel any pending unload while requests are active
            if self._ttl_task and not self._ttl_task.done():
                self._ttl_task.cancel()
                self._ttl_task = None

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                if settings.default_volume_multiplier != 1.0:
                    chunk.audio *= settings.default_volume_multiplier
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
        finally:
            async with self._lock:
                self._active_requests -= 1
                if self._active_requests == 0:
                    if settings.model_ttl == 0:
                        await self._do_unload()
                    else:
                        self._reset_ttl_timer()

    def _reset_ttl_timer(self) -> None:
        """Reset the idle TTL timer. -1 = never unload."""
        if self._ttl_task and not self._ttl_task.done():
            self._ttl_task.cancel()
            self._ttl_task = None

        if settings.model_ttl <= 0:
            return

        self._ttl_task = asyncio.create_task(self._ttl_countdown())

    async def _ttl_countdown(self) -> None:
        """Wait for TTL seconds, then unload the model."""
        try:
            await asyncio.sleep(settings.model_ttl)
            async with self._lock:
                if self._active_requests == 0:
                    logger.info(
                        f"Model idle for {settings.model_ttl}s, unloading from GPU..."
                    )
                    await self._do_unload()
        except asyncio.CancelledError:
            pass

    async def _do_unload(self) -> None:
        """Unload model and free GPU memory in a thread to avoid blocking the event loop."""
        if self._backend and self._backend.is_loaded:
            await asyncio.to_thread(self._backend.unload)
            logger.info("Model unloaded, GPU memory freed")

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._ttl_task and not self._ttl_task.done():
            self._ttl_task.cancel()
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
