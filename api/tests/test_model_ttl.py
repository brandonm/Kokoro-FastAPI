"""Tests for model TTL (idle unload) functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.inference.model_manager import ModelManager


@pytest.fixture
def manager():
    """Create a fresh ModelManager for each test."""
    # Reset singleton
    ModelManager._instance = None
    mgr = ModelManager()
    mgr._device = "cuda"
    # Mock backend
    backend = MagicMock()
    backend.is_loaded = True
    backend.unload = MagicMock()

    async def mock_generate(*args, **kwargs):
        from api.src.inference.base import AudioChunk
        import numpy as np
        yield AudioChunk(np.zeros(100, dtype=np.float32))

    backend.generate = mock_generate
    mgr._backend = backend
    yield mgr
    # Cleanup any pending tasks
    if mgr._ttl_task and not mgr._ttl_task.done():
        mgr._ttl_task.cancel()
    ModelManager._instance = None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_unloads_after_idle(mock_settings, manager):
    """Model should be unloaded after TTL seconds of inactivity."""
    mock_settings.model_ttl = 0.2
    mock_settings.default_volume_multiplier = 1.0

    async for _ in manager.generate("test", ("voice", "/path")):
        pass

    # Wait well past TTL
    await asyncio.sleep(0.5)

    manager._backend.unload.assert_called_once()


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_resets_on_generate(mock_settings, manager):
    """Each generate call should reset the TTL timer."""
    mock_settings.model_ttl = 0.5
    mock_settings.default_volume_multiplier = 1.0

    # First request
    async for _ in manager.generate("test", ("voice", "/path")):
        pass

    # Wait 300ms (< 500ms TTL), then make another request
    await asyncio.sleep(0.3)
    async for _ in manager.generate("test", ("voice", "/path")):
        pass

    # 300ms after second request — still within TTL
    await asyncio.sleep(0.3)
    manager._backend.unload.assert_not_called()

    # Wait for full TTL after last request
    await asyncio.sleep(0.4)
    manager._backend.unload.assert_called_once()


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_negative_never_unloads(mock_settings, manager):
    """TTL of -1 should never unload the model."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0

    async for _ in manager.generate("test", ("voice", "/path")):
        pass

    await asyncio.sleep(0.2)

    manager._backend.unload.assert_not_called()
    assert manager._ttl_task is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ttl_zero_unloads_immediately(mock_settings, manager):
    """TTL of 0 should unload immediately after each request."""
    mock_settings.model_ttl = 0
    mock_settings.default_volume_multiplier = 1.0

    async for _ in manager.generate("test", ("voice", "/path")):
        pass

    # _do_unload is async (to_thread), give it a tick
    await asyncio.sleep(0.05)

    manager._backend.unload.assert_called_once()


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_ensure_loaded_reloads_after_unload(mock_settings, manager):
    """After TTL unloads the model, next request should reload it."""
    mock_settings.model_ttl = 0.2
    mock_settings.default_volume_multiplier = 1.0
    mock_settings.use_gpu = True

    # Trigger generate, then wait for unload
    async for _ in manager.generate("test", ("voice", "/path")):
        pass
    await asyncio.sleep(0.5)
    manager._backend.unload.assert_called_once()

    # Simulate unloaded state
    manager._backend.is_loaded = False

    # Mock load_model to track reload
    manager.load_model = AsyncMock()

    await manager._ensure_loaded()

    manager.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_unload_all_cancels_timer(manager):
    """unload_all() should cancel any pending TTL timer."""
    manager._ttl_task = asyncio.create_task(asyncio.sleep(100))

    manager.unload_all()

    assert manager._ttl_task.cancelled() or manager._ttl_task.done()
    assert manager._backend is None


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_concurrent_requests_prevent_unload(mock_settings, manager):
    """Model should not unload while any request is still streaming."""
    mock_settings.model_ttl = 0
    mock_settings.default_volume_multiplier = 1.0

    # Slow backend that yields two chunks with a delay between them
    async def slow_generate(*args, **kwargs):
        from api.src.inference.base import AudioChunk
        import numpy as np
        yield AudioChunk(np.zeros(100, dtype=np.float32))
        await asyncio.sleep(0.3)
        yield AudioChunk(np.zeros(100, dtype=np.float32))

    manager._backend.generate = slow_generate

    async def consume_slow():
        async for _ in manager.generate("test", ("voice", "/path")):
            pass

    # Start two concurrent requests
    task1 = asyncio.create_task(consume_slow())
    task2 = asyncio.create_task(consume_slow())

    # Let them start streaming
    await asyncio.sleep(0.1)

    # Both are mid-stream — model should NOT be unloaded yet
    manager._backend.unload.assert_not_called()

    # Wait for both to finish
    await task1
    await task2
    await asyncio.sleep(0.05)

    # Now that both are done, model should be unloaded (ttl=0)
    manager._backend.unload.assert_called_once()


@pytest.mark.asyncio
@patch("api.src.inference.model_manager.settings")
async def test_active_requests_tracked_correctly(mock_settings, manager):
    """Active request count should increment and decrement properly."""
    mock_settings.model_ttl = -1
    mock_settings.default_volume_multiplier = 1.0

    assert manager._active_requests == 0

    # Start a request but don't finish consuming
    gen = manager.generate("test", ("voice", "/path"))
    chunk = await gen.__anext__()
    assert manager._active_requests == 1

    # Finish consuming
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        pass
    # finally block in generate runs on StopAsyncIteration
    await asyncio.sleep(0.01)

    assert manager._active_requests == 0
