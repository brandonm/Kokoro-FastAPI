"""Subprocess worker for TTS inference.

Runs in a separate process so that killing it fully releases the CUDA
context and all GPU memory — unlike in-process unload which leaves
the ~400-500 MB CUDA runtime resident.

Protocol (over multiprocessing.Pipe connection):
    Parent → Child:
        ("generate", text, voice_name, voice_path, speed, lang_code, return_timestamps)
        None  → shutdown

    Child → Parent:
        ("ready",)
        ("chunk", audio_ndarray, timestamps_or_none)
        ("done",)
        ("error", message)
"""

import gc
import multiprocessing as mp
import os
import tempfile


def worker_entry(
    conn: mp.connection.Connection,
    model_path: str,
    config_path: str,
    device: str,
):
    """Subprocess entry point. Loads model, handles requests, exits on shutdown."""
    import torch
    from loguru import logger

    pipelines: dict = {}
    model = None

    try:
        logger.info(f"[Worker PID {os.getpid()}] Loading model on {device}")
        model = KModel_load(config_path, model_path, device)

        conn.send(("ready",))
        logger.info(f"[Worker PID {os.getpid()}] Ready")

        while True:
            try:
                if not conn.poll(1.0):
                    # Check if parent is still alive
                    if os.getppid() == 1:
                        logger.warning(
                            f"[Worker PID {os.getpid()}] Parent died, shutting down"
                        )
                        break
                    continue
                msg = conn.recv()
            except (EOFError, OSError) as e:
                logger.warning(
                    f"[Worker PID {os.getpid()}] Pipe closed ({type(e).__name__}), shutting down"
                )
                break

            if msg is None:
                logger.info(f"[Worker PID {os.getpid()}] Shutdown signal received")
                break

            cmd = msg[0]
            if cmd == "generate":
                _handle_generate(conn, msg, model, device, pipelines, logger)

    except Exception as e:
        logger.error(f"[Worker PID {os.getpid()}] Fatal error: {e}", exc_info=True)
        try:
            conn.send(("error", f"Worker fatal: {e}"))
        except Exception as send_err:
            logger.error(
                f"[Worker PID {os.getpid()}] Could not send error to parent: {send_err}"
            )
    finally:
        logger.info(f"[Worker PID {os.getpid()}] Cleaning up")
        del model
        pipelines.clear()
        gc.collect()
        try:
            conn.close()
        except Exception:
            pass


def KModel_load(config_path: str, model_path: str, device: str):
    """Load the Kokoro model on the specified device."""
    import torch
    from kokoro import KModel

    model = KModel(config=config_path, model=model_path)
    model.train(False)  # inference mode
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to(torch.device("mps"))
    return model


def _handle_generate(conn, msg, model, device, pipelines, logger):
    """Process a single generate request."""
    import os
    import tempfile

    import torch
    from kokoro import KPipeline

    _, text, voice_name, voice_path, speed, lang_code, return_timestamps = msg
    try:
        # Device-mapped voice loading (mirrors KokoroV1.generate)
        voice_tensor = torch.load(
            voice_path, map_location=device, weights_only=True
        )
        tmp = os.path.join(
            tempfile.gettempdir(),
            f"wkr_{os.path.basename(voice_path)}",
        )
        torch.save(voice_tensor, tmp)

        if lang_code not in pipelines:
            pipelines[lang_code] = KPipeline(
                lang_code=lang_code, model=model, device=device
            )

        for result in pipelines[lang_code](
            text, voice=tmp, speed=speed, model=model
        ):
            if result.audio is None:
                continue
            timestamps = None
            if (
                return_timestamps
                and hasattr(result, "tokens")
                and result.tokens
                and hasattr(result, "pred_dur")
                and result.pred_dur is not None
            ):
                timestamps = [
                    (
                        str(t.text).strip(),
                        float(t.start_ts),
                        float(t.end_ts),
                    )
                    for t in result.tokens
                    if all(
                        hasattr(t, a) for a in ("text", "start_ts", "end_ts")
                    )
                    and t.text
                    and t.text.strip()
                ]
            conn.send(("chunk", result.audio.numpy(), timestamps))

        conn.send(("done",))
    except Exception as e:
        logger.error(f"[Worker] generate failed: {e}", exc_info=True)
        try:
            conn.send(("error", str(e)))
        except Exception:
            pass
