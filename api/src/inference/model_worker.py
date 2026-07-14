"""Subprocess worker for TTS inference.

Runs in a separate process so that killing it fully releases the CUDA
context and all GPU memory — unlike in-process unload which leaves
the ~400-500 MB CUDA runtime resident.

Protocol (over multiprocessing.Pipe connection):
    Parent → Child:
        ("generate", req_id, text, voice_name, voice_path, speed, lang_code, return_timestamps)
        None  → shutdown

    Child → Parent:
        ("ready",)
        ("chunk", req_id, audio_ndarray, timestamps_or_none)
        ("done", req_id)
        ("error", req_id, message)

Every reply carries the req_id it belongs to, and every generation ends with
exactly one terminator ("done" or "error").  A generation streams an unbounded
number of chunks, so a parent that stops reading early — the client barged in
and hung up — would otherwise leave the tail of that generation queued in the
pipe, where the *next* request would read it as its own audio.  Tagging replies
lets the parent recognise and discard leftovers instead of playing them.

`cancel_id` is a shared multiprocessing.Value holding the req_id the parent has
given up on.  The worker checks it between chunks and ends that generation
early, so an abandoned request can be drained promptly rather than after the
whole utterance has been synthesised.
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
    cancel_id=None,
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
                _handle_generate(
                    conn, msg, model, device, pipelines, logger, cancel_id
                )

    except Exception as e:
        logger.error(f"[Worker PID {os.getpid()}] Fatal error: {e}", exc_info=True)
        try:
            # req_id 0 is never issued, so the parent discards this rather than
            # mistaking it for a reply to whatever it is currently awaiting; the
            # pipe closing right after is what actually surfaces the failure.
            conn.send(("error", 0, f"Worker fatal: {e}"))
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


def _handle_generate(conn, msg, model, device, pipelines, logger, cancel_id=None):
    """Process a single generate request.

    Always terminated by exactly one ("done", req_id) or ("error", req_id, ...),
    including when the parent cancels — the parent relies on that terminator to
    know the pipe is clean again.
    """
    import os
    import tempfile

    import torch
    from kokoro import KPipeline

    (
        _,
        req_id,
        text,
        voice_name,
        voice_path,
        speed,
        lang_code,
        return_timestamps,
    ) = msg
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
            if cancel_id is not None and cancel_id.value == req_id:
                logger.info(
                    f"[Worker] request {req_id} cancelled by parent, stopping early"
                )
                break
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
            conn.send(("chunk", req_id, result.audio.numpy(), timestamps))

        conn.send(("done", req_id))
    except Exception as e:
        logger.error(f"[Worker] generate failed: {e}", exc_info=True)
        try:
            conn.send(("error", req_id, str(e)))
        except Exception:
            pass
