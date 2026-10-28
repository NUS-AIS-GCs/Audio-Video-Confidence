"""Separate Hugging Face Docker Space UI for readiness and safe upload admission."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import tempfile
import time

from .runtime import (
    DeploymentConfig,
    DeploymentError,
    ReadinessReport,
    startup_readiness,
    validate_upload,
)


LOGGER = logging.getLogger("avalidate.deployment")


class DeploymentUI:
    def __init__(self, config: DeploymentConfig, readiness: ReadinessReport):
        self.config = config
        self.readiness = readiness
        self.upload_store = None

    def submit_owned(self, handle, request):
        """Authenticate before resolving a handle or opening/deleting any media."""
        from .auth import login_owner
        from .uploads import UploadError
        raw = getattr(request, "request", None)
        owner = login_owner(raw)
        if (owner is None or self.upload_store is None
                or raw.scope.get("aval.upload_guard") is not self.upload_store):
            return {"status": "REJECTED", "predictions": None,
                    "error_code": "UPLOAD_AUTHENTICATION_REQUIRED"}
        if not handle:
            return {"status": "REJECTED", "predictions": None, "error_code": "UPLOAD_REQUIRED"}
        try:
            with self.upload_store.lease(handle, owner) as (path, _filename):
                return self.submit(path)
        except UploadError:
            return {"status": "REJECTED", "predictions": None, "error_code": "UPLOAD_NOT_FOUND"}

    def submit(self, upload_path: str | Path | None) -> dict:
        if not self.config.authenticated_uploads_enabled:
            return {"status": "REJECTED", "predictions": None,
                    "error_code": "AUTHENTICATION_CONFIGURATION_REQUIRED"}
        if upload_path is None:
            return {"status": "REJECTED", "predictions": None, "error_code": "UPLOAD_REQUIRED"}

        path = None
        root = Path(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir())).resolve()
        try:
            if not isinstance(upload_path, (str, Path)):
                raise DeploymentError("UNSUPPORTED_UPLOAD")
            path = Path(upload_path).resolve(strict=True)
            path.relative_to(root)
            if not path.is_file():
                raise DeploymentError("UNSUPPORTED_UPLOAD")
            with path.open("rb") as stream:
                content = stream.read(self.config.max_upload_bytes + 1)
            return validate_upload(
                content, filename=path.name, config=self.config, readiness=self.readiness,
            )
        except DeploymentError as exc:
            return {"status": "REJECTED", "predictions": None, "error_code": exc.code}
        except (OSError, ValueError):
            return {"status": "REJECTED", "predictions": None, "error_code": "UNSUPPORTED_UPLOAD"}
        finally:
            if path is not None:
                try:
                    path.relative_to(root)
                    path.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass


def create_app():
    """Build the Gradio app lazily so readiness tests need no Gradio install."""
    try:
        import gradio as gr
    except Exception:
        raise RuntimeError("GRADIO_DEPENDENCY_MISSING") from None

    try:
        config = DeploymentConfig.from_environment()
    except DeploymentError as exc:
        raise RuntimeError(exc.code) from None
    if not config.authenticated_uploads_enabled:
        raise RuntimeError("AUTHENTICATION_CONFIGURATION_REQUIRED") from None
    try:
        readiness = startup_readiness(config)
    except DeploymentError as exc:
        raise RuntimeError(exc.code) from None

    categories = sorted({issue.category for issue in readiness.issues})
    ui = DeploymentUI(config, readiness)
    LOGGER.info("startup_status=%s missing_categories=%s", readiness.status, ",".join(categories))

    from .live.service import LiveCaptureService
    from .uploads import UploadStore
    from .ui.live_section import build_live_section
    from .ui.theme import ABOUT_MD, BRAND_CSS, header_html
    from .ui.upload_section import build_upload_section

    live_service = LiveCaptureService()
    uploads = UploadStore(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir()), config.max_upload_bytes)
    live_service.upload_store = uploads
    ui.upload_store = uploads
    model_ready = readiness.inference_enabled
    with gr.Blocks(
        title="AValidate",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(header_html(model_ready), head=f"<style>{BRAND_CSS}</style>")
        if not model_ready:
            gr.Markdown(
                "**MODEL_NOT_READY. Real inference is disabled.** This interface does not show example, "
                "annotation-derived, or synthetic predictions. A successful upload check is not an "
                "audio-video verification result. Live capture shows buffer and sync diagnostics only; "
                "predictions are disabled.")
            with gr.Accordion(f"Startup status: {len(categories)} items not ready", open=False):
                gr.Markdown("\n".join(f"- `{item}`" for item in categories))
        with gr.Tabs():
            with gr.Tab("Upload verification"):
                build_upload_section(gr, ui)
            with gr.Tab("Live verification"):
                build_live_section(gr, live_service)
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)
                gr.Markdown(
                    "A future inference path must use the verified Phase 6 contracts after exact encoder "
                    "artifacts, decoder behavior, and raw-to-TRAIN export parity are independently verified. "
                    "Scores, if later enabled, remain uncalibrated; no per-second output is inferred from attention.")
    demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
    demo.aval_live_service = live_service
    demo.aval_upload_store = uploads
    return demo, config


def main() -> None:
    demo, config = create_app()
    port_text = os.environ.get("PORT", "7860")
    try:
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise RuntimeError("INVALID_SERVER_PORT") from None
    blocked = [str(config.model_root)]
    from .live.security import live_app_kwargs
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    previous = {kind: signal.signal(kind, stop) for kind in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Gradio's threaded server does not install process signal handlers.
        # Own the main loop so Docker SIGTERM reaches application lifespans.
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            auth=config.gradio_auth,
            show_error=False,
            enable_monitoring=False,
            max_file_size=config.max_upload_bytes,
            blocked_paths=blocked,
            app_kwargs=live_app_kwargs(demo.aval_live_service),
            prevent_thread_lock=True,
        )
        while getattr(demo, "is_running", False) and not stopping:
            time.sleep(0.1)
    finally:
        try:
            demo.close()
        finally:
            demo.aval_live_service.close()
            demo.aval_upload_store.close()
            # Gradio joins for five seconds; permit bounded body readers to
            # unwind their finally blocks before this process exits.
            thread = getattr(getattr(demo, "server", None), "thread", None)
            if thread is not None:
                thread.join(timeout=35)
            for kind, handler in previous.items():
                signal.signal(kind, handler)


if __name__ == "__main__":
    main()
