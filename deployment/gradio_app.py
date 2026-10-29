"""Authenticated readiness UI: public predictions are disabled pending release review."""
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


def create_app(*, internal_adapter=None, general_research=None, research_capture=None, private_demo=False):
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
    from .ui.theme import ABOUT_MD, BRAND_CSS, FOOTER_HTML, HOW_HTML, READINESS_HTML, header_html
    from .ui.upload_section import build_upload_section

    live_service = LiveCaptureService()
    uploads = UploadStore(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir()), config.max_upload_bytes)
    live_service.upload_store = uploads
    ui.upload_store = uploads
    if internal_adapter is not None:
        internal_adapter.upload_store = uploads
    if general_research is not None:
        general_research.upload_store = uploads
    model_ready = readiness.inference_enabled
    with gr.Blocks(
        title="AValidate",
        analytics_enabled=False,
    ) as demo:
        header=header_html(model_ready)
        if private_demo:header=header.replace('Model verification unavailable','Private model demonstration')
        gr.HTML(header, head=f"<style>{BRAND_CSS}</style>")
        if private_demo:
            if general_research is None or research_capture is None:
                raise RuntimeError('PRIVATE_DEMO_ADAPTERS_REQUIRED')
            gr.Markdown('**Private authenticated demonstration.** Upload and completed live captures use the '
                'original models in a separate isolated worker. Raw scores are uncalibrated; public release '
                'and scientific numerical acceptance remain unapproved.')
        if not model_ready and not private_demo:
            gr.HTML(READINESS_HTML)
            with gr.Accordion(f"Startup status: {len(categories)} items not ready", open=False):
                gr.Markdown("\n".join(f"- `{item}`" for item in categories))
        with gr.Tabs():
            if private_demo:
                from .ui.general_research_section import build_general_research_section
                with gr.Tab('Verify uploaded video'):
                    build_general_research_section(gr, general_research, private_demo=True)
                with gr.Tab('Verify live capture'):
                    build_live_section(gr, research_capture, research=True)
            else:
                with gr.Tab("Upload verification"):
                    build_upload_section(gr, ui)
                with gr.Tab("Live verification"):
                    build_live_section(gr, live_service)
            with gr.Tab("How It Works"):
                gr.HTML(HOW_HTML)
            with gr.Tab("Model Information"):
                gr.Markdown(ABOUT_MD)
            if internal_adapter is not None:
                from .ui.internal_section import build_internal_section
                with gr.Tab("Internal TRAIN demonstration"):
                    build_internal_section(gr, internal_adapter)
            if general_research is not None and not private_demo:
                from .ui.general_research_section import build_general_research_section
                with gr.Tab("General media research — NOT APPROVED"):
                    build_general_research_section(gr, general_research)
            if research_capture is not None and not private_demo:
                with gr.Tab("Local live research — NOT APPROVED"):
                    build_live_section(gr, research_capture, research=True)
        gr.HTML(FOOTER_HTML)
    demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
    demo.aval_live_service = live_service
    demo.aval_upload_store = uploads
    return demo, config


def main(*, internal_adapter=None, general_research=None) -> None:
    demo, config = create_app(internal_adapter=internal_adapter, general_research=general_research)
    port_text = os.environ.get("PORT", "7860")
    try:
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise RuntimeError("INVALID_SERVER_PORT") from None
    blocked = [str(config.model_root)]
    from .live.security import live_app_kwargs
    from .ui.theme import BRAND_CSS, LOGIN_CSS, LOGIN_HTML
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
            # The pinned login view does not load Blocks' custom CSS. Keep its
            # native form/registry and brand only the trusted server HTML slot.
            auth_message=f"<style>{BRAND_CSS}{LOGIN_CSS}</style>{LOGIN_HTML}",
            css=BRAND_CSS,
            footer_links=[],
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
