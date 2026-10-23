"""Optional Forge W&B adapter. Never serialize credentials or exception text."""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import os
from pathlib import Path
from urllib.parse import urlparse

BASE_URL = "https://api.forge.coreweave.com"
ENTITY = "GC2-AValidate"
PROJECT = "avalidate"
GROUP = "phase3_correspondence_baselines"


class TrackingUnavailable(RuntimeError):
    """Only fixed, nonsecret blocker codes may enter persisted evidence."""


def safe_failure(stage, exc):
    # Do not persist SDK exception messages, request headers or environment dumps.
    return f"{stage}: {type(exc).__name__}; SDK exception text withheld for credential safety"


class ForgeTracker:
    def __init__(self, output_dir):
        self.output = Path(output_dir)
        self.sdk = self.run = None
        self.metadata = {"status": "BLOCKED_WANDB_UPLOAD", "base_url": BASE_URL,
                         "entity": ENTITY, "project": PROJECT, "group": GROUP}

    def start(self, config):
        if os.environ.get("WANDB_MODE", "online") != "online":
            raise TrackingUnavailable("WANDB_MODE must be online; no offline run can count as Forge upload")
        for name, expected in (("WANDB_BASE_URL", BASE_URL), ("WANDB_ENTITY", ENTITY), ("WANDB_PROJECT", PROJECT)):
            if os.environ.get(name, expected).rstrip("/") != expected:
                raise TrackingUnavailable(f"{name} conflicts with mandatory GC2 destination")
        try:
            # Capture SDK console output; never copy it into reports on failure.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                import wandb
                from wandb.sdk.lib import wbauth

                self.sdk = wandb
                auth = wbauth.authenticate_session(host=BASE_URL, source="gc2-baselines",
                                                  prompt=False, verify=False, update_session=False)
                if auth is None:
                    raise TrackingUnavailable("No configured credentials for CoreWeave Forge; interactive login is required")
                if auth.host.url.rstrip("/") != BASE_URL:
                    raise TrackingUnavailable("Configured authentication belongs to a different W&B host")
                api = wandb.Api(overrides={"base_url": BASE_URL, "entity": ENTITY, "project": PROJECT}, timeout=15)
                if api.viewer is None or api.project(PROJECT, entity=ENTITY) is None:
                    raise TrackingUnavailable("Forge authentication or required project access is unavailable")
                self.run = wandb.init(
                    entity=ENTITY, project=PROJECT, group=GROUP, job_type="deterministic-baseline-comparison",
                    name="phase3-train-val-correspondence", dir=str(self.output), mode="online",
                    config=config, save_code=False,
                    settings=wandb.Settings(base_url=BASE_URL, init_timeout=45, login_timeout=15,
                                            disable_code=True, disable_git=True, x_disable_stats=True,
                                            x_disable_meta=True, x_disable_machine_info=True, quiet=True),
                )
            if (self.run is None or self.run.settings.mode != "online"
                    or self.run.settings.base_url.rstrip("/") != BASE_URL
                    or self.run.entity != ENTITY or self.run.project != PROJECT):
                raise TrackingUnavailable("W&B did not create an online run at the required Forge destination")
            url = self.run.url
            host = urlparse(url).hostname or ""
            if host != "forge.coreweave.com" and not host.endswith(".forge.coreweave.com"):
                raise TrackingUnavailable("W&B returned a run URL outside CoreWeave Forge")
            self.metadata.update(run_id=self.run.id, run_url=url, sdk_version=importlib.metadata.version("wandb"),
                                 status="RUN_STARTED")
        except TrackingUnavailable:
            raise
        except Exception as exc:
            raise TrackingUnavailable(safe_failure("Forge initialization", exc)) from None

    def upload(self, metrics, comparison, files, manifest_hash):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.run.log(metrics)
                self.run.log({"method_comparison": self.sdk.Table(dataframe=None,
                    columns=list(comparison[0]), data=[list(row.values()) for row in comparison])})
                self.run.summary.update(metrics)
                self.run.summary["core_output_manifest_sha256"] = manifest_hash
                artifact = self.sdk.Artifact("phase3-correspondence-baseline-evidence", type="evaluation",
                                             metadata={"core_output_manifest_sha256": manifest_hash})
                for path in files:
                    artifact.add_file(str(path), name=path.relative_to(self.output).as_posix())
                logged = self.run.log_artifact(artifact)
                logged.wait(timeout=180)
                self.metadata.update(artifact_name=logged.qualified_name,
                                     artifact_digest=logged.digest,
                                     artifact_url=getattr(logged, "url", None))
                self.run.finish(exit_code=0)
            self.metadata["status"] = "COMPLETE"
            return dict(self.metadata)
        except Exception as exc:
            raise TrackingUnavailable(safe_failure("Forge metric/artifact upload", exc)) from None

    def abort(self):
        if self.run is not None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    self.run.finish(exit_code=1)
                except Exception:
                    pass


__all__ = ["BASE_URL", "ENTITY", "PROJECT", "GROUP", "ForgeTracker", "TrackingUnavailable"]
