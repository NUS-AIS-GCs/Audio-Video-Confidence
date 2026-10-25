"""Online-only CoreWeave Forge tracking, with nonsecret failure reporting."""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from .artifact_integration import _json
from .baseline_tracking import TrackingUnavailable, safe_failure
from .training_contracts import FORGE_BASE_URL, FORGE_ENTITY, FORGE_PROJECT

GROUP = "phase4_learned_fusion_mlp"
STUDY_GROUP = "phase4_mlp_regularization_study"
EPOCH_KEYS = {
    "train_loss": "train/loss", "val_loss": "val/loss", "val_roc_auc": "val/roc_auc",
    "val_average_precision": "val/average_precision", "val_accuracy_at_0_5": "val/accuracy_at_0_5",
    "val_f1_at_0_5": "val/f1_at_0_5", "learning_rate": "optimizer/learning_rate",
    "gradient_norm_mean": "train/gradient_norm_mean", "gradient_norm_max": "train/gradient_norm_max",
    "duration_seconds": "epoch/duration_seconds",
}


@contextlib.contextmanager
def _quiet_sdk():
    # Never copy arbitrary SDK messages/headers/authentication exceptions to reports.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


class ForgeMLPTracker:
    def __init__(self, output):
        self.output = Path(output)
        self.sdk = self.api = self.run = None
        self._terminal = None
        self._abort_attempted = False
        self._success_finalization_attempted = False
        self.metadata = {"status": "NOT_STARTED", "base_url": FORGE_BASE_URL,
                         "entity": FORGE_ENTITY, "project": FORGE_PROJECT, "group": GROUP}

    def preflight_access(self):
        """Read-only authentication/project access check; creates no run."""
        if os.environ.get("WANDB_MODE", "online") != "online":
            raise TrackingUnavailable("WANDB_MODE must be online")
        for key, expected in (("WANDB_BASE_URL", FORGE_BASE_URL), ("WANDB_ENTITY", FORGE_ENTITY),
                              ("WANDB_PROJECT", FORGE_PROJECT)):
            if os.environ.get(key, expected).rstrip("/") != expected:
                raise TrackingUnavailable(f"{key} conflicts with required Forge destination")
        try:
            with _quiet_sdk():
                import wandb
                from wandb.sdk.lib import wbauth
                auth = wbauth.authenticate_session(host=FORGE_BASE_URL, source="gc2-phase4",
                                                  prompt=False, verify=False, update_session=False)
                if auth is None:
                    raise TrackingUnavailable("no configured credentials for CoreWeave Forge; no training started")
                if auth.host.url.rstrip("/") != FORGE_BASE_URL:
                    raise TrackingUnavailable("configured credentials belong to a different host")
                api = wandb.Api(overrides={"base_url": FORGE_BASE_URL, "entity": FORGE_ENTITY,
                                          "project": FORGE_PROJECT}, timeout=20)
                if api.viewer is None or api.project(FORGE_PROJECT, entity=FORGE_ENTITY) is None:
                    raise TrackingUnavailable("Forge viewer/project access is unavailable")
                self.sdk, self.api = wandb, api
            return {"status": "AUTHENTICATED_PROJECT_ACCESS", "base_url": FORGE_BASE_URL,
                    "entity": FORGE_ENTITY, "project": FORGE_PROJECT,
                    "sdk_version": importlib.metadata.version("wandb"), "run_created": False}
        except TrackingUnavailable:
            raise
        except Exception as exc:
            raise TrackingUnavailable(safe_failure("Forge authenticated access", exc)) from None

    def start(self, config):
        group = config.get("wandb_group", GROUP)
        name = config.get("wandb_name") or f"phase4-mlp-seed{config['training']['seed']}-initial"
        if group not in (GROUP, STUDY_GROUP) or not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name) is None:
            raise TrackingUnavailable("invalid prespecified Forge group or run name")
        self.preflight_access()
        try:
            with _quiet_sdk():
                self.run = self.sdk.init(
                    entity=FORGE_ENTITY, project=FORGE_PROJECT, group=group,
                    job_type="learned-correspondence-training",
                    name=name,
                    dir=str(self.output), mode="online", config=config, save_code=False,
                    settings=self.sdk.Settings(base_url=FORGE_BASE_URL, init_timeout=60, login_timeout=15,
                                               disable_code=True, disable_git=True, x_disable_stats=True,
                                               x_disable_meta=True, x_disable_machine_info=True, quiet=True))
                if (self.run is None or self.run.settings.mode != "online"
                        or self.run.settings.base_url.rstrip("/") != FORGE_BASE_URL
                        or self.run.entity != FORGE_ENTITY or self.run.project != FORGE_PROJECT):
                    raise TrackingUnavailable("SDK did not create the required online Forge run")
                host = urlparse(self.run.url).hostname or ""
                if host != "forge.coreweave.com" and not host.endswith(".forge.coreweave.com"):
                    raise TrackingUnavailable("returned run URL is outside CoreWeave Forge")
                self.metadata.update(run_id=self.run.id, run_url=self.run.url, status="RUN_STARTED", group=group,
                                     sdk_version=importlib.metadata.version("wandb"))
                self.run.define_metric("epoch")
                for name in EPOCH_KEYS.values():
                    self.run.define_metric(name, step_metric="epoch")
                self.run.log({"tracking/ready": 1}, step=0)
                probe_path = self.output / "tracking_probe.json"
                probe_path.write_text(_json({"schema": "gc2-forge-write-probe-v1", "ready": True}), encoding="utf-8")
                probe = self.sdk.Artifact("phase4-mlp-online-readiness", type="verification")
                probe.add_file(probe_path, name=probe_path.name)
                uploaded = self.run.log_artifact(probe)
                uploaded.wait(timeout=180)
                # Read back the uploaded version, proving authenticated writes
                # completed before any real optimizer step is allowed.
                remote = self.api.artifact(uploaded.qualified_name)
                if remote.digest != uploaded.digest:
                    raise TrackingUnavailable("Forge readiness artifact read-back digest mismatch")
                self.metadata.update(status="ONLINE_WRITE_VERIFIED",
                                     readiness_artifact=uploaded.qualified_name)
            return dict(self.metadata)
        except TrackingUnavailable:
            raise
        except Exception as exc:
            raise TrackingUnavailable(safe_failure("Forge online write preflight", exc)) from None

    def log_epoch(self, row):
        try:
            values = {tag: row[key] for key, tag in EPOCH_KEYS.items()}
            if any(not math.isfinite(float(v)) for v in values.values()):
                raise TrackingUnavailable("nonfinite epoch metric cannot be logged")
            with _quiet_sdk():
                self.run.log({"epoch": row["epoch"], **values}, step=row["epoch"])
        except TrackingUnavailable:
            raise
        except Exception as exc:
            raise TrackingUnavailable(safe_failure("Forge epoch logging", exc)) from None

    def finish(self, final_metrics, files, history, manifest_sha256, *, before_finalize=None):
        if self._terminal is not None:
            raise TrackingUnavailable("Forge tracker has already terminated")
        try:
            with _quiet_sdk():
                self.run.log(final_metrics, step=len(history)+1)
                self.run.summary.update(final_metrics)
                self.run.summary["reproducibility_manifest_sha256"] = manifest_sha256
                uploaded_metadata = []
                for name, kind, chosen in (
                    ("phase4-mlp-selected-checkpoint", "model", [p for p in files if p.name in
                        ("best.pt", "checkpoint_metadata.json", "frozen_operating_point.json")]),
                    ("phase4-mlp-reproducibility-reports", "evaluation", [p for p in files if p.name != "best.pt"]),
                ):
                    artifact = self.sdk.Artifact(name, type=kind,
                                                metadata={"reproducibility_manifest_sha256": manifest_sha256})
                    for path in chosen:
                        if not path.is_file() or not path.resolve().is_relative_to(self.output.resolve()):
                            raise TrackingUnavailable("artifact path is outside the experiment output")
                        artifact.add_file(path, name=path.relative_to(self.output).as_posix())
                    uploaded = self.run.log_artifact(artifact)
                    uploaded.wait(timeout=180)
                    uploaded_metadata.append({"name": uploaded.qualified_name, "digest": uploaded.digest,
                                              "url": getattr(uploaded, "url", None)})
                # Active runs are eventually consistent. Refresh public API
                # objects and bypass history caches, with a bounded fail-closed
                # retry. Never finalize successfully just to flush the SDK.
                for attempt in range(3):
                    try:
                        rows = self._verify_remote(final_metrics, history, manifest_sha256, uploaded_metadata)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(1)
                if before_finalize is not None:
                    before_finalize()
                self._success_finalization_attempted = True
                self.run.finish(exit_code=0)
            self._terminal = "COMPLETE"
            self.metadata.update(status="COMPLETE", artifacts=uploaded_metadata,
                                 remote_history_epochs_verified=len(rows), remote_summary_verified=True)
            return dict(self.metadata)
        except BaseException as exc:
            self.abort()
            if not isinstance(exc, Exception) or isinstance(exc, TrackingUnavailable):
                raise
            raise TrackingUnavailable(safe_failure("Forge final upload/read-back", exc)) from None

    def _verify_remote(self, final_metrics, history, manifest_sha256, uploaded_metadata):
        # Public Run.summary caches an HTTPSummary even after load(force=True).
        # Flush the API's run cache so each retry gets a fresh summary object.
        self.api.flush()
        remote = self.api.run(f"{FORGE_ENTITY}/{FORGE_PROJECT}/{self.run.id}")
        remote.load(force=True)
        for key, value in {**final_metrics, "reproducibility_manifest_sha256": manifest_sha256}.items():
            actual = remote.summary.get(key)
            if type(value) in (float, int):
                if actual is None or not math.isclose(float(actual), float(value), rel_tol=1e-10, abs_tol=1e-12):
                    raise TrackingUnavailable("Forge final metric read-back mismatch")
            elif actual != value:
                raise TrackingUnavailable("Forge final summary read-back mismatch")
        rows = list(remote.scan_history(keys=["epoch", *EPOCH_KEYS.values()], page_size=1000, use_cache=False))
        if len(rows) != len(history):
            raise TrackingUnavailable("Forge epoch history count differs from saved history")
        for row, expected in zip(rows, history):
            if row["epoch"] != expected["epoch"] or any(
                not math.isclose(float(row[tag]), float(expected[key]), rel_tol=1e-10, abs_tol=1e-12)
                for key, tag in EPOCH_KEYS.items()
            ):
                raise TrackingUnavailable("Forge epoch curve read-back mismatch")
        for evidence in uploaded_metadata:
            if self.api.artifact(evidence["name"]).digest != evidence["digest"]:
                raise TrackingUnavailable("Forge final artifact digest mismatch")
        return rows

    def abort(self):
        if self._terminal == "COMPLETE" or self._abort_attempted:
            return
        self._abort_attempted = True
        self._terminal = "INCOMPLETE"
        self.metadata.update(status="INCOMPLETE", finalization_state=(
            "UNKNOWN_AFTER_SUCCESS_FINALIZATION_ERROR" if self._success_finalization_attempted
            else "FAILURE_REQUESTED"))
        if self.run is not None:
            try:
                with _quiet_sdk():
                    self.run.finish(exit_code=1)
            except Exception:
                self.metadata["finalization_state"] = "UNKNOWN_AFTER_ABORT_ERROR"
