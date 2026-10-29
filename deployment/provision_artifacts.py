"""Atomic, private runtime artifact provisioning. Does not enable model inference.

Original digest/size catalog is code-bound. Remote sources must be a private HF
repository at a full commit SHA. Tokens are read from an environment Secret and
passed to the pinned SDK; never accepted in URLs, CLI flags or output manifests.
The local-source mode is useful for offline, authenticated deployment staging.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from deployment.train_manifest import pinned_child


CATALOG_SHA256 = "a836d05cd9071282965578c964d3d555405ac02a2f631a9beb2539381bc95418"


class ProvisionError(Exception):
    pass


def artifact_catalog(include_a2=False):
    if type(include_a2) is not bool:
        raise ProvisionError("INVALID_ARTIFACT_PROFILE")
    payload = Path(__file__).with_name("artifact_catalog.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != CATALOG_SHA256:
        raise ProvisionError("ARTIFACT_CATALOG_MODIFIED")
    catalog = json.loads(payload)
    return {n: r for n, r in catalog["files"].items() if include_a2 or not n.startswith("A2/")}


def verify_artifacts(root, *, include_a2=False):
    root = Path(root).resolve()
    files = artifact_catalog(include_a2)
    missing, modified = [], []
    for name, binding in files.items():
        try:
            path = pinned_child(root, name)
            if not path.is_file():
                missing.append(name)
            elif path.stat().st_size != binding["bytes"]:
                modified.append(name)
            else:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != binding["sha256"]:
                        modified.append(name)
        except Exception:
            modified.append(name)
    return {"schema": "aval.artifact-inventory.v1", "ready": not missing and not modified,
            "missing": missing, "modified": modified, "files": len(files),
            "required_bytes": sum(r["bytes"] for r in files.values()), "inference_enabled": False}


def _copy_bounded(source, target, binding):
    digest, copied = hashlib.sha256(), 0
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
        while True:
            chunk = incoming.read(min(1024 * 1024, binding["bytes"] - copied + 1))
            if not chunk:
                break
            copied += len(chunk)
            if copied > binding["bytes"]:
                raise ProvisionError("ARTIFACT_SIZE_MISMATCH")
            digest.update(chunk)
            outgoing.write(chunk)
    target.chmod(0o600)
    if copied != binding["bytes"] or digest.hexdigest() != binding["sha256"]:
        raise ProvisionError("ARTIFACT_HASH_MISMATCH")


def _private_hf_source(repo_id, revision, token, files):
    if (type(repo_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo_id)
            or type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or type(token) is not str or not token.strip()):
        raise ProvisionError("PRIVATE_REPOSITORY_COMMIT_AND_SECRET_REQUIRED")
    if importlib.metadata.version("huggingface_hub") != "2.3.0":
        raise ProvisionError("ARTIFACT_SDK_VERSION_MISMATCH")
    try:
        from huggingface_hub import HfApi
        info = HfApi(endpoint="https://huggingface.co", token=token).repo_info(
            repo_id, repo_type="model", revision=revision, files_metadata=True)
        if info.private is not True or info.sha != revision:
            raise ProvisionError("PRIVATE_IMMUTABLE_REPOSITORY_REQUIRED")
        sizes = {s.rfilename: s.size for s in info.siblings}
        if any(sizes.get(name) != record["bytes"] for name, record in files.items()):
            raise ProvisionError("REMOTE_ARTIFACT_INVENTORY_MISMATCH")
    except ProvisionError:
        raise
    except Exception:
        raise ProvisionError("PRIVATE_ARTIFACT_SOURCE_UNAVAILABLE") from None


def provision_artifacts(destination, *, source_root=None, repo_id=None, revision=None,
                        token=None, include_a2=False):
    """Reuse a fully verified directory or publish a complete new one atomically."""
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ProvisionError("INVALID_ARTIFACT_DESTINATION")
    destination = destination.resolve()
    files = artifact_catalog(include_a2)
    if destination.exists():
        report = verify_artifacts(destination, include_a2=include_a2)
        if not report["ready"]:
            raise ProvisionError("EXISTING_ARTIFACTS_MISSING_OR_MODIFIED")
        return report
    if source_root is not None:
        source_root = Path(source_root).resolve()
        if repo_id is not None or source_root == destination or source_root.is_relative_to(destination) or destination.is_relative_to(source_root):
            raise ProvisionError("INVALID_ARTIFACT_SOURCE")
    elif repo_id is None:
        raise ProvisionError("ARTIFACT_SOURCE_REQUIRED")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    required = sum(r["bytes"] for r in files.values())
    if shutil.disk_usage(destination.parent).free < 2 * required + 32 * 1024 * 1024:
        raise ProvisionError("INSUFFICIENT_ARTIFACT_DISK")
    if source_root is None:
        _private_hf_source(repo_id, revision, token, files)
    try:
        # No partially populated destination is ever visible to a model loader.
        with tempfile.TemporaryDirectory(prefix=".aval-provision-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "models"
            staging.mkdir(mode=0o700)
            for name, binding in files.items():
                target = pinned_child(staging, name)
                if source_root is not None:
                    _copy_bounded(pinned_child(source_root, name), target, binding)
                else:
                    from huggingface_hub import hf_hub_download
                    with tempfile.TemporaryDirectory(dir=temporary) as cache:
                        downloaded = hf_hub_download(repo_id=repo_id, filename=name, revision=revision,
                            repo_type="model", token=token, endpoint="https://huggingface.co",
                            cache_dir=cache, etag_timeout=10)
                        _copy_bounded(downloaded, target, binding)
            report = verify_artifacts(staging, include_a2=include_a2)
            if not report["ready"]:
                raise ProvisionError("ARTIFACT_VERIFICATION_FAILED")
            staging.rename(destination)
        return report
    except ProvisionError:
        raise
    except Exception:
        raise ProvisionError("ARTIFACT_PROVISIONING_FAILED") from None


def runtime_bindings(root):
    root = Path(root).resolve()
    files = artifact_catalog(True)
    aliases = {"AUDIO_BASE": "audio/encoder.pt", "AUDIO_TRAINED": "audio/weights.pt",
               "AUDIO_NORMALIZATION": "audio/norm_stats.json", "VISION_TRAINED": "video/weights.pt"}
    env = {"AVAL_MODEL_ROOT": str(root), "AVAL_E3_DIR": str(root / "E3"), "AVAL_A2_DIR": str(root / "A2")}
    for role, name in aliases.items():
        env[f"AVAL_{role}_PATH"] = str(root / name)
        env[f"AVAL_{role}_SHA256"] = files[name]["sha256"]
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--include-a2", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    try:
        if args.verify_only:
            report = verify_artifacts(args.destination, include_a2=args.include_a2)
        else:
            report = provision_artifacts(args.destination, source_root=args.source_root, repo_id=args.repo,
                revision=args.revision, token=os.environ.get("AVAL_ARTIFACT_READ_TOKEN"), include_a2=args.include_a2)
        print(json.dumps(report))
        if not report["ready"]:
            raise SystemExit(1)
    except Exception:
        # SDK exceptions may include URLs, paths and request details: never echo them.
        raise SystemExit("PRIVATE_ARTIFACT_PROVISIONING_FAILED") from None


if __name__ == "__main__":
    main()
