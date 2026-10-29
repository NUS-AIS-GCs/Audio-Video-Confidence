"""Assemble a hash-inventoried gated Space bundle; never publish or alter Git.

Run from the repository root. Explicit new source names support review before a
commit; unknown untracked files, checkpoints, secrets and media are never copied.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


NEW_SOURCES = {"deployment/prediction_adapter.py", "deployment/ui/internal_section.py",
               "deployment/release_gate.py", "deployment/provision_artifacts.py", "deployment/provisioned_app.py"}


def assemble(destination: Path):
    root = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    destination.relative_to(root / "_local")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines())
    candidates = tracked | NEW_SOURCES
    modules = {n[:-3].replace("/", ".").removesuffix(".__init__"): n for n in candidates
               if n.endswith(".py") and n.split("/")[0] in {"deployment", "audio", "vision", "fusion"}}
    pending, seen, paths = ["deployment.gradio_app", "deployment.provisioned_app"], set(), set()
    while pending:
        module = pending.pop()
        if module in seen or module not in modules:
            continue
        seen.add(module)
        name = modules[module]
        paths.add(name)
        package = module if name.endswith("/__init__.py") else module.rpartition(".")[0]
        dependencies = set()
        for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                dependencies.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else (node.module or "")
                dependencies.add(base)
                dependencies.update(base + "." + a.name for a in node.names)
        dependencies.update(".".join(module.split(".")[:i]) for i in range(1, len(module.split("."))))
        pending.extend(dependencies)
    paths.update({"Dockerfile", "requirements-deployment.txt", "deployment/requirements-runtime.lock",
                  "deployment/web/live_capture.js", "deployment/web/live_timing.js",
                  "deployment/web/upload_preview.js", "audio/beats/LICENSE"})
    paths.add("deployment/artifact_catalog.json")
    candidates.add("deployment/artifact_catalog.json")
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(paths):
        if name not in candidates:
            raise RuntimeError("source outside reviewed allowlist")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    readme = """---
title: AValidate Audio-Video Verification
sdk: docker
app_port: 7860
---

# AValidate

Private, authenticated media admission and live capture diagnostics.
MODEL_NOT_READY: public predictions are null. No weights are required or included.
Keep the existing Space private. Preserve AVAL_ACCESS_MODE=gradio_auth and the existing
AVAL_GRADIO_USERNAME / AVAL_GRADIO_PASSWORD Secrets. Docker runs as UID 1000 on port 7860.
Default entry point: python -m deployment.gradio_app. No deployment is performed by this bundle.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    paths.update({"README.md", ".dockerignore", "Dockerfile.dockerignore"})
    parents = {p.as_posix() for n in paths for p in Path(n).parents if p.as_posix() != "."}
    ignore = ["# Exact release allowlist, including against stale remote files.", "**"]
    for parent in sorted(parents, key=lambda p: (p.count("/"), p)):
        ignore.extend(["!" + parent + "/", parent + "/**"])
    ignore.extend("!" + n for n in sorted(paths))
    for name in (".dockerignore", "Dockerfile.dockerignore"):
        (destination / name).write_text("\n".join(ignore) + "\n", encoding="utf-8", newline="\n")
    files = {n: {"bytes": (destination / n).stat().st_size,
                 "sha256": hashlib.sha256((destination / n).read_bytes()).hexdigest()} for n in sorted(paths)}
    manifest = {"schema": "aval.gated-space-release.v2", "files": files,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "uncommitted_source": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
                "public_predictions_enabled": False, "model_artifacts_included": False,
                "target_space": "vvenkata/avalidate-audio-video-verification"}
    (destination.parent / (destination.name + "-manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.output)
    print(f"Prepared {len(result['files'])} files; public inference disabled; no publishing performed.")


if __name__ == "__main__":
    main()
