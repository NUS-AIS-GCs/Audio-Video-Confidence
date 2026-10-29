"""Optional private runtime provisioning followed by the unchanged gated public UI.

Never the default Docker entry point. Operator configuration is required; artifact
download or successful hash checks cannot authorize upload or live predictions.
"""
import os

from deployment.provision_artifacts import provision_artifacts, runtime_bindings
from deployment.runtime import DeploymentConfig


def main():
    if not DeploymentConfig.from_environment().authenticated_uploads_enabled:
        raise RuntimeError("AUTHENTICATION_CONFIGURATION_REQUIRED")
    root = os.environ.get("AVAL_PRIVATE_ARTIFACT_ROOT", "/tmp/aval-private-models")
    try:
        provision_artifacts(root, source_root=os.environ.get("AVAL_ARTIFACT_SOURCE_ROOT"),
            repo_id=os.environ.get("AVAL_ARTIFACT_REPO"), revision=os.environ.get("AVAL_ARTIFACT_COMMIT"),
            token=os.environ.get("AVAL_ARTIFACT_READ_TOKEN"), include_a2=True)
    except Exception:
        raise RuntimeError("PRIVATE_ARTIFACT_PROVISIONING_FAILED") from None
    os.environ.update(runtime_bindings(root))
    from deployment.gradio_app import main as serve
    serve()


if __name__ == "__main__":
    main()
