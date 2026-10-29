"""No-network, resource-isolated general-media research app. Never Docker default.

For engineering tests only. Existing public and authenticated TRAIN entry points
remain separate. No environment variable in this module authorizes production.
"""
import os
from pathlib import Path
import tempfile

from deployment.runtime import DeploymentConfig
from deployment.provision_artifacts import runtime_bindings


def main():
    config = DeploymentConfig.from_environment()
    if not config.authenticated_uploads_enabled:
        raise RuntimeError('AUTHENTICATION_CONFIGURATION_REQUIRED')
    from deployment.general_inference import GeneralResearchService
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with tempfile.TemporaryDirectory(prefix='aval-general-derived-') as directory:
        try:
            service = GeneralResearchService(config.model_root,Path(directory))
        except Exception:
            raise RuntimeError('GENERAL_RESEARCH_INITIALIZATION_FAILED') from None
        os.environ.update(runtime_bindings(config.model_root))
        from deployment.gradio_app import main as serve
        serve(general_research=service)


if __name__ == '__main__':
    main()
