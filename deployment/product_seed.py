"""Prepare a fresh private read-only Docker model volume; never contains secrets."""
from pathlib import Path
from deployment.provision_artifacts import provision_artifacts, verify_artifacts


def main():
    root=Path('/cache/verified')
    report=provision_artifacts(root,source_root=Path('/source-models'))
    if not report['ready']:raise RuntimeError('MODEL_PROVISIONING_FAILED')
    source_head2=Path('/source-models/Head2')
    if source_head2.exists():
        from deployment.temporal_presence import provision
        provision(source_head2,root/'Head2')
    for path in root.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    if not verify_artifacts(root)['ready']:raise RuntimeError('MODEL_PROVISIONING_FAILED')
    print('PRIVATE_CATALOG_ARTIFACTS_VERIFIED',flush=True)


if __name__=='__main__':main()
