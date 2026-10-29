"""Read-only target checks plus disposable permission probes; never starts the product.

Supported profile: rootful local Docker, no userns remapping, host broker UID 1000.
No deployment, remote host mutation, model load, credentials or device access.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

# Support direct invocation from the checked-out source tree.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def require_identity(system, uid, security_options):
    if system!='linux' or uid!=1000:
        raise RuntimeError('LINUX_HOST_UID_1000_REQUIRED')
    if any('rootless' in str(item).lower() or 'userns' in str(item).lower() for item in security_options):
        raise RuntimeError('ROOTFUL_DOCKER_WITHOUT_USERNS_REQUIRED')


def require_private(metadata):
    if metadata.st_uid!=1000 or stat.S_IMODE(metadata.st_mode)!=0o700:
        raise RuntimeError('PRIVATE_DIRECTORY_UID1000_MODE0700_REQUIRED')


def require_model_permissions(metadata, is_directory):
    expected=0o750 if is_directory else 0o440
    if metadata.st_uid!=1000 or metadata.st_gid!=0 or stat.S_IMODE(metadata.st_mode)!=expected:
        raise RuntimeError('MODEL_OWNER1000_GROUP0_PERMISSIONS_REQUIRED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--image',required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',args.image):
        raise RuntimeError('IMMUTABLE_REVIEWED_IMAGE_ID_REQUIRED')
    info=json.loads(subprocess.check_output(['docker','info','--format','{{json .SecurityOptions}}'],text=True))
    require_identity(sys.platform,os.getuid() if hasattr(os,'getuid') else -1,info)
    if args.models.is_symlink() or args.private_dir.is_symlink():raise RuntimeError('REAL_PRIVATE_DIRECTORIES_REQUIRED')
    models=args.models.resolve(strict=True);private=args.private_dir.resolve(strict=True)
    require_private(private.stat());require_model_permissions(models.stat(),True)
    from deployment.provision_artifacts import artifact_catalog, verify_artifacts
    if not verify_artifacts(models)['ready']:raise RuntimeError('MODEL_CATALOG_NOT_READY')
    for relative in artifact_catalog():
        path=models/relative
        if path.is_symlink():raise RuntimeError('REAL_MODEL_FILES_REQUIRED')
        require_model_permissions(path.stat(),False)
        for parent in path.parents:
            if parent==models:break
            if parent.is_symlink():raise RuntimeError('REAL_MODEL_DIRECTORIES_REQUIRED')
            require_model_permissions(parent.stat(),True)
    image=json.loads(subprocess.check_output(['docker','image','inspect',args.image],text=True))[0]
    if image['Os']!='linux' or image['Architecture']!='amd64' or image['Config']['User']!='1000:1000':
        raise RuntimeError('AUDITED_LINUX_AMD64_UID1000_IMAGE_REQUIRED')
    common=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--cpus','2','--pids-limit','128',
        '--tmpfs','/tmp:rw,nosuid,size=512m','-e','PYTHONPATH=/app',
        '--mount',f'type=bind,source={models},target=/source-models,readonly']
    # Exercise the root seeder's actual DAC access with zero capabilities.
    seed=subprocess.run([*common,'--user','0:0','--memory','1g','--memory-swap','1g',args.image,
        'python','-B','-c',"from deployment.provision_artifacts import verify_artifacts;assert verify_artifacts('/source-models')['ready'];print('SEEDER_PRIVATE_READ_OK')"],
        check=True,text=True,capture_output=True,timeout=120)
    with tempfile.TemporaryDirectory(prefix='preflight-',dir=private) as directory:
        probe=Path(directory);probe.chmod(0o700)
        key=probe/'key';key.write_bytes(b'GC2_PERMISSION_PROBE_NOT_A_SECRET');key.chmod(0o600)
        (probe/'spool').mkdir(mode=0o700)
        script=Path(__file__).with_name('verify_sandbox.py').resolve()
        worker=subprocess.run([*common,'--memory','4g','--memory-swap','4g',
            '--mount',f'type=bind,source={probe},target=/permission-probe',
            '--mount',f'type=bind,source={script},target=/preflight.py,readonly',args.image,
            'python','-B','/preflight.py'],check=True,text=True,capture_output=True,timeout=120)
    print(seed.stdout.strip());print(worker.stdout.strip())
    print('TARGET_PREFLIGHT_PASSED; scientific/public release approval is unchanged')


if __name__=='__main__':main()
