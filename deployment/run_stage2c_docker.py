"""Coordinate an isolated Docker TRAIN encoder audit from host PowerShell.

Uses the existing image by immutable ID; optional build uses the original
Dockerfile. No shell, server, production switch or checkpoint image layer.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='avalidate:stage2a')
    parser.add_argument('--audit-directory',type=Path,default=Path('_local/phase8_stage2c'))
    parser.add_argument('--docker-exe',type=Path)
    parser.add_argument('--build-runtime',action='store_true')
    parser.add_argument('--run-name',default='docker')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    audit=(repo/args.audit_directory).resolve()
    audit.relative_to(repo)
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run_name):
        raise ValueError('simple unique run name required')
    output=audit/args.run_name
    output.mkdir(exist_ok=False)
    exe=args.docker_exe or shutil.which('docker')
    if not exe:
        exe=Path(os.environ.get('LOCALAPPDATA',''))/'Programs/DockerDesktop/resources/bin/docker.exe'
    def call(*command,check=True):
        return subprocess.run([str(exe),*map(str,command)],check=check,capture_output=True,text=True,timeout=900)
    version=call('version','--format','{{json .}}')
    (output/'docker-version.json').write_text(version.stdout,encoding='utf-8')
    if args.build_runtime:
        image='avalidate:stage2c-runtime'
        build=call('build','--tag',image,'--file',repo/'Dockerfile',repo,check=False)
        (output/'docker-build.log').write_text(build.stdout+build.stderr,encoding='utf-8')
        if build.returncode:
            raise RuntimeError('test-only image build failed; execution unverified')
    else:
        image=args.image
    inspected=json.loads(call('image','inspect',image).stdout)[0]
    image_id=inspected['Id']
    if not image_id.startswith('sha256:') or len(image_id)!=71:
        raise RuntimeError('immutable image ID unavailable')
    (output/'executed-image-id.txt').write_text(image_id+'\n',encoding='utf-8')
    models=(audit/'models').resolve(strict=True);inputs=(audit/'inputs').resolve(strict=True)
    manifest_hash=(audit/'manifest.sha256').read_text().strip()
    if len(manifest_hash)!=64 or any(x not in '0123456789abcdef' for x in manifest_hash):
        raise RuntimeError('explicit manifest SHA-256 required')
    name='avalidate-stage2c-'+uuid.uuid4().hex[:12]
    mounts=[(models,'/models',True),(inputs,'/inputs',True),(output,'/results',False)]
    mounts.extend((repo/n,'/app/'+n,True) for n in ('deployment','audio','vision','fusion'))
    command=['create','--name',name,'--network','none','--read-only','--cap-drop','ALL',
             '--security-opt','no-new-privileges','--memory','4g','--cpus','2','--pids-limit','128',
             '--tmpfs','/tmp:rw,nosuid,size=512m']
    for source,target,readonly in mounts:
        command.extend(['--mount',f'type=bind,source={source},target={target}'+(',readonly' if readonly else '')])
    command.extend([image_id,'python','-B','-m','deployment.stage2c_probe','--models','/models',
        '--inputs','/inputs','--manifest-sha256',manifest_hash,'--output','/results/run','--threads','4'])
    (output/'container-id.txt').write_text(call(*command).stdout,encoding='utf-8')
    finished=False
    try:
        call('start',name)
        deadline=time.monotonic()+900
        with (output/'docker-stats.jsonl').open('w',encoding='utf-8') as log:
            while True:
                inspected=json.loads(call('inspect',name).stdout)[0]
                if not inspected['State']['Running']:break
                sample=call('stats','--no-stream','--format','{{json .}}',name,check=False)
                if sample.stdout.strip():
                    log.write(json.dumps({'utc_epoch':time.time(),'stats':json.loads(sample.stdout)})+'\n');log.flush()
                if time.monotonic()>deadline:raise TimeoutError('test container deadline exceeded')
                time.sleep(2)
        logs=call('logs',name,check=False)
        (output/'docker.log').write_text(logs.stdout+logs.stderr,encoding='utf-8')
        record={k:inspected[k] for k in ('Id','Image','State','Mounts')}
        record['limits']={k:inspected['HostConfig'][k] for k in ('ReadonlyRootfs','Memory','NanoCpus','PidsLimit','NetworkMode','CapDrop','SecurityOpt')}
        (output/'container-execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if inspected['State']['ExitCode']!=0:
            raise RuntimeError(f"encoder audit exited {inspected['State']['ExitCode']}; inspect {output/'docker.log'}")
        if not (output/'run/measurements.json').is_file():
            raise RuntimeError('container produced no encoder measurements')
        finished=True
        print('Verified test-only container execution:',output/'run/measurements.json')
    finally:
        if not finished:call('stop','--time','2',name,check=False)
        call('rm',name,check=False)  # Only the uniquely named container created here.


if __name__=='__main__':main()
