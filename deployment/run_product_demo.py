"""Start/stop the private localhost UI and an independently isolated Docker worker.

On a trusted self-managed Linux/Windows Docker host, access directly on localhost
or through an operator's SSH tunnel. This does not deploy or enable public gates.
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid

from deployment.product_upload import UploadJobStore, HEALTH_TICKET
from deployment.provision_artifacts import verify_artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path)
    parser.add_argument('--private-dir',type=Path,default=Path('_local/product_demo/private'))
    parser.add_argument('--port',type=int,default=7861)
    parser.add_argument('--image',default='avalidate:private-product')
    parser.add_argument('--no-build',action='store_true')
    parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    if args.private_dir.is_symlink():raise RuntimeError('PRIVATE_DIRECTORY_REQUIRED')
    private=args.private_dir.resolve()
    if args.stop:
        if not private.is_dir():raise RuntimeError('DEMO_NOT_RUNNING')
        (private/'stop.request').write_text('stop',encoding='ascii');return
    if args.models is None:parser.error('--models is required to start')
    if not 1024<=args.port<=65535:parser.error('port must be 1024..65535')
    models=args.models.resolve()
    if not verify_artifacts(models)['ready']:raise RuntimeError('ORIGINAL_MODEL_ARTIFACTS_REQUIRED')
    docker=shutil.which('docker')
    if not docker and os.environ.get('LOCALAPPDATA'):
        candidate=Path(os.environ['LOCALAPPDATA'])/'Programs/DockerDesktop/resources/bin/docker.exe'
        if candidate.is_file():docker=str(candidate)
    if not docker:raise RuntimeError('DOCKER_REQUIRED')
    private.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(private,0o700)
    from filelock import FileLock
    with FileLock(str(private/'.demo.lock'),timeout=0):
        (private/'stop.request').unlink(missing_ok=True)
        credentials=private/'login.json'
        if credentials.is_symlink():raise RuntimeError('PRIVATE_CREDENTIAL_FILE_REQUIRED')
        if not credentials.exists():
            with credentials.open('x',encoding='utf-8') as stream:
                json.dump({'username':os.environ.get('AVAL_GRADIO_USERNAME','avalidate'),
                           'password':os.environ.get('AVAL_GRADIO_PASSWORD') or secrets.token_urlsafe(32)},stream)
            os.chmod(credentials,0o600)
        login=json.loads(credentials.read_text(encoding='utf-8'))
        if not all(isinstance(login.get(n),str) and login[n] for n in ('username','password')):
            raise RuntimeError('AUTHENTICATION_CONFIGURATION_REQUIRED')
        def call(*command,timeout=300):
            result=subprocess.run([docker,*map(str,command)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,timeout=timeout)
            if result.returncode:raise RuntimeError('DOCKER_OPERATION_FAILED')
            return result.stdout
        if not args.no_build:
            text=call('build','-t',args.image,'.',timeout=900)
            (private/'build.log').write_text(text,encoding='utf-8')
        metadata=json.loads(call('image','inspect',args.image))[0]
        common=['--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
                '--cpus','2','--pids-limit','128','--tmpfs','/tmp:rw,nosuid,size=512m']
        volume='aval-product-'+uuid.uuid4().hex[:16];worker_name=volume+'-worker'
        call('volume','create',volume)
        broker=None;worker=None;logs=[]
        env={**os.environ,'AVAL_ACCESS_MODE':'gradio_auth','AVAL_GRADIO_USERNAME':login['username'],
             'AVAL_GRADIO_PASSWORD':login['password'],'GRADIO_ANALYTICS_ENABLED':'False'}
        started=time.monotonic();stopping=False
        def stop(*_):
            nonlocal stopping
            stopping=True
        for kind in (signal.SIGINT,signal.SIGTERM):signal.signal(kind,stop)
        try:
            seed=call('run','--rm',*common,'--user','0:0','--memory','1g','--memory-swap','1g',
                '--mount',f'type=bind,source={models},target=/source-models,readonly',
                '--mount',f'type=volume,source={volume},target=/cache',args.image,
                'python','-B','-m','deployment.product_seed')
            (private/'seed.log').write_text(seed,encoding='utf-8')
            broker_log=(private/'broker.log').open('w',encoding='utf-8');logs.append(broker_log)
            broker=subprocess.Popen([sys.executable,'-B','-m','deployment.product_app','--private-dir',str(private),
                '--port',str(args.port)],env=env,stdout=broker_log,stderr=subprocess.STDOUT)
            for _ in range(200):
                if (private/'research.key').is_file() and (private/'upload-spool').is_dir():break
                if broker.poll() is not None:raise RuntimeError('BROKER_START_FAILED')
                time.sleep(.1)
            else:raise RuntimeError('BROKER_START_TIMEOUT')
            worker_log=(private/'worker.log').open('w',encoding='utf-8');logs.append(worker_log)
            worker=subprocess.Popen([docker,'run','--rm','--name',worker_name,*common,'--memory','4g','--memory-swap','4g',
                '-e','PYTHONPATH=/app','--mount',f'type=volume,source={volume},target=/models,readonly',
                '--mount',f'type=bind,source={private}/spool,target=/spool',
                '--mount',f'type=bind,source={private}/upload-spool,target=/upload-spool',
                '--mount',f'type=bind,source={private}/research.key,target=/research.key,readonly',args.image,
                'python','-B','-m','deployment.product_worker','--models','/models/verified',
                '--live-spool','/spool','--upload-spool','/upload-spool','--key-file','/research.key'],
                stdout=worker_log,stderr=subprocess.STDOUT)
            store=UploadJobStore(private/'upload-spool',private/'research.key')
            for _ in range(400):
                if worker.poll() is not None:raise RuntimeError('ISOLATED_WORKER_START_FAILED')
                if broker.poll() is not None:raise RuntimeError('BROKER_START_FAILED')
                try:
                    health=store.read(HEALTH_TICKET,'.health.json',128*1024)
                    if health['status']=='READY':break
                except Exception:pass
                time.sleep(.5)
            else:raise RuntimeError('ISOLATED_WORKER_START_TIMEOUT')
            status={'url':f'http://127.0.0.1:{args.port}','image':metadata['Id'],'image_bytes':metadata['Size'],
                'startup_seconds':time.monotonic()-started,'worker':health,'credentials_file':str(credentials),
                'public_predictions_enabled':False,'worker_name':worker_name}
            (private/'ready.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            print('PRIVATE_PRODUCT_READY '+status['url'],flush=True)
            print('Login credentials file: '+str(credentials),flush=True)
            while not stopping and not (private/'stop.request').exists():
                if broker.poll() is not None or worker.poll() is not None:
                    if (private/'stop.request').exists():break
                    raise RuntimeError('PRODUCT_PROCESS_STOPPED')
                time.sleep(.5)
        finally:
            (private/'stop.request').write_text('stop',encoding='ascii')
            if broker:
                try:broker.wait(timeout=20)
                except subprocess.TimeoutExpired:broker.terminate();broker.wait(timeout=10)
            if worker:
                subprocess.run([docker,'stop','--time','10',worker_name],capture_output=True,timeout=30)
                try:worker.wait(timeout=15)
                except subprocess.TimeoutExpired:worker.terminate()
            for folder in ('spool','upload-spool'):
                path=private/folder
                if path.is_dir() and (private/'research.key').is_file():
                    store=UploadJobStore(path,private/'research.key')
                    for grant in list(path.glob('*.grant.json'))[:16]:
                        ticket=grant.name.removesuffix('.grant.json')
                        if re.fullmatch('[a-f0-9]{64}',ticket):store.discard(ticket)
            removed=subprocess.run([docker,'volume','rm',volume],capture_output=True,timeout=30)
            (private/'cleanup.json').write_text(json.dumps({'model_volume_removed':removed.returncode==0}),encoding='utf-8')
            (private/'ready.json').unlink(missing_ok=True)
            for log in logs:log.close()


if __name__=='__main__':main()
