"""Run Stage 2D profiling without a server, image mutation, or writable models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='avalidate:stage2a')
    p.add_argument('--run-name',default='profile-01')
    p.add_argument('--audit',type=Path,default=Path('_local/phase8_stage2d'))
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    audit=(repo/args.audit).resolve();audit.relative_to(repo)
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run_name):raise ValueError('simple run name required')
    out=audit/args.run_name;out.mkdir(exist_ok=False)
    previous=repo/'_local/phase8_stage2c';models=previous/'models';inputs=previous/'inputs'
    baseline=previous/'docker-final/run';manifest=(previous/'manifest.sha256').read_text().strip()
    exe=shutil.which('docker') or str(Path(os.environ['LOCALAPPDATA'])/'Programs/DockerDesktop/resources/bin/docker.exe')
    def call(*a,check=True):return subprocess.run([exe,*map(str,a)],check=check,capture_output=True,text=True,timeout=1200)
    image=json.loads(call('image','inspect',args.image).stdout)[0]['Id']
    (out/'image-id.txt').write_text(image)
    (out/'docker-version.json').write_text(call('version','--format','{{json .}}').stdout)
    suffix=uuid.uuid4().hex[:12];volume='avalidate-stage2d-'+suffix
    call('volume','create',volume)
    try:
        seed=['run','--rm','--user','0:0','--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--memory','1g','--cpus','2','--pids-limit','64']
        for src,dest,ro,kind in [(models,'/source-models',True,'bind'),(inputs,'/inputs',True,'bind'),
                (volume,'/cache',False,'volume'),(out,'/results',False,'bind'),
                (repo/'deployment','/app/deployment',True,'bind'),
                (repo/'fusion','/app/fusion',True,'bind')]:
            seed+=['--mount',f'type={kind},source={src},target={dest}'+(',readonly' if ro else '')]
        seeded=call(*seed,image,'python','-B','-m','deployment.seed_artifact_volume',manifest,check=False)
        (out/'seed.log').write_text(seeded.stdout+seeded.stderr)
        if seeded.returncode:raise RuntimeError('authenticated volume seeding failed')
        for profile in ('native-volume','windows-bind'):
            name='avalidate-stage2d-'+profile+'-'+suffix;result=out/profile;result.mkdir()
            command=['create','--name',name,'--network','none','--read-only','--cap-drop','ALL',
                '--security-opt','no-new-privileges','--memory','4g','--cpus','2','--pids-limit','128',
                '--tmpfs','/tmp:rw,nosuid,size=512m']
            mounts=[(volume if profile=='native-volume' else models,'/models',True,'volume' if profile=='native-volume' else 'bind'),
                (inputs,'/inputs',True,'bind'),(baseline,'/baseline',True,'bind'),(result,'/results',False,'bind')]
            mounts += [(repo/n,'/app/'+n,True,'bind') for n in ('deployment','audio','vision','fusion')]
            for src,dest,ro,kind in mounts:command+=['--mount',f'type={kind},source={src},target={dest}'+(',readonly' if ro else '')]
            command += [image,'python','-B','-m','deployment.stage2d_profile','--models','/models','--inputs','/inputs',
                '--baseline','/baseline','--manifest-sha256',manifest,'--output','/results/run','--profile',profile]
            if profile=='native-volume':command.append('--compare-inference-mode')
            call(*command);finished=False
            try:
                call('start',name);deadline=time.monotonic()+1200
                with (result/'stats.jsonl').open('w') as stats:
                    while True:
                        state=json.loads(call('inspect',name).stdout)[0]
                        if not state['State']['Running']:break
                        observed=call('stats','--no-stream','--format','{{json .}}',name,check=False)
                        if observed.stdout.strip():stats.write(json.dumps({'epoch':time.time(),'stats':json.loads(observed.stdout)})+'\n');stats.flush()
                        if time.monotonic()>deadline:raise TimeoutError('profiling deadline exceeded')
                        time.sleep(2)
                logs=call('logs',name,check=False);(result/'execution.log').write_text(logs.stdout+logs.stderr)
                record={k:state[k] for k in ('Id','Image','State','Mounts')}
                record['limits']={k:state['HostConfig'][k] for k in ('ReadonlyRootfs','NetworkMode','Memory','NanoCpus','PidsLimit','CapDrop','SecurityOpt')}
                (result/'execution.json').write_text(json.dumps(record,indent=2))
                if state['State']['ExitCode']!=0:raise RuntimeError(profile+' profile failed; inspect execution.log')
                finished=True;print(profile,'verified',flush=True)
            finally:
                if not finished:call('stop','--timeout','2',name,check=False)
                call('rm',name,check=False)
    finally:
        call('volume','rm',volume,check=False) # Only this task's uniquely named temporary volume.


if __name__=='__main__':main()
