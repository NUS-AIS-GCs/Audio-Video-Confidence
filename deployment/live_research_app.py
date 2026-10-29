"""Loopback-only authenticated capture broker. Models run only in offline replay.

Never use as a hosted Space entry point or publish its port. Runtime login
Secrets are supplied by the operator; a separate private signing key authorizes
local research replay only, never a public/scientific release.
"""
import argparse
import os
from pathlib import Path
import secrets
import signal
import time

from starlette.middleware import Middleware

from deployment.live.research_capture import ResearchCaptureService
from deployment.live.research_store import ResearchStore


class LoopbackOnly:
    def __init__(self, app, port):self.app,self.port=app,port
    async def __call__(self, scope, receive, send):
        if scope['type']=='http':
            headers=dict(scope.get('headers',[]));allowed={f'127.0.0.1:{self.port}'.encode(),f'localhost:{self.port}'.encode()}
            origins={b'http://'+h for h in allowed}
            if (headers.get(b'host') not in allowed or
                    (b'origin' in headers and headers[b'origin'] not in origins) or
                    (scope.get('client') and scope['client'][0] not in ('127.0.0.1','::1','localhost'))):
                await send({'type':'http.response.start','status':403,'headers':[(b'cache-control',b'no-store')]})
                await send({'type':'http.response.body','body':b'LOCAL_RESEARCH_ONLY'});return
        return await self.app(scope,receive,send)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--port',type=int,default=7861)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('port must be 1024..65535')
    root=args.private_dir.resolve()
    # Explicit operator path; no browser filename/path reaches this code.
    if args.private_dir.is_symlink():raise RuntimeError('LIVE_RESEARCH_PRIVATE_DIRECTORY_REQUIRED')
    root.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(root,0o700)
    spool=root/'spool';spool.mkdir(exist_ok=True,mode=0o700);os.chmod(spool,0o700)
    key=root/'research.key'
    if not key.exists():
        with key.open('xb') as stream:stream.write(secrets.token_bytes(32))
        os.chmod(key,0o600)
    service=ResearchCaptureService(ResearchStore(spool,key))
    from deployment.gradio_app import create_app
    from deployment.live.security import live_app_kwargs
    demo,config=create_app(research_capture=service)
    kwargs=live_app_kwargs(demo.aval_live_service)
    kwargs['middleware'].insert(0,Middleware(LoopbackOnly,port=args.port))
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        demo.launch(server_name='127.0.0.1',server_port=args.port,share=False,auth=config.gradio_auth,
            prevent_thread_lock=True,quiet=True,show_error=False,enable_monitoring=False,
            max_file_size=config.max_upload_bytes,blocked_paths=[str(root),str(config.model_root)],app_kwargs=kwargs)
        print('LOCAL_CAPTURE_BROKER_READY',flush=True)
        while not stopping and demo.is_running:time.sleep(.1)
    finally:
        service.close();demo.close();demo.aval_live_service.close();demo.aval_upload_store.close()
        for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':main()
