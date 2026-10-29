"""Loopback-only authenticated private upload + live demonstration broker."""
import argparse
import os
from pathlib import Path
import secrets
import signal
import time

from starlette.middleware import Middleware

from deployment.live.research_capture import ResearchCaptureService
from deployment.live.research_store import ResearchStore
from deployment.live_research_app import LoopbackOnly
from deployment.product_upload import ProductUploadBroker, UploadJobStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',type=Path,required=True);parser.add_argument('--port',type=int,default=7861)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('port must be 1024..65535')
    if args.private_dir.is_symlink():raise RuntimeError('PRIVATE_DIRECTORY_REQUIRED')
    root=args.private_dir.resolve();root.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(root,0o700)
    for name in ('spool','upload-spool','empty-models','broker-temp'):
        path=root/name
        if path.is_symlink():raise RuntimeError('PRIVATE_DIRECTORY_REQUIRED')
        path.mkdir(exist_ok=True,mode=0o700)
    key=root/'research.key'
    if not key.exists():
        with key.open('xb') as stream:stream.write(secrets.token_bytes(32))
        os.chmod(key,0o600)
    os.environ.update(AVAL_MODEL_ROOT=str(root/'empty-models'),GRADIO_TEMP_DIR=str(root/'broker-temp'))
    uploads=ProductUploadBroker(UploadJobStore(root/'upload-spool',key))
    capture=ResearchCaptureService(ResearchStore(root/'spool',key))
    from deployment.gradio_app import create_app
    from deployment.live.security import live_app_kwargs
    from deployment.ui.theme import BRAND_CSS, LOGIN_CSS, LOGIN_HTML
    demo,config=create_app(general_research=uploads,research_capture=capture,private_demo=True)
    kwargs=live_app_kwargs(demo.aval_live_service)
    kwargs['middleware'].insert(0,Middleware(LoopbackOnly,port=args.port))
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,stop)
    try:
        demo.launch(server_name='127.0.0.1',server_port=args.port,share=False,auth=config.gradio_auth,
            auth_message=f'<style>{BRAND_CSS}{LOGIN_CSS}</style>{LOGIN_HTML}',css=BRAND_CSS,footer_links=[],
            prevent_thread_lock=True,quiet=True,show_error=False,enable_monitoring=False,
            max_file_size=config.max_upload_bytes,blocked_paths=[str(root)],app_kwargs=kwargs)
        print('PRIVATE_PRODUCT_BROKER_READY',flush=True)
        while not stopping and demo.is_running and not (root/'stop.request').exists():time.sleep(.1)
    finally:
        uploads.close();capture.close();demo.close();demo.aval_live_service.close();demo.aval_upload_store.close()


if __name__=='__main__':main()
