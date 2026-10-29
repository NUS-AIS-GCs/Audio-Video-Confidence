"""One offline genuine worker for the private product demo's separate spools."""
import argparse
import os
import re
from pathlib import Path
import signal
import tempfile
from threading import Timer
import time

from deployment.general_inference import GeneralResearchService, response_for_research, error
from deployment.live.buffer import LiveBufferError
from deployment.live.replay import LiveReplayAdapter, abort_job, JOB_TIMEOUT
from deployment.live.research_store import ResearchStore
from deployment.product_upload import UploadJobStore, HEALTH_TICKET


def run_upload(worker, store, ticket):
    owner=None
    try:
        grant=store.check(ticket);owner=grant['owner']
        if store.path(ticket,'.result.json').exists():raise LiveBufferError('UPLOAD_ALREADY_PROCESSED')
        capture=store.capture(ticket)
        content=store.media(ticket,capture)
        store.check(ticket,owner)
        result=response_for_research(worker.predict_research(content,capture['filename']))
        del content
        store.check(ticket,owner)
        store.write(ticket,'.result.json',{'ticket':ticket,'owner':owner,'result':result})
        try:store.check(ticket,owner)
        except LiveBufferError:
            store.path(ticket,'.result.json').unlink(missing_ok=True);raise
        return result['status']
    except Exception as exc:
        if owner:
            try:
                store.check(ticket,owner)
                if not store.path(ticket,'.result.json').exists():
                    code=exc.code if isinstance(exc,LiveBufferError) else 'GENERAL_INFERENCE_FAILED'
                    store.write(ticket,'.result.json',{'ticket':ticket,'owner':owner,'result':error(code)})
            except Exception:pass
        return 'REJECTED'
    finally:store.delete_capture(ticket)


def abort_upload(store,ticket):
    try:store.discard(ticket)
    finally:os._exit(124)


def main():
    import resource
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('models','live-spool','upload-spool','key-file'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    import torch
    from filelock import FileLock
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    live=ResearchStore(args.live_spool,args.key_file);uploads=UploadJobStore(args.upload_spool,args.key_file)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic()
    with FileLock(str(args.live_spool/'.worker.lock'),timeout=0),tempfile.TemporaryDirectory(prefix='aval-product-derived-') as derived:
        timer=Timer(JOB_TIMEOUT,lambda:os._exit(125));timer.daemon=True;timer.start()
        try:worker=GeneralResearchService(args.models,Path(derived));adapter=LiveReplayAdapter(worker,live)
        finally:timer.cancel()
        startup=time.monotonic()-started
        metrics={}
        def health(status,**extra):
            metrics.update(extra)
            uploads.write(HEALTH_TICKET,'.health.json',{'status':status,'at':time.time(),
                'startup_seconds':startup,'profile_sha256':worker.profile_sha256,
                'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**metrics},replace=True)
        health('READY');print('PRODUCT_WORKER_READY',flush=True)
        try:
            while not stopped:
                # Alternate sources to avoid an upload monopolizing live replay.
                for store,is_upload in ((uploads,True),(live,False)):
                    for path in sorted(store.root.glob('*.grant.json'))[:4]:
                        ticket=path.name.removesuffix('.grant.json')
                        if not re.fullmatch('[a-f0-9]{64}',ticket):continue
                        if not store.path(ticket,'.capture.json').exists() or store.path(ticket,'.result.json').exists():continue
                        try:store.check(ticket)
                        except LiveBufferError:
                            store.delete_capture(ticket);continue
                        health('BUSY');job_start=time.monotonic()
                        abort=(lambda s=store,t=ticket:abort_upload(s,t)) if is_upload else (lambda s=store,t=ticket:abort_job(s,t))
                        watchdog=Timer(JOB_TIMEOUT,abort);watchdog.daemon=True;watchdog.start()
                        try:
                            if is_upload:status=run_upload(worker,store,ticket)
                            else:
                                status=adapter.run(ticket)['status'];adapter.last_audit=None
                            print(('UPLOAD_' if is_upload else 'LIVE_')+status,flush=True)
                        finally:watchdog.cancel()
                        health('READY',last_job_seconds=time.monotonic()-job_start)
                        break
                health('READY');time.sleep(.25)
        finally:
            uploads.path(HEALTH_TICKET,'.health.json').unlink(missing_ok=True)


if __name__=='__main__':main()
