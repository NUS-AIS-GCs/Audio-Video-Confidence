"""Authenticated loopback-demo upload tickets; no models or inference in this broker."""
import os
from pathlib import Path
import secrets
from threading import Event, Lock
import time

from deployment.auth import login_owner
from deployment.general_inference import error, response_for_research
from deployment.live.buffer import LiveBufferError
from deployment.live.research_store import ResearchStore, TTL, AUTH_LEASE, canonical, digest
from deployment.runtime import DeploymentConfig, MAX_UPLOAD_BYTES_HARD_LIMIT

JOB_SCHEMA = 'aval.private-upload-job.v1'
HEALTH_TICKET = '0'*64


class UploadJobStore(ResearchStore):
    """Separate upload spool; preserve the existing live 8 MiB package limit."""
    def issue_upload(self, owner, content, filename):
        suffix=Path(filename).suffix.lower() if type(filename) is str else ''
        if suffix not in ('.mp4','.mov'):
            raise LiveBufferError('UNSUPPORTED_EXTENSION')
        if type(content) is not bytes or not 0<len(content)<=MAX_UPLOAD_BYTES_HARD_LIMIT:
            raise LiveBufferError('UPLOAD_TOO_LARGE')
        ticket=secrets.token_hex(32);cancel=secrets.token_hex(32);now=self.clock()
        capture={'schema':JOB_SCHEMA,'ticket':ticket,'owner':owner,'created':now,'expires':now+TTL,
                 'filename':'source'+suffix,'bytes':len(content),'source_sha256':digest(content)}
        grant={'ticket':ticket,'owner':owner,'created':now,'expires':now+TTL,'active':True,
               'lease_until':now+AUTH_LEASE,'capture_sha256':digest(canonical(capture)),
               'cancel_sha256':digest(cancel.encode())}
        try:
            with self.path(ticket,'.upload.bin').open('xb') as stream:stream.write(content)
            os.chmod(self.path(ticket,'.upload.bin'),0o600)
            self.write(ticket,'.capture.json',capture)
            self.write(ticket,'.grant.json',grant)  # ready marker, published last
        except Exception:
            self.delete_capture(ticket)
            self.path(ticket,'.grant.json').unlink(missing_ok=True)
            raise
        return ticket,cancel

    def media(self, ticket, capture):
        if (capture.get('schema')!=JOB_SCHEMA or capture.get('filename') not in ('source.mp4','source.mov')
                or type(capture.get('bytes')) is not int or not 0<capture['bytes']<=MAX_UPLOAD_BYTES_HARD_LIMIT):
            raise LiveBufferError('UPLOAD_JOB_INVALID')
        path=self.path(ticket,'.upload.bin')
        if path.stat().st_size!=capture['bytes']:raise LiveBufferError('UPLOAD_JOB_CHANGED')
        with path.open('rb') as stream:content=stream.read(MAX_UPLOAD_BYTES_HARD_LIMIT+1)
        if len(content)!=capture['bytes'] or digest(content)!=capture['source_sha256']:
            raise LiveBufferError('UPLOAD_JOB_CHANGED')
        return content

    def delete_capture(self, ticket):
        super().delete_capture(ticket)
        self.path(ticket,'.upload.bin').unlink(missing_ok=True)

    def discard(self, ticket):
        try:self.revoke(ticket)
        except LiveBufferError:pass
        self.delete_capture(ticket)
        for suffix in ('.grant.json','.result.json'):
            self.path(ticket,suffix).unlink(missing_ok=True)


class ProductUploadBroker:
    """One outstanding upload; login registry checked while renewing its lease."""
    def __init__(self, store, *, timeout=180, sleep=time.sleep):
        self.store=store;self.config=DeploymentConfig(store.root,(),())
        self.upload_store=None;self.timeout=timeout;self.sleep=sleep
        self._lock=Lock();self._slot=Lock();self._active={};self._closed=Event()
        # A restarted broker cannot resume authorization from old login sessions.
        for path in list(store.root.glob('*.grant.json'))[:8]:
            store.discard(path.name.removesuffix('.grant.json'))
        for path in list(store.root.glob('*.upload.bin'))[:8]:
            store.delete_capture(path.name.removesuffix('.upload.bin'))

    def cancel_owned(self, request):
        owner=login_owner(getattr(request,'request',None))
        if owner is None:return
        with self._lock:
            ticket=self._active.pop(owner,None)
            if ticket:self.store.discard(ticket)

    def submit_owned(self, handle, request):
        raw=getattr(request,'request',None);owner=login_owner(raw)
        if owner is None or self.upload_store is None or raw.scope.get('aval.upload_guard') is not self.upload_store:
            return error('UPLOAD_AUTHENTICATION_REQUIRED')
        if not handle:return error('UPLOAD_REQUIRED')
        if not self._slot.acquire(blocking=False):return error('RESOURCE_BUSY')
        ticket=None
        try:
            health=self.store.read(HEALTH_TICKET,'.health.json',128*1024)
            if health.get('status') not in ('READY','BUSY') or not 0<=self.store.clock()-health.get('at',0)<180:
                return error('PRODUCT_WORKER_NOT_READY')
            with self.upload_store.lease(handle,owner) as (path,filename):
                with path.open('rb') as stream:content=stream.read(self.config.max_upload_bytes+1)
                with self._lock:
                    ticket,cancel=self.store.issue_upload(owner,content,filename)
                    self._active[owner]=ticket
                del content
                deadline=time.monotonic()+self.timeout
                while time.monotonic()<deadline and not self._closed.is_set():
                    with self._lock:
                        if login_owner(raw)!=owner or self._active.get(owner)!=ticket:
                            return error('UPLOAD_CANCELLED')
                        self.store.renew(ticket,owner)
                        envelope=self.store.result(ticket,owner,cancel)
                    if envelope is not None:
                        result=response_for_research(envelope['result'])
                        with self._lock:
                            if login_owner(raw)!=owner:return error('UPLOAD_AUTHENTICATION_REQUIRED')
                            if self._active.get(owner)!=ticket:return error('UPLOAD_CANCELLED')
                            self.store.check(ticket,owner)
                            return result
                    self.sleep(.25)
                return error('PRODUCT_INFERENCE_TIMEOUT')
        except Exception as exc:
            from deployment.uploads import UploadError
            if isinstance(exc,UploadError):return error('UPLOAD_NOT_FOUND')
            if isinstance(exc,LiveBufferError):
                code=exc.code
                if code=='LIVE_RESEARCH_NOT_FOUND':code='PRODUCT_WORKER_NOT_READY' if ticket is None else 'UPLOAD_CANCELLED'
                return error(code)
            return error('GENERAL_INFERENCE_FAILED')
        finally:
            with self._lock:
                if self._active.get(owner)==ticket:self._active.pop(owner,None)
                if ticket:self.store.discard(ticket)
            self._slot.release()

    def close(self):
        self._closed.set()
        with self._lock:
            for ticket in self._active.values():self.store.discard(ticket)
            self._active.clear()
