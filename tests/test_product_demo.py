"""Private-demo transport/security controls; unit stubs are never model evidence."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.general_inference import error
from deployment.live.buffer import LiveBufferError
from deployment.product_upload import UploadJobStore, ProductUploadBroker


@pytest.fixture
def store(tmp_path):
    key=tmp_path/'key';key.write_bytes(bytes(range(32)))
    spool=tmp_path/'spool';spool.mkdir()
    return UploadJobStore(spool,key)


def test_signed_upload_job_and_media_identity(store):
    ticket,cancel=store.issue_upload('a'*64,b'media','private-name.mov')
    capture=store.capture(ticket)
    assert capture['filename']=='source.mov'
    assert store.media(ticket,capture)==b'media'
    assert store.result(ticket,'a'*64,cancel) is None
    store.path(ticket,'.upload.bin').write_bytes(b'other')
    with pytest.raises(LiveBufferError,match='UPLOAD_JOB_CHANGED'):store.media(ticket,capture)


def test_wrong_login_cannot_revoke_or_read(store):
    ticket,cancel=store.issue_upload('a'*64,b'media','x.mp4')
    with pytest.raises(LiveBufferError):store.result(ticket,'b'*64,cancel)
    with pytest.raises(LiveBufferError):store.revoke(ticket,owner='b'*64)
    assert store.path(ticket,'.upload.bin').exists()
    store.revoke(ticket,owner='a'*64)
    assert not store.path(ticket,'.upload.bin').exists()
    assert not store.path(ticket,'.capture.json').exists()


def test_expired_auth_lease_and_ticket_cannot_be_renewed(store):
    now=[100.];store.clock=lambda:now[0]
    ticket,_=store.issue_upload('a'*64,b'media','x.mp4')
    now[0]=106.
    with pytest.raises(LiveBufferError):store.renew(ticket,'a'*64)
    now[0]=401.
    with pytest.raises(LiveBufferError):store.check(ticket)
    store.discard(ticket)
    assert not list(store.root.iterdir())


@pytest.mark.parametrize('name,data,code',[('x.webm',b'x','UNSUPPORTED_EXTENSION'),
    ('x.mp4',b'','UPLOAD_TOO_LARGE')])
def test_invalid_upload_never_publishes_a_job(store,name,data,code):
    with pytest.raises(LiveBufferError,match=code):store.issue_upload('a'*64,data,name)
    assert not list(store.root.iterdir())


def test_upload_cap_is_enforced_before_publication(store,monkeypatch):
    monkeypatch.setattr('deployment.product_upload.MAX_UPLOAD_BYTES_HARD_LIMIT',4)
    with pytest.raises(LiveBufferError):store.issue_upload('a'*64,b'12345','x.mp4')
    assert not list(store.root.iterdir())


def test_broker_restart_revokes_old_jobs(store):
    ticket,_=store.issue_upload('a'*64,b'media','x.mp4')
    broker=ProductUploadBroker(store)
    assert not store.path(ticket,'.grant.json').exists()
    assert not store.path(ticket,'.upload.bin').exists()
    broker.close()


def test_unsigned_health_or_unauthenticated_request_never_grants_inference(store):
    broker=ProductUploadBroker(store)
    assert broker.submit_owned('fake',None)==error('UPLOAD_AUTHENTICATION_REQUIRED')
    broker.close()


def test_worker_revalidates_grant_after_processing(store):
    from deployment.product_worker import run_upload
    ticket,_=store.issue_upload('a'*64,b'media','x.mp4')
    def predict(*_):
        store.revoke(ticket)
        return error('INVALID_DURATION')
    assert run_upload(SimpleNamespace(predict_research=predict),store,ticket)=='REJECTED'
    assert not store.path(ticket,'.result.json').exists()
    assert not store.path(ticket,'.upload.bin').exists()


def test_single_use_and_payload_cleanup_on_rejected_media(store):
    from deployment.product_worker import run_upload
    ticket,cancel=store.issue_upload('a'*64,b'media','x.mp4')
    calls=[]
    def predict(*_):
        calls.append(1);return error('INVALID_DURATION')
    worker=SimpleNamespace(predict_research=predict)
    assert run_upload(worker,store,ticket)=='REJECTED'
    assert store.result(ticket,'a'*64,cancel)['result']==error('INVALID_DURATION')
    run_upload(worker,store,ticket)
    assert calls==[1] and not store.path(ticket,'.upload.bin').exists()
    store.discard(ticket)
    assert not list(store.root.iterdir())


def test_transient_open_reader_does_not_destroy_signed_file(store,monkeypatch):
    import os
    ticket='c'*64
    store.write(ticket,'.result.json',{'value':'old'})
    replace=os.replace;attempts=[]
    def locked_reader(source,target):
        attempts.append(1)
        assert store.read(ticket,'.result.json',1024)=={'value':'old'}
        if len(attempts)<3:raise PermissionError('sharing violation')
        return replace(source,target)
    monkeypatch.setattr('deployment.live.research_store.os.replace',locked_reader)
    monkeypatch.setattr('deployment.live.research_store.time.sleep',lambda _:None)
    store.write(ticket,'.result.json',{'value':'new'},replace=True)
    assert store.read(ticket,'.result.json',1024)=={'value':'new'}
    assert len(attempts)==3 and not list(store.root.glob('*.tmp-*'))


def test_permanent_sharing_failure_preserves_old_atomic_value(store,monkeypatch):
    ticket='c'*64;store.write(ticket,'.result.json',{'value':'old'})
    def denied(*_):raise PermissionError('sharing violation')
    monkeypatch.setattr('deployment.live.research_store.os.replace',denied)
    monkeypatch.setattr('deployment.live.research_store.time.sleep',lambda _:None)
    with pytest.raises(PermissionError):store.write(ticket,'.result.json',{'value':'new'},replace=True)
    assert store.read(ticket,'.result.json',1024)=={'value':'old'}
    assert not list(store.root.glob('*.tmp-*'))


def test_logout_invalidates_pending_broker_job_and_cleans_media(store,tmp_path):
    from contextlib import contextmanager
    from starlette.requests import Request
    from deployment.auth import owner_digest
    from deployment.product_upload import HEALTH_TICKET
    app=SimpleNamespace(auth=True,auth_dependency=None,cookie_id='test',tokens={'token':'user'})
    payload=tmp_path/'upload.mp4';payload.write_bytes(b'fixture')
    class Files:
        @contextmanager
        def lease(self,handle,owner):
            assert owner==owner_digest('user','token')
            try:yield payload,'upload.mp4'
            finally:payload.unlink(missing_ok=True)
    files=Files()
    raw=Request({'type':'http','app':app,'headers':[(b'cookie',b'access-token-test=token')],
                 'aval.upload_guard':files})
    broker=ProductUploadBroker(store,sleep=lambda _:app.tokens.clear())
    broker.upload_store=files
    store.write(HEALTH_TICKET,'.health.json',{'status':'READY','at':store.clock()})
    result=broker.submit_owned('handle',SimpleNamespace(request=raw))
    assert result==error('UPLOAD_CANCELLED')
    assert not payload.exists() and not list(store.root.glob('*.grant.json'))
    assert not list(store.root.glob('*.upload.bin'))


def test_foreign_cancel_cannot_remove_another_login_job(store):
    from starlette.requests import Request
    from deployment.auth import owner_digest
    app=SimpleNamespace(auth=True,auth_dependency=None,cookie_id='test',tokens={'a':'user','b':'user'})
    def request(token):
        return SimpleNamespace(request=Request({'type':'http','app':app,
            'headers':[(b'cookie',('access-token-test='+token).encode())]}))
    broker=ProductUploadBroker(store)
    owner=owner_digest('user','a');ticket,_=store.issue_upload(owner,b'fixture','x.mp4')
    broker._active[owner]=ticket
    broker.cancel_owned(request('b'))
    assert store.path(ticket,'.upload.bin').exists()
    broker.cancel_owned(request('a'))
    assert not store.path(ticket,'.upload.bin').exists()
    assert not broker._active
