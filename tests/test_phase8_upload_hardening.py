"""Direct supported Gradio HTTP contract and owned-upload lifecycle regressions.

Synthetic bytes/media only; no checkpoint or held-out dataset is loaded.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import threading
import time
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request
from starlette.routing import Mount

from deployment.gradio_app import create_app
from deployment.live.security import live_app_kwargs
from deployment.uploads import (UploadError, UploadStore, UploadTransportMiddleware,
                                MAX_REQUESTS_PER_MINUTE, MULTIPART_OVERHEAD)


@pytest.fixture
def site(tmp_path, monkeypatch):
    import gradio as gr
    password = secrets.token_urlsafe(24)
    for k, v in {'AVAL_ACCESS_MODE':'gradio_auth', 'AVAL_GRADIO_USERNAME':'alice',
                 'AVAL_GRADIO_PASSWORD':password, 'AVAL_MODEL_ROOT':str(tmp_path/'models'),
                 'GRADIO_TEMP_DIR':str(tmp_path/'uploads'), 'GRADIO_ANALYTICS_ENABLED':'False'}.items():
        monkeypatch.setenv(k, v)
    demo, config = create_app()
    app = gr.mount_gradio_app(FastAPI(), demo, path='/', auth=[('alice',password),('bob',password)],
        app_kwargs=live_app_kwargs(demo.aval_live_service), max_file_size=config.max_upload_bytes,
        blocked_paths=[str(config.model_root)], show_error=False)
    clients = []
    with TestClient(app) as anon:
        def login(name='alice'):
            c = TestClient(app)
            # Independent cookie jars, same running ASGI event loop/queue workers.
            c.portal = anon.portal
            assert c.post('/login', data={'username':name,'password':password}).status_code == 200
            clients.append(c)
            return c
        yield anon, login, demo
        for c in clients:
            c.close()
    assert not demo.aval_upload_store._entries
    assert not demo.aval_upload_store._worker.is_alive()
    assert not list(demo.aval_upload_store.root.glob('payload-*'))
    demo.close()


def upload(client, name='clip.mp4', data=b'unit-only media bytes'):
    return client.post('/gradio_api/upload', files={'files':(name,data,'video/mp4')})


def verify(client, demo, handle):
    fn = next(f for f in demo.fns.values() if getattr(f.fn,'__name__','') == 'check')
    session = secrets.token_hex(12)
    r = client.post('/gradio_api/queue/join', json={
        'fn_index':fn._id, 'data':[handle], 'session_hash':session})
    assert r.status_code == 200, r.text
    event = r.json()['event_id']
    with client.stream('GET','/gradio_api/queue/data',params={'session_hash':session}) as stream:
        for line in stream.iter_lines():
            if line.startswith('data: '):
                row = json.loads(line[6:])
                if row.get('msg') == 'process_completed' and row.get('event_id') == event:
                    assert row['success'], row
                    return row['output']['data'][1]
    raise AssertionError('no verification result')


def test_auth_then_single_file_transport_before_shared_gradio_parser(site):
    anon, login, demo = site
    assert upload(anon).status_code == 401
    alice = login()
    for fields in (
        [('files',('a.mp4',b'first')),('files',('b.mp4',b'second'))],
        [('field',(None,'value'))],
        [('files',('a.mp4',b'first')),('field',(None,'value'))],
        [('wrong-name',('a.mp4',b'first'))],
    ):
        with mock.patch('gradio.routes.upload_fn', side_effect=AssertionError('shared parser forbidden')):
            r = alice.post('/gradio_api/upload', files=fields)
        assert r.status_code == 400 and r.json()['predictions'] is None
        assert not demo.aval_upload_store._entries
        assert not list(demo.aval_upload_store.root.glob('payload-*'))
    a, b = upload(alice).json()[0], upload(alice).json()[0]
    assert a != b and len(a) == 64 and '/' not in a
    assert len(demo.aval_upload_store._entries) == 2
    assert upload(alice).status_code == 429


@pytest.mark.parametrize('variant', ['file=', 'file/', 'proxy=', 'stream/'])
def test_all_media_route_variants_disabled_for_owner_and_cross_user(site, variant):
    anon, login, demo = site
    alice, bob, second_login = login(), login('bob'), login()
    handle = upload(alice).json()[0]
    path = demo.aval_upload_store._entries[handle].path.as_posix()
    for suffix in (path, handle):
        url = '/gradio_api/'+variant+suffix
        for method in ('get','head'):
            assert getattr(anon,method)(url).status_code == 401
            for c in (alice,bob,second_login):
                response = getattr(c,method)(url)
                assert response.status_code == 403
                assert response.headers['cache-control'] == 'no-store'
                assert b'unit-only media' not in response.content


def test_actual_queue_ownership_before_open_or_delete_and_no_gradio_cache(site):
    _, login, demo = site
    alice, bob, second_login = login(), login('bob'), login()
    handle = upload(alice, data=b'corrupt content despite video MIME').json()[0]
    path = demo.aval_upload_store._entries[handle].path
    with mock.patch('deployment.gradio_app.DeploymentUI.submit', side_effect=AssertionError('must not open')):
        for client in (bob,second_login):
            result = verify(client,demo,handle)
            assert result == {'status':'REJECTED','predictions':None,'error_code':'UPLOAD_NOT_FOUND'}
            assert path.exists()
    assert verify(alice,demo,handle)['predictions'] is None
    assert not path.exists() and handle not in demo.aval_upload_store._entries
    # No backend event takes a File component, so no duplicate cache copies are made.
    import gradio as gr
    assert all(not isinstance(i,gr.File) for f in demo.fns.values() for i in f.inputs if f.fn)
    assert not list(Path(path.parents[1]).rglob('*.mp4'))


def test_content_not_client_size_mime_owner_or_filename_metadata_controls_admission(site):
    _, login, demo = site
    alice = login()
    handle = upload(alice,data=b'the actual server bytes').json()[0]
    path = demo.aval_upload_store._entries[handle].path
    metadata = {'format':{'format_name':'mov,mp4','tags':{'major_brand':'isom'}},'streams':[
        {'codec_type':'video','codec_name':'h264','duration':'10'},
        {'codec_type':'audio','codec_name':'aac','duration':'10'}]}
    def probe(args):
        assert Path(args[-1]).read_bytes() == b'the actual server bytes'
        return 0,json.dumps(metadata).encode(),b''
    with mock.patch('deployment.runtime.shutil.which',return_value='ffprobe'), \
         mock.patch('deployment.runtime._run_bounded_process',side_effect=probe), \
         mock.patch('torch.load',side_effect=AssertionError('public predictions forbidden')):
        result = verify(alice,demo,handle)
    assert result['status']=='MODEL_NOT_READY' and result['predictions'] is None
    assert result['error_code']=='RAW_INFERENCE_GATE_CLOSED' and not path.exists()


def test_verify_cannot_bypass_queue_serialization_via_named_or_indexed_rest(site):
    _,login,demo=site
    client=login();handle=upload(client).json()[0]
    fn=next(f for f in demo.fns.values() if getattr(f.fn,'__name__','')=='check')
    for route in ('/gradio_api/run/check','/gradio_api/api/check','/gradio_api/run/forged-name'):
        with mock.patch('deployment.gradio_app.DeploymentUI.submit',side_effect=AssertionError('queue bypass')):
            response=client.post(route,json={'session_hash':'untrusted','fn_index':fn._id,'data':[handle]})
        assert response.status_code==404
        assert handle in demo.aval_upload_store._entries


@pytest.mark.parametrize('data', [
    [{'path':'https://example.invalid/media','meta':{'_type':'gradio.FileData'}}],
    [{'path':'/models/private.pt','meta':{'_type':'gradio.FileData'}}],
    [[{'path':'/tmp/private.mp4'}]], ['x'*129], [True], ['a','b']])
def test_filedata_injection_rejected_before_framework_cache_traversal(site,data):
    _,login,demo=site
    with mock.patch('gradio.processing_utils.async_move_files_to_cache',
                    side_effect=AssertionError('must reject before traversal')):
        r=login().post('/gradio_api/queue/join',json={'fn_index':0,'session_hash':'claim','data':data})
    assert r.status_code==400 and r.json()['error_code']=='UPLOAD_INVALID_REQUEST'
    assert not demo.aval_upload_store._entries


def test_body_file_limits_chunked_truncation_and_sanitization_leave_no_artifacts(site):
    _,login,demo=site
    alice=login(); store=demo.aval_upload_store
    store.max_bytes=1024
    assert upload(alice,data=b'x'*1025).status_code==413
    assert alice.post('/gradio_api/upload',content=b'x'*(1024+MULTIPART_OVERHEAD+1)).status_code==413
    assert alice.post('/gradio_api/upload',content=b'body',headers={
        'content-type':'multipart/form-data; boundary=x','content-encoding':'gzip'}).status_code==415
    assert alice.post('/gradio_api/upload',content=b'--x\r\nContent-Disposition: form-data; name="files"; filename="a.mp4"\r\n\r\npartial',
        headers={'content-type':'multipart/form-data; boundary=x'}).status_code==400
    assert not list(store.root.glob('payload-*')) and not store._entries


def test_worker_expires_abandoned_or_logged_out_upload_without_another_request(site):
    _,login,demo=site
    alice=login();store=demo.aval_upload_store
    store.ttl=.05
    handle=upload(alice).json()[0];path=store._entries[handle].path
    deadline=time.monotonic()+2.5
    while path.exists() and time.monotonic()<deadline:
        time.sleep(.02)
    assert not path.exists() and handle not in store._entries
    store.ttl=300
    handle=upload(alice).json()[0];path=store._entries[handle].path
    alice.get('/logout')
    deadline=time.monotonic()+2.5
    while path.exists() and time.monotonic()<deadline:
        time.sleep(.02)
    assert not path.exists()


def test_global_capacity_rate_and_active_lease_cleanup_race(tmp_path):
    clock=[0.0]
    store=UploadStore(tmp_path,1024,ttl=1,interval=.01,clock=lambda:clock[0]);store.start()
    def ready(owner):
        handle,entry=store.reserve(owner);entry.path.write_bytes(b'private');store.publish(handle,'x.mp4')
        return handle,entry.path
    try:
        a,path=ready('alice');ready('alice');ready('bob');ready('bob')
        with pytest.raises(UploadError,match='UPLOAD_CAPACITY_REACHED'):store.reserve('carol')
        assert sum(p.stat().st_size for p in store.root.glob('payload-*'))==4*7
        entered,release=threading.Event(),threading.Event()
        def active():
            with store.lease(a,'alice'):
                entered.set();release.wait(2)
                assert path.read_bytes()==b'private'
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(active);assert entered.wait(1)
            clock[0]=2;store.expire();assert path.exists() and len(store._entries)==1
            with pytest.raises(UploadError,match='UPLOAD_NOT_FOUND'):
                with store.lease(a,'alice'):pass
            store.close();assert path.exists()  # never delete an active verification
            release.set();future.result()
        assert not path.exists() and not store._entries and store._directory_lock is None
        replacement=UploadStore(tmp_path,1024);replacement.start();replacement.close()
    finally:store.close()


def test_rate_limit_and_restart_orphan_cleanup_and_exclusive_worker(tmp_path):
    store=UploadStore(tmp_path,1024);store.start()
    with pytest.raises(Exception):UploadStore(tmp_path,1024).start()
    for _ in range(MAX_REQUESTS_PER_MINUTE):
        handle,entry=store.reserve('alice');entry.path.write_bytes(b'partial');store.discard(handle)
    with pytest.raises(UploadError,match='UPLOAD_RATE_LIMIT'):store.reserve('alice')
    store.close()
    orphan=store.root/('payload-'+'a'*64+'.part');orphan.write_bytes(b'old process partial')
    restarted=UploadStore(tmp_path,1024);restarted.start()
    assert not orphan.exists() and not restarted._entries
    restarted.close()


def test_receive_timeout_disconnect_concurrency_and_no_unbounded_body_retention(site):
    _,login,demo=site
    client=login();store=demo.aval_upload_store
    inner=next(r.app for r in client.app.routes if isinstance(r,Mount) and hasattr(r.app,'tokens'))
    cookie='; '.join(f'{k}={v}' for k,v in client.cookies.items()).encode()
    async def run():
        called=[]
        async def forbidden(scope,receive,send):called.append(True)
        middleware=UploadTransportMiddleware(forbidden,store,timeout=.03)
        scope={'type':'http','method':'POST','path':'/gradio_api/upload','app':inner,
               'headers':[(b'cookie',cookie),(b'content-type',b'multipart/form-data; boundary=x')]}
        async def stalled():await asyncio.Event().wait()
        async def issue(receive):
            sent=[]
            async def send(m):sent.append(m)
            await middleware(scope.copy(),receive,send)
            return sent
        first=asyncio.create_task(issue(stalled));second=asyncio.create_task(issue(stalled))
        await asyncio.sleep(.005)
        third=await issue(stalled)
        assert third[0]['status']==429
        assert all(x[0]['status']==408 for x in await asyncio.gather(first,second))
        async def disconnected():return {'type':'http.disconnect'}
        assert await issue(disconnected)==[]
        task=asyncio.create_task(issue(stalled));await asyncio.sleep(.005);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert not store._entries and not list(store.root.glob('payload-*')) and not called
    asyncio.run(run())
