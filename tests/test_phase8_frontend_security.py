"""Stage 2F: real pinned Gradio HTTP boundary and bounded capture lifecycle.

All media here is synthetic. No checkpoint inference or held-out scoring occurs.
"""
import asyncio
import base64
import copy
import json
import secrets
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import cv2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deployment.gradio_app import create_app
from deployment.live import service as live
from deployment.live.security import LiveTransportMiddleware, MAX_LIVE_BODY_BYTES, live_app_kwargs
from deployment.ui.upload_section import public_admission_result


def pcm():
    return base64.b64encode(np.full(1600, 1000, dtype='<i2').tobytes()).decode()


def capability(started):
    return {key: started[key] for key in ('session', 'tab_key')}


@pytest.fixture
def http_app(tmp_path, monkeypatch):
    import gradio as gr
    password = secrets.token_urlsafe(24)
    monkeypatch.setenv('AVAL_ACCESS_MODE', 'gradio_auth')
    monkeypatch.setenv('AVAL_GRADIO_USERNAME', 'alice')
    monkeypatch.setenv('AVAL_GRADIO_PASSWORD', password)
    monkeypatch.setenv('AVAL_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('GRADIO_TEMP_DIR', str(tmp_path / 'uploads'))
    demo, config = create_app()
    service = demo.aval_live_service
    widget = next(b for b in demo.blocks.values() if isinstance(b, gr.HTML) and len(b.server_fns) == 4)
    app = gr.mount_gradio_app(FastAPI(), demo, path='/', auth=[('alice', password), ('bob', password)],
        app_kwargs=live_app_kwargs(service),
        max_file_size=config.max_upload_bytes, blocked_paths=[str(config.model_root)], show_error=False)
    clients = []
    with TestClient(app) as client:
        clients.append(client)
        def login(username):
            c = TestClient(app)
            c.portal = client.portal
            assert c.post('/login', data={'username': username, 'password': password}).status_code == 200
            clients.append(c)
            return c
        def call(c, tab, fn, data, **kwargs):
            body = json.dumps({'component_id': widget._id, 'session_hash': tab, 'fn_name': fn, 'data': data})
            headers = {'content-type': 'application/json', **kwargs.pop('headers', {})}
            return c.post('/gradio_api/component_server', content=body.encode('ascii'),
                          headers=headers, **kwargs)
        yield client, login, call, service, demo
        for c in clients[1:]:
            c.close()
    assert not service._sessions
    assert service._worker is None or not service._worker.is_alive()
    demo.close()


def test_http_authentication_and_no_anonymous_fallback(http_app):
    anon, login, call, service, demo = http_app
    assert anon.get('/config').status_code == 401
    assert anon.post('/gradio_api/upload', files={'files': ('unit.mp4', b'unit')}).status_code == 401
    assert call(anon, 'spoof', 'live_start', {'sample_rate': 16000, 'username': 'alice'}).status_code == 401
    assert anon.post('/gradio_api/component_server', content=b'{}', headers={
        'content-type':'application/json', 'content-length':str(MAX_LIVE_BODY_BYTES + 1)}).status_code == 401
    bad = TestClient(anon.app)
    try:
        assert bad.post('/login', data={'username': 'alice', 'password': 'invalid'}).status_code == 400
    finally:
        bad.close()
    alice = login('alice')
    assert alice.get('/config').status_code == 200
    started = call(alice, 'client-claim', 'live_start', {'sample_rate': 16000, 'owner': 'bob'}).json()
    assert started['status'] == 'STARTED'
    owner = service._sessions[started['session']].owner
    assert len(owner) == 64 and owner not in ('anonymous', 'alice', 'bob', 'client-claim')
    widget = next(b for b in demo.blocks.values() if len(getattr(b, 'server_fns', [])) == 4)
    fn = next(f for f in widget.server_fns if f.__name__ == 'live_start')
    forged = type('Forged', (), {'username': 'alice', 'session_hash': 'claim'})()
    assert fn({'sample_rate': 16000}, forged) == {
        'status': 'MODEL_NOT_READY', 'predictions': None, 'error_code': 'LIVE_AUTHENTICATION_REQUIRED'}
    with mock.patch('deployment.auth.importlib.metadata.version', return_value='unsupported'):
        r = call(alice, 'tab', 'live_start', {'sample_rate':16000})
        assert r.status_code == 401 and r.json()['predictions'] is None


def test_http_tabs_and_users_are_isolated_even_with_spoofed_hashes(http_app):
    _, login, call, service, _ = http_app
    alice = login('alice')
    # Same authentication cookie, different tabs. Body session_hash is deliberately identical.
    a = call(alice, 'untrusted-same-hash', 'live_start', {'sample_rate': 16000}).json()
    b = call(alice, 'untrusted-same-hash', 'live_start', {'sample_rate': 16000}).json()
    bob = login('bob')
    c = call(bob, 'untrusted-same-hash', 'live_start', {'sample_rate': 16000}).json()
    assert service.active_sessions == 3
    for started, client in ((a, alice), (b, alice), (c, bob)):
        assert call(client, 'changed-hash', 'live_push', capability(started)).json()['status'] == 'CAPTURING'
    wrong_tab = {'session': a['session'], 'tab_key': b['tab_key']}
    wrong_user = capability(a)  # Even both capabilities do not authorize a different login.
    for fn in ('live_push', 'live_stop', 'live_check'):
        for client, payload in ((alice, wrong_tab), (bob, wrong_user), (alice, {'session': a['session']})):
            rejected = call(client, 'spoof', fn, {**payload, 'owner': 'alice', 'username': 'alice'},
                            headers={'X-User': 'alice', 'X-Gradio-User': 'alice'}).json()
            assert rejected['error_code'] == 'LIVE_SESSION_NOT_FOUND'
            assert 'session' not in rejected and 'timing' not in rejected
    # Concurrent tab pushes must not overwrite or terminate either capture.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, alice, 'tab', 'live_push', capability(s)) for s in (a, b)]
        assert all(f.result().json()['status'] == 'CAPTURING' for f in futures)
        futures = [pool.submit(call, client, 'tab', 'live_push', capability(s))
                   for client, s in ((alice, b), (bob, c))]
        assert all(f.result().json()['status'] == 'CAPTURING' for f in futures)
    assert call(alice, 'tab', 'live_stop', capability(a)).json()['status'] == 'STOPPED'
    assert call(alice, 'tab', 'live_push', capability(b)).json()['status'] == 'CAPTURING'
    assert call(bob, 'tab', 'live_push', capability(c)).json()['status'] == 'CAPTURING'


def test_http_repeated_creation_is_bounded_and_restart_requires_capability(http_app):
    _, login, call, service, _ = http_app
    alice = login('alice')
    sessions = [call(alice, str(i), 'live_start', {'sample_rate': 16000}).json() for i in range(4)]
    assert call(alice, 'fifth', 'live_start', {'sample_rate': 16000}).json()['error_code'] == 'LIVE_CAPACITY_REACHED'
    assert call(alice, 'spoof', 'live_start', {'sample_rate': 16000, 'session': sessions[0]['session']}).json()['error_code'] == 'LIVE_SESSION_NOT_FOUND'
    restarted = call(alice, 'tab', 'live_start', {'sample_rate': 16000, **capability(sessions[0])}).json()
    assert restarted['status'] == 'STARTED' and restarted['session'] != sessions[0]['session']
    assert service.active_sessions == 4


def test_http_limits_run_before_json_and_service_decoding(http_app):
    _, login, call, service, _ = http_app
    alice = login('alice')
    with mock.patch.object(service, 'start', side_effect=AssertionError('must not dispatch')):
        r = alice.post('/gradio_api/component_server', content=b'{',
            headers={'content-type': 'application/json', 'content-length': str(MAX_LIVE_BODY_BYTES + 1)})
        assert r.status_code == 413 and r.json()['error_code'] == 'LIVE_BODY_TOO_LARGE'
        assert alice.post('/gradio_api/component_server', content=b'{}',
            headers={'content-type': 'application/json', 'content-encoding': 'gzip'}).status_code == 415
        for field, value in [('session_hash', 'x' * 129), ('session_hash', None),
                             ('session_hash', ['claim']), ('fn_name', 'x' * 65), ('component_id', True)]:
            envelope = {'component_id':0, 'session_hash':'unit', 'fn_name':'live_start', 'data':{'sample_rate':16000}}
            envelope[field] = value
            r = alice.post('/gradio_api/component_server', json=envelope)
            assert r.status_code == 400 and r.json()['error_code'] == 'LIVE_INVALID_BODY'
    assert service.active_sessions == 0


def test_http_login_and_tab_capabilities_are_independent(http_app):
    _, login, call, _, _ = http_app
    a, another_login = login('alice'), login('alice')
    started = call(a, 'tab', 'live_start', {'sample_rate': 16000}).json()
    for key in ('', 'non-ascii-☃', '\ud800', 'x' * 129):
        r = call(a, 'tab', 'live_stop', {'session': started['session'], 'tab_key': key}).json()
        assert r == {'status': 'REJECTED', 'error_code': 'LIVE_SESSION_NOT_FOUND'}
    assert call(another_login, 'tab', 'live_stop', capability(started)).json()['error_code'] == 'LIVE_SESSION_NOT_FOUND'
    assert call(a, 'tab', 'live_push', capability(started)).json()['status'] == 'CAPTURING'


def test_http_private_artifacts_remain_blocked(http_app, tmp_path):
    anon, login, _, _, _ = http_app
    private = tmp_path / 'models' / 'unit-checkpoint.pt'
    private.parent.mkdir(); private.write_bytes(b'unit-only fixture, no checkpoint')
    path = '/gradio_api/file=' + private.as_posix()
    assert anon.get(path).status_code == 401
    assert login('alice').get(path).status_code == 403


def test_http_upload_then_public_handler_remains_fail_closed(http_app, tmp_path):
    _, login, _, _, demo = http_app
    alice = login('alice')
    uploaded = alice.post('/gradio_api/upload', files={'files': ('unit.mp4', b'unit-only container fixture', 'video/mp4')})
    assert uploaded.status_code == 200
    path = uploaded.json()[0]
    # The transport now returns a session-owned opaque handle, never a cache path.
    entry = demo.aval_upload_store._entries[path]
    import gradio as gr
    from starlette.requests import Request
    from starlette.routing import Mount
    inner = next(r.app for r in alice.app.routes if isinstance(r, Mount) and hasattr(r.app, 'tokens'))
    request = gr.Request(Request({'type':'http', 'app':inner, 'headers':[(b'cookie',
        '; '.join(f'{k}={v}' for k,v in alice.cookies.items()).encode())],
        'aval.upload_guard':demo.aval_upload_store}))
    metadata = {'format': {'format_name': 'mov,mp4', 'tags': {'major_brand': 'isom'}},
                'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'duration': '10'},
                            {'codec_type': 'audio', 'codec_name': 'aac', 'duration': '10'}]}
    fn = next(f.fn for f in demo.fns.values() if getattr(f.fn, '__name__', '') == 'check')
    with mock.patch('deployment.runtime.shutil.which', return_value='unit-ffprobe'), \
         mock.patch('deployment.runtime._run_bounded_process', return_value=(0, json.dumps(metadata).encode(), b'')), \
         mock.patch('torch.load', side_effect=AssertionError('public upload must not load a model')):
        _, (_, result, button) = list(fn(path, request))
    assert result['status'] == 'MODEL_NOT_READY' and result['predictions'] is None
    assert result['error_code'] == 'RAW_INFERENCE_GATE_CLOSED'
    assert button['interactive'] is False
    assert not entry.path.exists() and path not in demo.aval_upload_store._entries


@pytest.mark.parametrize('bad_start', [-1, 1.5, True, 10**100, None])
def test_http_invalid_timestamp_does_not_mutate_prior_chunks(http_app, bad_start):
    _, login, call, service, _ = http_app
    alice = login('alice'); started = call(alice, 'tab', 'live_start', {'sample_rate': 16000}).json()
    session = service._sessions[started['session']]
    before = copy.deepcopy(session.buffer.counters)
    r = call(alice, 'tab', 'live_push', {**capability(started), 'audio': [
        {'start': 0, 'pcm': pcm()}, {'start': bad_start, 'pcm': pcm()}]}).json()
    assert r['status'] == 'REJECTED'
    assert session.buffer.counters == before and not session.buffer._seconds


@pytest.mark.parametrize('bad_time', [None, '0', float('nan'), 1e30, True])
def test_http_invalid_frame_timestamp_rolls_back_audio_and_diagnostics(http_app, bad_time):
    _, login, call, service, _ = http_app
    alice = login('alice'); started = call(alice, 'tab', 'live_start', {'sample_rate':16000}).json()
    session = service._sessions[started['session']]
    old_seen = session.last_seen
    image = base64.b64encode(cv2.imencode('.png', np.zeros((8,8,3), np.uint8))[1]).decode()
    rejected = call(alice, 'tab', 'live_push', {**capability(started),
        'audio':[{'start':0, 'pcm':pcm()}], 'frames':[{'t':bad_time, 'image':image}],
        'client':{'video_fps':29.0}}).json()
    assert rejected['status'] == 'REJECTED'
    assert not session.buffer._seconds and not session.client and session.last_seen == old_seen


def test_push_stop_race_is_serialized_and_cannot_resurrect(monkeypatch):
    service = live.LiveCaptureService()
    entered, release = threading.Event(), threading.Event()
    original = live.decode_pcm16
    def held_decode(value):
        entered.set(); assert release.wait(3)
        return original(value)
    monkeypatch.setattr(live, 'decode_pcm16', held_decode)
    started = service.start('trusted-owner', {'sample_rate': 16000})
    cap = capability(started)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            push = pool.submit(service.push, 'trusted-owner', {**cap, 'audio': [{'start': 0, 'pcm': pcm()}]})
            assert entered.wait(3)
            stop = pool.submit(service.stop, 'trusted-owner', cap)
            assert not stop.done()
            release.set()
            assert push.result()['status'] == 'CAPTURING'
            assert stop.result()['status'] == 'STOPPED'
        assert not service._sessions
        assert service.push('trusted-owner', cap)['error_code'] == 'LIVE_SESSION_NOT_FOUND'
    finally:
        release.set(); service.close()


def test_expiry_releases_memory_without_registry_access():
    service = live.LiveCaptureService(idle_timeout_s=0.08)
    started = service.start('trusted-owner', {'sample_rate': 16000})
    try:
        service.push('trusted-owner', {**capability(started), 'audio': [{'start': 0, 'pcm': pcm()}]})
        buffer = service._sessions[started['session']].buffer
        assert buffer._seconds
        deadline = time.monotonic() + 2
        while buffer._seconds and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not buffer._seconds and not service._sessions
    finally:
        service.close()


def test_chunked_body_timeout_and_inflight_capacity_before_parser():
    async def scenario():
        parsed = []
        async def downstream(scope, receive, send):
            parsed.append(await receive())
        service = live.LiveCaptureService()
        middleware = LiveTransportMiddleware(downstream, service, body_limit=256, body_timeout=0.05, max_requests=1)
        scope = {'type': 'http', 'path': '/gradio_api/component_server',
                 'headers': [(b'content-type', b'application/json'),
                             (b'cookie', b'access-token-unit=unit-token')],
                 'app': SimpleNamespace(auth={'fixture':'unit'}, auth_dependency=None,
                                        tokens={'unit-token':'fixture'}, cookie_id='unit')}
        async def run(receive):
            messages = []
            async def send(m): messages.append(m)
            await middleware(dict(scope), receive, send)
            return messages
        async def must_not_read():
            raise AssertionError('anonymous requests must not read a body')
        scope['app'].auth = None
        assert (await run(must_not_read))[0]['status'] == 401
        scope['app'].auth = {'fixture':'unit'}
        chunks = iter([{'type': 'http.request', 'body': b'x' * 129, 'more_body': True},
                       {'type': 'http.request', 'body': b'y' * 128, 'more_body': False}])
        async def chunked(): return next(chunks)
        response = await run(chunked)
        assert response[0]['status'] == 413 and not parsed
        entered = asyncio.Event()
        async def stalled():
            entered.set(); await asyncio.Event().wait()
        first = asyncio.create_task(run(stalled))
        await entered.wait()
        second = await run(stalled)
        assert second[0]['status'] == 429
        assert (await first)[0]['status'] == 408 and not parsed
        async def small():
            return {'type': 'http.request', 'body': json.dumps({'component_id':0, 'session_hash':'unit',
                    'fn_name':'live_stop', 'data':{}}).encode(), 'more_body': False}
        await run(small)
        assert len(parsed) == 1  # capacity restored after timeout and oversized body
        service.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('result', [None, {'status': []},
    {'status': 'EXPERIMENTAL_INTERNAL_TRAIN_ONLY', 'predictions': {'E3': {'score': 0.9}}, 'path': '/private'},
    {'status': 'PREDICTED', 'predictions': {'E3': 0.5}}])
def test_upload_technical_details_cannot_promote_internal_results(result):
    clean = public_admission_result(result)
    assert clean['status'] == 'MODEL_NOT_READY' and clean['predictions'] is None
    assert 'score' not in json.dumps(clean) and '/private' not in json.dumps(clean)


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js unavailable')
def test_browser_security_controller_and_preview():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([shutil.which('node'), '--test', str(root/'tests/js/test_live_security.js')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
