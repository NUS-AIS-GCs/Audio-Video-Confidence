"""Live research security/clock tests, never substitute scores or model weights."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from deployment.auth import owner_digest
from deployment.live.buffer import LiveBufferError
from deployment.live.replay import abort_job, decode_capture, LiveReplayAdapter
from deployment.live.research_capture import ResearchCaptureService, LIVE_PROFILE, clock_report
from deployment.live.research_store import ResearchStore, MAX_CAPTURE, MAX_RESULT, canonical


@pytest.fixture
def store(tmp_path):
    key=tmp_path/'research.key';key.write_bytes(bytes(range(32)))
    root=tmp_path/'spool';root.mkdir()
    return ResearchStore(root,key)


def profile(sr=16000):
    return {'consent':True,'context_rate':sr,'device_rate':48000,'device_channels':2,
        'echo_cancellation':False,'noise_suppression':False,'auto_gain_control':False,
        'mono_mix':'explicit-speakers-mono','browser_resampler':'AudioContext-implementation-defined'}


def capture_service(store):
    s=ResearchCaptureService(store)
    app=SimpleNamespace(auth=True,tokens={'a':'user','b':'user'})
    s.bind_auth(app)
    return s,app,owner_digest('user','a'),owner_digest('user','b')


def feed(s,owner,credentials,*,gap=False,duplicate=False,drift=False,skip_first_frame=False,count=22):
    ok,jpeg=cv2.imencode('.jpg',np.full((240,320,3),64,np.uint8));assert ok
    image=base64.b64encode(jpeg).decode()
    pcm=base64.b64encode(np.full(8000,2000,dtype='<i2').tobytes()).decode()
    for i in range(count):
        frames=[] if i%2 or (skip_first_frame and i==0) else [{'t':i/2,'image':image,'frame_id':0 if duplicate else i*15,
            'media_time':i/2,'timestamp_kind':'callbackNow'}]
        p={**credentials,'audio':[{'start':i*8000+(1 if gap and i else 0),'pcm':pcm}],
           'frames':frames,'client':{'dropped_video_frames':0},
           'research_clock':{'context':i/2+1,'performance':1000+i*500+(i*20 if drift else 0)}}
        r=s.push(owner,p)
        if r['status']!='CAPTURING':return r
    return r


def test_signed_store_owner_tampering_and_expiry(store):
    owner='a'*64;ticket,cancel=store.issue(owner,{'schema':'unit fixture; no model inference'})
    assert store.capture(ticket)['owner']==owner
    with pytest.raises(LiveBufferError):store.check(ticket,'b'*64)
    p=store.path(ticket,'.capture.json');body=json.loads(p.read_text());body['body']['owner']='b'*64;p.write_text(json.dumps(body))
    with pytest.raises(LiveBufferError):store.capture(ticket)
    store.revoke(ticket,owner,cancel)
    assert not p.exists()
    with pytest.raises(LiveBufferError):store.check(ticket)


def test_store_cannot_accept_paths_or_oversized_payloads(store):
    for ticket in ('../secret','x'*64,{'path':'/models/E3/best.pt'},'a'*65):
        with pytest.raises(LiveBufferError):store.path(ticket,'.capture.json')
    ticket='a'*64
    store.path(ticket,'.capture.json').write_bytes(b'x'*(MAX_CAPTURE+1))
    with pytest.raises(LiveBufferError):store.capture(ticket)


def test_research_key_and_spool_fail_closed(tmp_path):
    key=tmp_path/'key';key.write_bytes(b'short')
    with pytest.raises(LiveBufferError):ResearchStore(tmp_path,key)


def test_store_expiry_and_duplicate_result(store):
    store.clock=lambda:1000
    ticket,cancel=store.issue('a'*64,{'schema':'unit fixture, no predictions'})
    store.write(ticket,'.result.json',{'ticket':ticket,'owner':'a'*64,'status':'UNIT_NO_INFERENCE'})
    with pytest.raises(FileExistsError):store.write(ticket,'.result.json',{})
    store.clock=lambda:1300
    with pytest.raises(LiveBufferError):store.result(ticket,'a'*64,cancel)


def test_complete_window_signed_and_replay_input_validated(store):
    s,app,a,b=capture_service(store)
    try:
        started=s.start(a,{'sample_rate':16000,'research':profile()});assert started['status']=='STARTED'
        creds={k:started[k] for k in ('session','tab_key')}
        assert feed(s,a,creds)['ready']
        assert s.seal(b,creds)['status']=='REJECTED'
        sealed=s.seal(a,creds);assert sealed['status']=='RESEARCH_CAPTURE_SEALED',sealed
        ticket=sealed['ticket'];capture=store.capture(ticket)
        wave,frames,timing,native=decode_capture(capture)
        assert wave.shape==(160000,) and frames.shape==(10,224,224,3)
        assert timing['repairs_applied'] is False and timing['historical_shared_clock_approved'] is False
        assert not s._sessions
        assert s.poll(b,sealed)['status']=='REJECTED'
        assert s.poll(a,sealed)['status']=='RESEARCH_REPLAY_PENDING'
        assert s.cancel(b,sealed)['status']=='REJECTED'
        assert s.cancel(a,sealed)['status']=='RESEARCH_CAPTURE_REVOKED'
        assert not store.path(ticket,'.capture.json').exists()
        assert not store.path(ticket,'.grant.json').exists()
        assert s.poll(a,sealed)['status']=='REJECTED'
    finally:s.close()


@pytest.mark.parametrize('mutation,expected',[
    ('gap','LIVE_RESEARCH_AUDIO_DISCONTINUITY'),('duplicate','LIVE_RESEARCH_DUPLICATE_FRAME'),
    ('drift','LIVE_RESEARCH_CLOCK_DRIFT')])
def test_drift_gap_duplicate_rejected(store,mutation,expected):
    s,app,a,b=capture_service(store)
    try:
        started=s.start(a,{'sample_rate':16000,'research':profile()});creds={k:started[k] for k in ('session','tab_key')}
        r=feed(s,a,creds,**{mutation:True})
        if r['status']=='CAPTURING':r=s.seal(a,creds)
        assert r['status']=='REJECTED' and r['error_code']==expected,r
        assert not list(store.root.glob('*.capture.json'))
    finally:s.close()


def test_profile_consent_authentication_and_revocation(store):
    s,app,a,b=capture_service(store)
    try:
        assert s.start(a,{'sample_rate':16000})['status']=='REJECTED'
        p=profile();p['consent']=False
        assert s.start(a,{'sample_rate':16000,'research':p})['status']=='REJECTED'
        started=s.start(a,{'sample_rate':16000,'research':profile()});creds={k:started[k] for k in ('session','tab_key')}
        feed(s,a,creds);sealed=s.seal(a,creds);ticket=sealed['ticket']
        app.tokens.pop('a');s._expire()
        assert s.poll(a,sealed)['status']=='REJECTED'
        assert not store.path(ticket,'.capture.json').exists()
    finally:s.close()


def test_missing_clock_and_incomplete_window(store):
    s,app,a,b=capture_service(store)
    try:
        started=s.start(a,{'sample_rate':16000,'research':profile()});creds={k:started[k] for k in ('session','tab_key')}
        assert s.push(a,{**creds,'audio':[],'frames':[]})['error_code']=='LIVE_RESEARCH_CLOCK_REQUIRED'
        assert s.seal(a,creds)['status']=='REJECTED'
        s.stop(a,creds)
        assert s.seal(a,creds)['error_code']=='LIVE_SESSION_NOT_FOUND'
    finally:s.close()


def test_restarts_revoke_completed_grants(store):
    ticket,cancel=store.issue('a'*64,{'schema':'unit no model'})
    s=ResearchCaptureService(store)
    try:
        with pytest.raises(LiveBufferError):store.check(ticket)
        assert not store.path(ticket,'.capture.json').exists()
    finally:s.close()


def test_live_adapter_requires_actual_existing_worker(store):
    with pytest.raises(Exception,match='LIVE_GENUINE_WORKER_REQUIRED'):LiveReplayAdapter(SimpleNamespace(),store)


def test_clock_report_rejects_reverse_and_excessive_drift():
    with pytest.raises(LiveBufferError):clock_report([[1,1000],[.5,11000]])
    with pytest.raises(LiveBufferError):clock_report([[1,1000],[11,12000]])
    assert clock_report([[1,1000],[11,11000]])['max_clock_drift_seconds']==0


def test_broker_heartbeat_is_required_and_cannot_resurrect_expired_lease(store):
    now=[1000.];store.clock=lambda:now[0]
    ticket,_=store.issue('a'*64,{'schema':'unit, no model output'})
    now[0]=1004.;store.renew(ticket,'a'*64)
    now[0]=1008.;assert store.check(ticket)['active']
    now[0]=1009.
    with pytest.raises(LiveBufferError):store.check(ticket)
    with pytest.raises(LiveBufferError):store.renew(ticket,'a'*64)


def test_concurrent_session_budget_and_active_capture_logout_cleanup(store):
    s,app,a,b=capture_service(store)
    try:
        for _ in range(4):assert s.start(a,{'sample_rate':16000,'research':profile()})['status']=='STARTED'
        assert s.start(b,{'sample_rate':16000,'research':profile()})['error_code']=='LIVE_CAPACITY_REACHED'
        app.tokens.pop('a');s._expire()
        assert not s._sessions and not s.evidence
    finally:s.close()


@pytest.mark.parametrize('client',[
    {'dropped_video_frames':1},{'audio_muted':True},{'video_muted':True},{'track_ended':True}])
def test_reported_frame_drop_or_device_loss_rejected(store,client):
    s,app,a,b=capture_service(store)
    try:
        r=s.start(a,{'sample_rate':16000,'research':profile()});creds={k:r[k] for k in ('session','tab_key')}
        result=s.push(a,{**creds,'audio':[],'frames':[],'client':client,
            'research_clock':{'context':1.,'performance':1000.}})
        assert result['status']=='REJECTED'
        assert not s.evidence[r['session']]['anchors']  # rejected push is atomic
    finally:s.close()


def test_push_limits_precede_extension_decode(store):
    s,app,a,b=capture_service(store)
    try:
        r=s.start(a,{'sample_rate':16000,'research':profile()});creds={k:r[k] for k in ('session','tab_key')}
        with patch('deployment.live.service.decode_pcm16',side_effect=AssertionError('must bound before decode')):
            bad=s.push(a,{**creds,'audio':[{'start':0,'pcm':''}]*9,'frames':[]})
            assert bad['error_code']=='PUSH_TOO_LARGE'
        _,encoded=cv2.imencode('.jpg',np.zeros((240,641,3),np.uint8))
        bad=s.push(a,{**creds,'audio':[],'frames':[{'image':base64.b64encode(encoded).decode()}],
                      'research_clock':{'context':1,'performance':1000}})
        assert bad['error_code']=='LIVE_RESEARCH_JPEG_PROFILE_REQUIRED'
    finally:s.close()


def test_watchdog_cleans_media_and_never_publishes_a_timed_out_score(store):
    ticket,cancel=store.issue('a'*64,{'schema':'unit fixture; no model output'})
    with patch('deployment.live.replay.os._exit') as terminated:abort_job(store,ticket)
    terminated.assert_called_once_with(124)
    assert not store.path(ticket,'.capture.json').exists()
    r=store.result(ticket,'a'*64,cancel)
    assert r['error_code']=='LIVE_INFERENCE_TIMEOUT' and r['predictions'] is None and 'score' not in r


def test_later_complete_window_preserves_actual_source_coordinates(store):
    s,app,a,b=capture_service(store)
    try:
        r=s.start(a,{'sample_rate':16000,'research':profile()});creds={k:r[k] for k in ('session','tab_key')}
        assert feed(s,a,creds,skip_first_frame=True,count=24)['ready']
        sealed=s.seal(a,creds);assert sealed['status']=='RESEARCH_CAPTURE_SEALED',sealed
        capture=store.capture(sealed['ticket']);wave,frames,timing,native=decode_capture(capture)
        assert timing['source_seconds']==list(range(1,11))
        assert timing['achieved_source_frame_seconds']==list(range(1,11))
        assert timing['achieved_frame_seconds']==list(range(10))
        assert timing['broker_tensor_comparison']['frames_exact']
        assert timing['broker_tensor_comparison']['waveform_exact']
    finally:s.close()

@pytest.mark.parametrize('count,skip_first,code', [(20, False, None), (19, False, 'LIVE_RESEARCH_COMPLETE_WINDOW_REQUIRED'), (20, True, 'LIVE_WINDOW_DROPPED_FRAME')])
def test_one_shot_seal_requires_exact_complete_window(store, count, skip_first, code):
    service, _, owner, _ = capture_service(store)
    try:
        started = service.start(owner, {'sample_rate':16000, 'research':profile()})
        creds = {k:started[k] for k in ('session', 'tab_key')}
        response = feed(service, owner, creds, count=count, skip_first_frame=skip_first)
        assert response['status'] == 'CAPTURING'
        assert not response['ready']  # rolling grace has not elapsed
        result = service.seal(owner, {**creds, 'complete':True})
        if code:
            assert result['status'] == 'REJECTED' and result['error_code'] == code
            assert not list(store.root.glob('*.capture.json'))
        else:
            assert result['status'] == 'RESEARCH_CAPTURE_SEALED'
            capture = store.capture(result['ticket'])
            assert capture['source_seconds'] == list(range(10))
            assert len(base64.b64decode(capture['pcm16'])) == 320000
            wave, frames, timing, _ = decode_capture(capture)
            assert wave.shape == (160000,) and frames.shape == (10,224,224,3)
            assert timing['repairs_applied'] is False
            assert timing['broker_tensor_comparison']['waveform_exact']
            assert service.seal(owner, {**creds, 'complete':True})['status'] == 'REJECTED'
    finally:
        service.close()


@pytest.mark.parametrize('defect', ['muted', 'gap', 'short', 'long', 'missing', 'offgrid'])
def test_end_of_input_never_repairs_invalid_bins(defect):
    from deployment.live.buffer import LiveBuffer
    buffer = LiveBuffer(16000)
    frames = [(float(k) + (.2 if defect == 'offgrid' else 0), np.ones((224,224,3),np.uint8))
              for k in range(10) if not (defect == 'missing' and k == 9)]
    for second in range(10):
        count = 15999 if defect == 'gap' and second == 9 else 16000
        if defect == 'short' and second == 9: count = 15000
        if defect == 'long' and second == 9: count = 16001
        samples = np.full(count, 0 if defect == 'muted' else .1, np.float32)
        # Send each frame with its corresponding real audio packet.
        matching = [f for f in frames if round(f[0]) == second]
        buffer.add_push([(second * 16000, samples)], matching)
    with pytest.raises(LiveBufferError):
        buffer.finish_window()

@pytest.mark.parametrize('amplitude,zero_second,expected', [(0, None, 'LIVE_WINDOW_MUTED_AUDIO'), (1, None, None),
    (2000, 0, 'LIVE_WINDOW_MUTED_AUDIO'), (2000, 4, 'LIVE_WINDOW_MUTED_AUDIO'), (2000, 9, 'LIVE_WINDOW_MUTED_AUDIO')])
def test_microphone_seal_metadata_preserves_silence_and_quiet_contract(store, amplitude, zero_second, expected):
    service, _, owner, foreign = capture_service(store)
    try:
        started = service.start(owner, {'sample_rate':16000, 'research':profile()})
        creds = {k:started[k] for k in ('session','tab_key')}
        _, jpeg = cv2.imencode('.jpg',np.full((240,320,3),64,np.uint8))
        image = base64.b64encode(jpeg).decode()
        for half_second in range(20):
            second = half_second // 2
            value = 0 if second == zero_second else amplitude
            pcm = base64.b64encode(np.full(8000,value,dtype='<i2').tobytes()).decode()
            frames = [] if half_second % 2 else [{'t':float(second),'image':image,'frame_id':second*30,
                'media_time':float(second),'timestamp_kind':'callbackNow'}]
            result = service.push(owner,{**creds,'audio':[{'start':half_second*8000,'pcm':pcm}],
                'frames':frames,'research_clock':{'context':1+half_second*.5,'performance':1000+half_second*500},
                'client':{'dropped_video_frames':0}})
            assert result['status'] == 'CAPTURING'
        denied = service.seal(foreign,{**creds,'complete':True})
        assert denied['status'] == 'REJECTED' and 'capture_diagnostics' not in denied
        sealed = service.seal(owner,{**creds,'complete':True})
        audio_stats = sealed.get('capture_diagnostics', result['capture_progress'])['audio_seconds']
        assert [s['second'] for s in audio_stats] == list(range(10))
        for s in audio_stats:
            value = 0 if s['second'] == zero_second else amplitude
            assert s['received_samples'] == 16000 and s['complete'] is True
            assert s['nonzero_samples'] == (16000 if value else 0)
            assert s['rms'] == abs(value)/32768 and s['peak_abs'] == abs(value)/32768
            assert s['start_sample'] == s['second']*16000
            assert s['end_sample_exclusive'] == (s['second']+1)*16000
        if expected:
            assert sealed['error_code'] == expected and sealed['predictions'] is None
            d = sealed['capture_diagnostics']
            assert d['accepted_audio_samples'] == 160000
            zeros = [s['second'] for s in d['buffer']['segments'] if s['status']=='muted_audio']
            assert zeros == ([zero_second] if zero_second is not None else list(range(10)))
            text = json.dumps(d)
            for private in ('pcm','image','session','tab_key','owner','cancel_key'):
                assert private not in text
            assert not list(store.root.glob('*.capture.json'))
        else:
            assert sealed['status'] == 'RESEARCH_CAPTURE_SEALED'  # 1 PCM16 unit is not rejected as quiet
    finally:
        service.close()

def test_microphone_summary_excludes_unreceived_zero_initialized_storage(store):
    service, _, owner, _ = capture_service(store)
    try:
        started=service.start(owner,{'sample_rate':16000,'research':profile()})
        creds={k:started[k] for k in ('session','tab_key')}
        pcm=base64.b64encode(np.full(8000,2000,dtype='<i2').tobytes()).decode()
        r=service.push(owner,{**creds,'audio':[{'start':0,'pcm':pcm}],'frames':[],
            'research_clock':{'context':1.,'performance':1000.},'client':{'dropped_video_frames':0}})
        assert r['status']=='CAPTURING'
        stats=r['capture_progress']['audio_seconds']
        assert len(stats)==1 and stats[0]['received_samples']==8000
        assert stats[0]['nonzero_samples']==8000 and stats[0]['complete'] is False
        assert stats[0]['rms']==2000/32768 and stats[0]['peak_abs']==2000/32768
        assert stats[0]['status'] is None
        rejected=service.seal(owner,{**creds,'complete':True})
        assert rejected['status']=='REJECTED' and rejected['error_code']=='LIVE_RESEARCH_COMPLETE_WINDOW_REQUIRED'
        assert rejected['capture_diagnostics']['audio_seconds']==stats
        assert not list(store.root.glob('*.capture.json'))
    finally:
        service.close()
