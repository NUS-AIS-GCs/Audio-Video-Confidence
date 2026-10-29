"""Synthetic native-sample, packet timing and finalization regression matrix.

No device APIs, recordings, model stubs or inference scores. Actual authenticated
capture service, JPEG/PCM transport, strict clock checks and signed tickets.
"""
import base64
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from deployment.auth import owner_digest
from deployment.live.buffer import LiveBuffer
from deployment.live.replay import decode_capture
from deployment.live.research_capture import ResearchCaptureService
from deployment.live.research_store import ResearchStore


def scenario(tmp_path, *, batch=0., muted=False, missing=False, offset=0.,
             sr=16000, step=.25, wall_scale=1., publish=True):
    key=tmp_path/'key';key.write_bytes(bytes(range(32)))
    root=tmp_path/'spool';root.mkdir();store=ResearchStore(root,key)
    wall=[0.];service=ResearchCaptureService(store,clock=lambda:wall[0])
    service.bind_auth(SimpleNamespace(auth=True,tokens={'unit':'user'}))
    owner=owner_digest('user','unit')
    profile={'consent':True,'context_rate':sr,'device_rate':48000,'device_channels':2,
        'echo_cancellation':False,'noise_suppression':False,'auto_gain_control':False,
        'mono_mix':'explicit-speakers-mono','browser_resampler':'AudioContext-implementation-defined'}
    started=service.start(owner,{'sample_rate':sr,'research':profile})
    credentials={k:started[k] for k in ('session','tab_key')}
    _,encoded=cv2.imencode('.jpg',np.full((240,320,3),64,np.uint8))
    image=base64.b64encode(encoded).decode();sent=set();pos=0;responses=[]
    try:
        while pos<14*sr:
            count=round((batch if batch and pos==5*sr else step)*sr);end=pos+count
            wall[0]=wall_scale*end/sr
            frames=[]
            for k in range(14):
                if k not in sent and k+.1<=end/sr:
                    sent.add(k)
                    if not missing:frames.append({'t':k+offset,'image':image,'frame_id':k*30,
                        'media_time':float(k),'timestamp_kind':'callbackNow'})
            pcm=np.full(count,0 if muted else 4096,dtype='<i2')
            response=service.push(owner,{**credentials,'audio':[{'start':pos,'pcm':base64.b64encode(pcm).decode()}],
                'frames':frames,'client':{'dropped_video_frames':0},
                'research_clock':{'context':end/sr+1,'performance':1000*end/sr+1000}})
            responses.append(response)
            if response['status']!='CAPTURING':break
            pos=end
            if response['ready']:
                if publish:
                    sealed=service.seal(owner,credentials)
                    assert sealed['status']=='RESEARCH_CAPTURE_SEALED',sealed
                    capture=store.capture(sealed['ticket'])
                    wave,frames,timing,_=decode_capture(capture)
                    assert wave.shape==(160000,) and frames.shape==(10,224,224,3)
                    assert timing['broker_tensor_comparison']['waveform_exact']
                    assert timing['broker_tensor_comparison']['frames_exact']
                    service.cancel(owner,sealed)
                    assert not list(root.glob('*.capture.json'))
                break
        return responses
    finally:service.close()


@pytest.mark.parametrize('batch',[0.,1.49,1.5,1.51,2.])
@pytest.mark.parametrize('sr',[16000,48000])
def test_batched_co_delivered_frames_do_not_become_late(tmp_path,batch,sr):
    # 1.5 seconds is the browser backlog boundary; 2 is the unchanged server
    # per-push bound. Before the fix >=1.5 lost frame 5 and failed at 13 seconds.
    rows=scenario(tmp_path,batch=batch,sr=sr)
    assert rows[-1]['status']=='CAPTURING' and rows[-1]['ready']
    assert rows[-1]['capture_progress']['accepted_audio_seconds']<=13
    assert rows[-1]['counters']['late_frames']==0


@pytest.mark.parametrize('wall_scale',[.5,1.,2.])
def test_native_sample_duration_is_independent_of_receipt_wall_clock(tmp_path,wall_scale):
    rows=scenario(tmp_path,wall_scale=wall_scale)
    progress=rows[-1]['capture_progress']
    assert progress['accepted_audio_samples']==168000
    assert progress['accepted_audio_seconds']==10.5
    assert progress['server_elapsed_seconds']==10.5*wall_scale


@pytest.mark.parametrize('case,kwargs,status',[
    ('muted',{'muted':True},'muted_audio'),
    ('missing',{'missing':True},'dropped_frame'),
    ('offgrid',{'offset':.2},'dropped_frame')])
def test_incomplete_inputs_still_reject_at_exact_sample_limit(tmp_path,case,kwargs,status):
    rows=scenario(tmp_path,**kwargs)
    rejected=rows[-1]
    assert rejected['status']=='REJECTED' and rejected['error_code']=='LIVE_RESEARCH_CAPTURE_TOO_LONG'
    d=rejected['capture_diagnostics']
    assert d['stage']=='capture_ingest' and d['limit_kind']=='audio_samples'
    assert d['accepted_audio_samples']==d['max_audio_samples']==208000
    assert d['proposed_audio_samples']==212000
    assert d['accepted_clock_observations']==52 and not d['buffer']['ready']
    assert status in {s['status'] for s in d['buffer']['segments']}
    assert len(d['buffer']['segments'])==10
    text=json.dumps(d)
    assert all(x not in text for x in ('pcm','image','session','tab_key','owner','ticket','cancel_key'))
    assert not list((tmp_path/'spool').glob('*.capture.json'))


def test_clock_observation_budget_is_distinct_from_sample_duration(tmp_path):
    rows=scenario(tmp_path,step=.125)
    d=rows[-1]['capture_diagnostics']
    assert d['limit_kind']=='clock_observations'
    assert d['accepted_clock_observations']==d['max_clock_observations']==64
    assert d['accepted_audio_seconds']==8 and d['proposed_audio_samples']<d['max_audio_samples']


def test_batching_does_not_resurrect_frames_from_a_previous_finalized_packet():
    b=LiveBuffer(16000);audio=np.full(24000,.125,np.float32);frame=np.zeros((224,224,3),np.uint8)
    b.add_push([(0,audio)],[])
    assert b._statuses[0]=='dropped_frame'
    b.add_push([],[(0.,frame)])
    assert b._statuses[0]=='dropped_frame' and b.counters['late_frames']==1


def test_invalid_batch_does_not_commit_partial_state(tmp_path):
    # Existing service transaction and late-frame tests also cover this path;
    # directly verify a valid audio prefix + malformed future frame is atomic.
    from deployment.live.service import LiveCaptureService
    s=LiveCaptureService();owner='unit-owner'
    try:
        started=s.start(owner,{'sample_rate':16000});creds={k:started[k] for k in ('session','tab_key')}
        _,encoded=cv2.imencode('.jpg',np.zeros((224,224,3),np.uint8))
        result=s.push(owner,{**creds,'audio':[{'start':0,'pcm':base64.b64encode(np.ones(24000,'<i2').tobytes()).decode()}],
            'frames':[{'t':100.,'image':base64.b64encode(encoded).decode()}]})
        assert result['status']=='REJECTED'
        b=s._sessions[creds['session']].buffer
        assert b._next_sample is None and not b._seconds and b.counters['audio_chunks']==0
    finally:s.close()
