"""Offline live-input adapter using the existing genuine research model worker.

No HTTP server. A local operator mounts private authenticated capture grants;
the strict no-network/read-only-root sandbox remains mandatory. Grant checks
before and after inference prevent revoked, stale or cross-owner publication.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Timer
import time
import uuid

import numpy as np

from deployment.general_inference import GeneralResearchService, ResearchPrediction
from deployment.live.buffer import LiveBufferError, FRAME_TOLERANCE_S, _resample
from deployment.live.research_capture import LIVE_PROFILE, RESEARCH_CAPTURE_PROFILE, capture_profile, clock_report
from deployment.live.research_store import ResearchStore, MAX_RESULT, canonical, digest
from deployment.live.service import image_size, decode_frame, decode_pcm16
from deployment.runtime import DeploymentError

SCHEMA='aval.live-input-research.v1'
STATUS='EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED'
JOB_TIMEOUT=180


def rejected(ticket,owner,code):
    return {'schema':SCHEMA,'status':'REJECTED','ticket':ticket,'owner':owner,
            'error_code':code,'predictions':None,'production_predictions_enabled':False}


def validate_result(result):
    if result.get('schema')!=SCHEMA or result.get('production_predictions_enabled') is not False:
        raise ValueError('invalid live research result')
    if (not re.fullmatch('[a-f0-9]{64}',result.get('ticket',''))
            or not re.fullmatch('[a-f0-9]{64}',result.get('owner',''))):
        raise ValueError('invalid live ownership')
    if result.get('status')=='REJECTED':
        if (set(result)!=set(rejected('','','ERROR')) or result.get('predictions') is not None
                or not re.fullmatch('[A-Z0-9_]{1,96}',result.get('error_code',''))):raise ValueError('invalid live error')
        return
    fields=ResearchPrediction.__dataclass_fields__
    values={k:result[k] for k in fields}
    values.update(schema='aval.general-media-research.v1',status='EXPERIMENTAL_GENERAL_MEDIA_NOT_APPROVED')
    expected=ResearchPrediction(**values).to_dict()
    expected.update(schema=SCHEMA,status=STATUS)
    if ({k:v for k,v in result.items() if k not in ('ticket','owner','live_input','runtime_profile')}!=expected
            or result['live_input']['profile_id']!=LIVE_PROFILE
            or result['live_input']['public_live_readiness']['inference_enabled'] is not False):
        raise ValueError('invalid live research identity')


def decode_capture(capture):
    """Validate the signed producer profile AND actual bytes/clocks, without repairs."""
    if (capture.get('schema')!='aval.live-capture-package.v1' or capture.get('profile_id')!=LIVE_PROFILE
            or capture.get('capture_profile')!=RESEARCH_CAPTURE_PROFILE):
        raise LiveBufferError('LIVE_RESEARCH_PROFILE_UNSUPPORTED')
    sr=capture['sample_rate'];capture_profile({'research':capture['browser_profile']},sr)
    if sr not in (16000,22050,24000,32000,44100,48000):raise LiveBufferError('UNSUPPORTED_SAMPLE_RATE')
    native=decode_pcm16(capture['pcm16'])
    if native.shape!=(10*sr,) or not np.isfinite(native).all():raise LiveBufferError('LIVE_RESEARCH_INCOMPLETE_AUDIO')
    frames=capture['frames']
    seconds=capture.get('source_seconds')
    if (type(seconds) is not list or len(seconds)!=10 or type(seconds[0]) is not int
            or not 0<=seconds[0]<=3 or seconds!=list(range(seconds[0],seconds[0]+10))):
        raise LiveBufferError('LIVE_RESEARCH_SOURCE_TIMELINE_REQUIRED')
    if type(frames) is not list or len(frames)!=10:raise LiveBufferError('LIVE_RESEARCH_INCOMPLETE_FRAMES')
    raw=[];rgb=[]
    for k,f in enumerate(frames):
        t=f['t']
        if (type(t) not in (int,float) or not np.isfinite(t) or abs(t-k)>FRAME_TOLERANCE_S
                or type(f.get('source_t')) not in (int,float) or not np.isfinite(f['source_t'])
                or f['source_t']-seconds[0]!=t
                or type(f.get('frame_id')) is not int or f['frame_id']<0
                or type(f.get('media_time')) not in (int,float) or not np.isfinite(f['media_time'])
                or f.get('timestamp_kind') not in ('captureTime','callbackNow')):
            raise LiveBufferError('LIVE_RESEARCH_UNSYNCHRONIZED_FRAME')
        if k and (f['frame_id']<=frames[k-1]['frame_id'] or f['media_time']<=frames[k-1]['media_time']):
            raise LiveBufferError('LIVE_RESEARCH_DUPLICATE_FRAME')
        image=base64.b64decode(f['image'],validate=True)
        size=image_size(image)
        if not image.startswith(b'\xff\xd8\xff') or size is None or not all(1<=x<=640 for x in size):
            raise LiveBufferError('LIVE_RESEARCH_JPEG_REQUIRED')
        raw.append(image);rgb.append(decode_frame(f['image']))
    clock=clock_report(capture['clock']['anchors'])
    if clock!=capture['clock']:raise LiveBufferError('LIVE_RESEARCH_CLOCK_CHANGED')
    expected=digest((native*32768).astype('<i2').tobytes()+b''.join(raw))
    if capture['capture_source_sha256']!=expected:raise LiveBufferError('LIVE_RESEARCH_CAPTURE_CHANGED')
    wave_raw=_resample(native,sr)
    waveform=np.clip(wave_raw,-1,1).astype(np.float32)
    timing={'achieved_frame_seconds':[f['t'] for f in frames],
        'achieved_source_frame_seconds':[f['source_t'] for f in frames],'source_seconds':seconds,
        'frame_ids':[f['frame_id'] for f in frames],'timestamp_kinds':[f['timestamp_kind'] for f in frames],
        'max_abs_frame_offset_seconds':max(abs(f['t']-k) for k,f in enumerate(frames)),
        'clock':clock,'capture_rate':sr,'resampled':sr!=16000,
        'resample_clipped_samples':int(np.count_nonzero(np.abs(wave_raw)>1)),
        'input_pcm_rail_samples':int(np.count_nonzero((native==-1)|(native==32767/32768))),
        'repairs_applied':False,'historical_shared_clock_approved':False}
    frame_array=np.stack(rgb)
    timing['broker_tensor_comparison']={
        'waveform_exact':digest(waveform.tobytes())==capture['broker_tensors']['waveform_sha256'],
        'frames_exact':digest(frame_array.tobytes())==capture['broker_tensors']['frames_sha256'],
        'replay_waveform_sha256':digest(waveform.tobytes()),'replay_frames_sha256':digest(frame_array.tobytes()),
        'broker_versions':capture['broker_tensors']['versions']}
    return waveform,frame_array,timing,native


class LiveReplayAdapter:
    def __init__(self, worker, store):
        if type(worker) is not GeneralResearchService:
            raise DeploymentError('LIVE_GENUINE_WORKER_REQUIRED')
        self.worker=worker;self.store=store;self.last_audit=None
        # Bind live preprocessing implementation and controller, not just shapes.
        root=Path(__file__).resolve().parents[1]
        self.sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in (
            'live/replay.py','live/research_capture.py','live/research_store.py','live/buffer.py',
            'live/service.py','web/live_capture.js','web/live_timing.js','web/live_research_result.js')}
        self.profile={'live_profile':LIVE_PROFILE,'general_worker_profile':worker.profile,
                      'live_source_sha256':self.sources,'public_scope_approved':False}
        self.profile_sha256=digest(canonical(self.profile))

    def run(self, ticket):
        self.last_audit=None;start=time.monotonic();owner=None
        try:
            grant=self.store.check(ticket);owner=grant['owner']
            if self.store.path(ticket,'.result.json').exists():raise LiveBufferError('LIVE_RESEARCH_ALREADY_PROCESSED')
            capture=self.store.capture(ticket)
            if not self.worker._lock.acquire(blocking=False):raise LiveBufferError('RESOURCE_BUSY')
            try:
                self.worker._integrity()
                root=Path(__file__).resolve().parents[1]
                if any(digest((root/n).read_bytes())!=h for n,h in self.sources.items()):
                    raise LiveBufferError('LIVE_RESEARCH_RUNTIME_CHANGED')
                waveform,frames,timing,native=decode_capture(capture)
                self.store.check(ticket,owner)
                score,logit,audio,visual=self.worker._forward_window(waveform,frames)
                self.worker._integrity()
                self.store.check(ticket,owner)
                if time.monotonic()-start>JOB_TIMEOUT:raise LiveBufferError('LIVE_INFERENCE_TIMEOUT')
                from deployment.release_gate import unapproved_prediction
                result=ResearchPrediction(uuid.uuid4().hex,capture['capture_source_sha256'],
                    self.profile_sha256,score,logit,float(time.monotonic()-start)).to_dict()
                result.update(schema=SCHEMA,status=STATUS,ticket=ticket,owner=owner,
                    live_input={'profile_id':LIVE_PROFILE,'capture_package_sha256':grant['capture_sha256'],
                        'captured_at':capture['created'],'capture_expires_at':capture['expires'],
                        'browser_profile':capture['browser_profile'],'timing':timing,
                        'audio_embedding_shape':list(audio.embeddings.shape),'video_embedding_shape':list(visual.embeddings.shape),
                        'public_live_readiness':unapproved_prediction('live')},runtime_profile=self.profile)
                validate_result(result)
                self.last_audit=(waveform,frames,audio.embeddings,visual.embeddings,native)
            finally:self.worker._lock.release()
            self.store.check(ticket,owner)
            self.store.write(ticket,'.result.json',result)
            try:self.store.check(ticket,owner)
            except LiveBufferError:
                self.store.path(ticket,'.result.json').unlink(missing_ok=True)
                raise
            return result
        except Exception as exc:
            self.last_audit=None
            code=exc.code if isinstance(exc,(LiveBufferError,DeploymentError)) else 'LIVE_RESEARCH_REPLAY_FAILED'
            result=rejected(ticket,owner or '0'*64,code)
            if owner:
                try:
                    self.store.check(ticket,owner)
                    if not self.store.path(ticket,'.result.json').exists():self.store.write(ticket,'.result.json',result)
                except Exception:pass
            return result
        finally:
            try:
                grant=self.store.read(ticket,'.grant.json',MAX_RESULT)
                if grant.get('ticket')==ticket:self.store.delete_capture(ticket)
            except LiveBufferError:pass


def abort_job(store, ticket):
    """CLI watchdog: remove captured media even if auth expired during a hang."""
    try:
        store.delete_capture(ticket)
        grant=store.check(ticket)
        if not store.path(ticket,'.result.json').exists():
            store.write(ticket,'.result.json',rejected(ticket,grant['owner'],'LIVE_INFERENCE_TIMEOUT'))
    finally:os._exit(124)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--spool',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--watch-seconds',type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.watch_seconds<=600:parser.error('watch duration must be 1..600 seconds')
    store=ResearchStore(args.spool,args.key_file)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from filelock import FileLock
    # One local offline worker. No network server or model reload per capture.
    with FileLock(str(args.spool/'.worker.lock'),timeout=0),tempfile.TemporaryDirectory(prefix='aval-live-derived-') as derived:
        startup=Timer(JOB_TIMEOUT,lambda:os._exit(125));startup.daemon=True;startup.start()
        try:worker=GeneralResearchService(args.models,Path(derived));adapter=LiveReplayAdapter(worker,store)
        except Exception:raise RuntimeError('LIVE_GENUINE_WORKER_INITIALIZATION_FAILED') from None
        finally:startup.cancel()
        print('LIVE_REPLAY_WORKER_READY',flush=True)
        deadline=time.monotonic()+args.watch_seconds
        while time.monotonic()<deadline:
            for path in sorted(args.spool.glob('*.capture.json'))[:4]:
                ticket=path.name.removesuffix('.capture.json')
                if not re.fullmatch('[a-f0-9]{64}',ticket):continue
                timer=Timer(JOB_TIMEOUT,abort_job,args=(store,ticket));timer.daemon=True;timer.start()
                try:
                    result=adapter.run(ticket)
                    adapter.last_audit=None  # CLI retains no personal PCM/RGB/embeddings after publication
                    print(json.dumps({'ticket':ticket,'status':result['status']}),flush=True)
                finally:timer.cancel()
            time.sleep(.25)


if __name__=='__main__':main()
