"""Authenticated local capture-only extension; never loads or invokes a model."""
import copy
import base64
import importlib.metadata
import logging
import math

import numpy as np

from deployment.auth import owner_digest
from deployment.live.buffer import FRAME_TOLERANCE_S, LiveBufferError
from deployment.live.service import (LiveCaptureService, MAX_AUDIO_CHUNKS_PER_PUSH,
    MAX_AUDIO_SECONDS_PER_PUSH, MAX_FRAMES_PER_PUSH, MAX_FRAME_BYTES, _b64_limit, image_size)
from deployment.live.research_store import digest

LIVE_PROFILE = 'aval.browser-pcm16-jpeg.offline-research.v1.NOT_APPROVED'
LOGGER = logging.getLogger('avalidate.deployment.live')
MAX_CAPTURE_SECONDS = 13
MAX_CLOCK_OBSERVATIONS = 64
RESEARCH_CAPTURE_PROFILE = {
    'id':LIVE_PROFILE,
    'audio':'WebAudio explicit speakers-mono; implementation-defined AudioContext resampling; '
            'JavaScript Math.round(x*32768), clipped to int16, /32768; '
            'whole-window torchaudio resampling to 16k and clipping when needed. '
            'PCM scale does not establish FFmpeg rounding/resampling equivalence.',
    'video':'canvas longest side <=640, JPEG quality .9; OpenCV JPEG -> RGB -> INTER_LINEAR 224x224',
    'clock':'AudioContext sample count; initial getOutputTimestamp mapping; captureTime when supplied, '
            'otherwise callback time; repeated clock anchors screened against existing 100ms buffer limit',
    'encoder_compatibility':'NOT_VERIFIED', 'physical_sensor_synchronization_verified':False}


def capture_profile(payload, sr):
    p=payload.get('research',{})
    required={'consent','context_rate','device_rate','device_channels','echo_cancellation',
              'noise_suppression','auto_gain_control','mono_mix','browser_resampler'}
    if (type(p) is not dict or set(p)!=required or p['consent'] is not True
            or p['context_rate']!=sr or type(p['device_rate']) is not int or not 8000<=p['device_rate']<=192000
            or p['device_channels'] not in (1,2) or p['mono_mix']!='explicit-speakers-mono'
            or p['browser_resampler']!='AudioContext-implementation-defined'
            or any(p[k] is not False for k in ('echo_cancellation','noise_suppression','auto_gain_control'))):
        raise LiveBufferError('LIVE_RESEARCH_INPUT_PROFILE_REQUIRED')
    return copy.deepcopy(p)


def clock_report(points):
    if len(points)<2 or len(points)>MAX_CLOCK_OBSERVATIONS:raise LiveBufferError('LIVE_RESEARCH_CLOCK_REQUIRED')
    c0,p0=points[0]
    if points[-1][0]-c0<9:raise LiveBufferError('LIVE_RESEARCH_CLOCK_INCOMPLETE')
    if any(c<=pc or p<=pp for (pc,pp),(c,p) in zip(points,points[1:])):
        raise LiveBufferError('LIVE_RESEARCH_CLOCK_DISCONTINUITY')
    offsets=[(c-c0)-(p-p0)/1000 for c,p in points]
    if max(abs(x) for x in offsets)>FRAME_TOLERANCE_S:
        raise LiveBufferError('LIVE_RESEARCH_CLOCK_DRIFT')
    return {'anchors':points,'max_clock_drift_seconds':max(abs(x) for x in offsets),
            'end_clock_drift_seconds':offsets[-1], 'screening_limit_seconds':FRAME_TOLERANCE_S,
            'physical_sensor_synchronization_verified':False}


class ResearchCaptureService(LiveCaptureService):
    def __init__(self, store, **kwargs):
        super().__init__(**kwargs)
        self.store=store;self.evidence={};self.grants={};self.auth_app=None
        # Restart invalidates old logins and all their outstanding replay grants.
        for path in list(store.root.glob('*.grant.json'))[:16]:
            ticket=path.name.removesuffix('.grant.json')
            try:
                store.revoke(ticket)
                store.path(ticket,'.grant.json').unlink(missing_ok=True)
            except LiveBufferError:raise LiveBufferError('LIVE_RESEARCH_UNSAFE_SPOOL') from None

    def bind_auth(self, app):
        with self._lock:
            if self.auth_app is not None and self.auth_app is not app:
                raise LiveBufferError('LIVE_AUTHENTICATION_REQUIRED')
            self.auth_app=app

    def _owner_active(self, owner):
        app=self.auth_app
        return app is not None and getattr(app,'auth',None) and any(
            isinstance(token,str) and isinstance(name,str) and owner_digest(name,token)==owner
            for token,name in getattr(app,'tokens',{}).copy().items())

    def _expire(self):
        super()._expire()
        for token,session in list(self._sessions.items()):
            if not self._owner_active(session.owner):
                self._sessions.pop(token).buffer.clear()
        for token in list(self.evidence):
            if token not in self._sessions:self.evidence.pop(token)
        for ticket,owner in list(self.grants.items()):
            try:
                self.store.check(ticket,owner)
                if not self._owner_active(owner):raise LiveBufferError('LIVE_RESEARCH_REVOKED_OR_EXPIRED')
                self.store.renew(ticket,owner)
            except LiveBufferError:
                try:self.store.revoke(ticket)
                except LiveBufferError:pass
                self.grants.pop(ticket,None)
                for suffix in ('.grant.json','.capture.json','.result.json'):
                    self.store.path(ticket,suffix).unlink(missing_ok=True)

    def _reap(self):
        # Completed grants must also expire/revoke when no capture session remains.
        while not self._shutdown.wait(.5):
            with self._lock:self._expire()
        with self._lock:self._worker=None

    def start(self, owner, payload):
        try:profile=capture_profile(payload,payload.get('sample_rate'))
        except Exception:return {'status':'REJECTED','error_code':'LIVE_RESEARCH_INPUT_PROFILE_REQUIRED'}
        with self._lock:
            if not self._owner_active(owner):return {'status':'REJECTED','error_code':'LIVE_AUTHENTICATION_REQUIRED'}
            # Four active captures/exports globally; restarting does not evade the budget.
            self._expire()
            if len(self._sessions)+len(self.grants)>=self.max_sessions:
                return {'status':'REJECTED','error_code':'LIVE_CAPACITY_REACHED'}
            r=super().start(owner,payload)
            if r['status']=='STARTED':
                self.evidence[r['session']]={'profile':profile,'next_sample':0,'frames':{},'anchors':[],
                                              'started_monotonic':self.clock()}
            return r

    def _progress(self, session, evidence, *, proposed_samples=None, limit=None):
        """Bounded numeric/readiness metadata only: no PCM, images or capabilities."""
        sr=session.buffer.sample_rate
        accepted=evidence['next_sample']
        return {'stage':'capture_ingest','limit_kind':limit,'sample_rate':sr,
                'accepted_audio_samples':accepted,'proposed_audio_samples':proposed_samples,
                'accepted_audio_seconds':accepted/sr,
                'max_audio_samples':MAX_CAPTURE_SECONDS*sr,'max_audio_seconds':MAX_CAPTURE_SECONDS,
                'accepted_clock_observations':len(evidence['anchors']),
                'max_clock_observations':MAX_CLOCK_OBSERVATIONS,
                'server_elapsed_seconds':max(0.,self.clock()-evidence['started_monotonic']),
                'buffer':session.buffer.diagnostics()}

    def _limit_rejection(self, session, committed, proposed_samples, kind):
        diagnostic=self._progress(session,committed,proposed_samples=proposed_samples,limit=kind)
        LOGGER.warning('live_research_capture_limit kind=%s rate=%d accepted_samples=%d proposed_samples=%d valid_bins=%d',
            kind,diagnostic['sample_rate'],diagnostic['accepted_audio_samples'],proposed_samples,
            diagnostic['buffer']['segments_valid'])
        return {'status':'REJECTED','error_code':'LIVE_RESEARCH_CAPTURE_TOO_LONG',
                'capture_diagnostics':diagnostic}

    def _push(self, owner, payload):
        try:
            session=self._get(owner,payload);committed=self.evidence[payload['session']]
            e=copy.deepcopy(committed)
            if not self._owner_active(owner):raise LiveBufferError('LIVE_AUTHENTICATION_REQUIRED')
            audio,frames=payload.get('audio',[]),payload.get('frames',[])
            if (not isinstance(audio,list) or not isinstance(frames,list)
                    or len(audio)>MAX_AUDIO_CHUNKS_PER_PUSH or len(frames)>MAX_FRAMES_PER_PUSH
                    or not all(isinstance(x,dict) for x in audio+frames)):
                raise LiveBufferError('PUSH_TOO_LARGE')
            pcm=[c.get('pcm') for c in audio]
            if (not all(isinstance(p,str) for p in pcm) or
                    sum(map(len,pcm))>_b64_limit(2*int(MAX_AUDIO_SECONDS_PER_PUSH*session.buffer.sample_rate))+4*len(pcm)):
                raise LiveBufferError('PUSH_TOO_LARGE')
            for chunk in audio:
                from deployment.live.service import decode_pcm16
                if chunk.get('start')!=e['next_sample']:raise LiveBufferError('LIVE_RESEARCH_AUDIO_DISCONTINUITY')
                e['next_sample']+=len(decode_pcm16(chunk.get('pcm')))
            anchor=payload.get('research_clock')
            if (type(anchor) is not dict or set(anchor)!={'context','performance'}
                    or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in anchor.values())):
                raise LiveBufferError('LIVE_RESEARCH_CLOCK_REQUIRED')
            point=[anchor['context'],anchor['performance']]
            if e['anchors'] and (point[0]<=e['anchors'][-1][0] or point[1]<=e['anchors'][-1][1]):
                raise LiveBufferError('LIVE_RESEARCH_CLOCK_DISCONTINUITY')
            if len(e['anchors'])>=MAX_CLOCK_OBSERVATIONS:
                return self._limit_rejection(session,committed,e['next_sample'],'clock_observations')
            e['anchors'].append(point)
            for f in frames:
                image=f.get('image')
                if not isinstance(image,str) or len(image)>_b64_limit(MAX_FRAME_BYTES):
                    raise LiveBufferError('INVALID_FRAME_PAYLOAD')
                raw=base64.b64decode(image,validate=True);size=image_size(raw)
                if not raw.startswith(b'\xff\xd8\xff') or size is None or not all(1<=x<=640 for x in size):
                    raise LiveBufferError('LIVE_RESEARCH_JPEG_PROFILE_REQUIRED')
                if (type(f.get('frame_id')) is not int or f['frame_id']<0
                        or type(f.get('media_time')) not in (float,int) or not math.isfinite(f['media_time'])
                        or f.get('timestamp_kind') not in ('captureTime','callbackNow')):
                    raise LiveBufferError('LIVE_RESEARCH_FRAME_ID_REQUIRED')
                t=f['t'];k=round(t)
                if 0<=k<14 and abs(t-k)<=FRAME_TOLERANCE_S:
                    old=e['frames'].get(k)
                    if old is None or abs(t-k)<abs(old['t']-k):e['frames'][k]=copy.deepcopy(f)
            client=payload.get('client',{})
            if any(client.get(k) for k in ('audio_muted','video_muted','track_ended')):
                raise LiveBufferError('LIVE_RESEARCH_DEVICE_UNAVAILABLE')
            if client.get('dropped_video_frames',0)!=0:raise LiveBufferError('LIVE_RESEARCH_DROPPED_FRAMES')
            if e['next_sample']>MAX_CAPTURE_SECONDS*session.buffer.sample_rate:
                return self._limit_rejection(session,committed,e['next_sample'],'audio_samples')
            result=super()._push(owner,payload)
            if result['status']=='CAPTURING':
                self.evidence[payload['session']]=e
                result['research_ready']=session.buffer.ready
                result['capture_progress']=self._progress(session,e)
            return result
        except LiveBufferError as exc:return {'status':'REJECTED','error_code':exc.code}
        except Exception:return {'status':'REJECTED','error_code':'LIVE_RESEARCH_CAPTURE_REJECTED'}

    def seal(self, owner, payload):
        with self._lock:
            try:
                session=self._get(owner,payload);e=self.evidence[payload['session']]
                if not self._owner_active(owner):raise LiveBufferError('LIVE_AUTHENTICATION_REQUIRED')
                snap=session.buffer.snapshot()
                seconds=snap['source_seconds']
                if (len(seconds)!=10 or not 0<=seconds[0]<=3 or
                        seconds!=tuple(range(seconds[0],seconds[0]+10)) or not set(seconds)<=set(e['frames'])):
                    raise LiveBufferError('LIVE_RESEARCH_COMPLETE_WINDOW_REQUIRED')
                # Keep the achieved capture time as well as its local window
                # coordinate; bin labels never replace the actual source time.
                frames=[{**e['frames'][k],'source_t':e['frames'][k]['t'],
                         't':e['frames'][k]['t']-seconds[0]} for k in seconds]
                if (any(b['frame_id']<=a['frame_id'] or b['media_time']<=a['media_time'] for a,b in zip(frames,frames[1:]))
                        or len({f['frame_id'] for f in frames})!=10):
                    raise LiveBufferError('LIVE_RESEARCH_DUPLICATE_FRAME')
                timing=clock_report(e['anchors'])
                native=np.concatenate([session.buffer._seconds[k].audio for k in seconds])
                pcm=(native*32768).astype('<i2').tobytes()
                capture={'schema':'aval.live-capture-package.v1','profile_id':LIVE_PROFILE,
                    'browser_profile':e['profile'],'capture_profile':RESEARCH_CAPTURE_PROFILE,
                    'sample_rate':session.buffer.sample_rate,'pcm16':base64.b64encode(pcm).decode('ascii'),
                    'frames':frames,'clock':timing,'timing':snap['timing'],'source_seconds':list(seconds),
                    'broker_tensors':{'waveform_sha256':digest(snap['waveform'].tobytes()),
                        'frames_sha256':digest(snap['frames'].tobytes()),
                        'versions':{n:importlib.metadata.version(n) for n in ('numpy','torch','torchaudio','opencv-python-headless')}},
                    'capture_source_sha256':digest(pcm+b''.join(base64.b64decode(f['image']) for f in frames))}
                ticket,cancel=self.store.issue(owner,capture);self.grants[ticket]=owner
                self._sessions.pop(payload['session']).buffer.clear();self.evidence.pop(payload['session'])
                return {'status':'RESEARCH_CAPTURE_SEALED','ticket':ticket,'cancel_key':cancel,
                        'profile_id':LIVE_PROFILE,'expires_in_seconds':300,'predictions':None}
            except LiveBufferError as exc:return {'status':'REJECTED','error_code':exc.code,'predictions':None}
            except Exception:return {'status':'REJECTED','error_code':'LIVE_RESEARCH_SEAL_FAILED','predictions':None}

    def cancel(self, owner, payload):
        with self._lock:
            try:
                ticket=payload['ticket']
                if self.grants.get(ticket)!=owner:raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
                self.store.revoke(ticket,owner,payload['cancel_key']);self.grants.pop(ticket,None)
                # A missing grant also fails replay checks. Do not accumulate
                # unbounded inactive ticket files across repeated recordings.
                self.store.path(ticket,'.grant.json').unlink(missing_ok=True)
                return {'status':'RESEARCH_CAPTURE_REVOKED','predictions':None}
            except Exception:return {'status':'REJECTED','error_code':'LIVE_RESEARCH_NOT_FOUND','predictions':None}

    def poll(self, owner, payload):
        with self._lock:
            try:
                self._expire()
                if not self._owner_active(owner) or self.grants.get(payload['ticket'])!=owner:
                    raise LiveBufferError('LIVE_RESEARCH_NOT_FOUND')
                result=self.store.result(payload['ticket'],owner,payload['cancel_key'])
                if result is None:return {'status':'RESEARCH_REPLAY_PENDING','predictions':None}
                # Verify the frozen result ABI again at the authenticated display boundary.
                from deployment.live.replay import validate_result
                validate_result(result)
                return result
            except Exception:return {'status':'REJECTED','error_code':'LIVE_RESEARCH_NOT_FOUND','predictions':None}

    def close(self):
        with self._lock:
            for ticket in list(self.grants):
                try:self.store.revoke(ticket)
                except LiveBufferError:pass
            self.grants.clear();self.evidence.clear()
        super().close()
