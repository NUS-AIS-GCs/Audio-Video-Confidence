"""TRAIN-whitelisted, local experimental backend. No public inference route.

The Phase 6 exact-parity pipeline and Phase 8 upload/readiness handlers remain
unchanged. Achieved video PTS are evidence, never fabricated shared-clock proof.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from threading import Lock
import time
import uuid

import numpy as np
import soundfile as sf
import torch

from deployment.audio_bundle import VerifiedAudioPredictor
from deployment.video_bundle import VerifiedVideoPredictor
from deployment.paired_diagnostics import InternalTrainPairStudy
from deployment.train_manifest import load_manifest, AUDIO_CARD, VIDEO_CARD, PERMITTED_IDS
from deployment.runtime import (DeploymentConfig, DeploymentError, MediaInfo, MediaStream,
    ReadinessReport, ReadinessIssue, MAX_PROBE_ALLOCATION_BYTES,
    _run_bounded_process, validate_upload)
from deployment.train_video_diagnostics import sample_train_frames
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact, EncoderOutput
from fusion.inference import CANDIDATES
from fusion.inference_contracts import BIN_STARTS
from fusion.mlp_training import model_fingerprint
from fusion.model_interface import validate_live_model


SCHEMA = 'aval.internal-train-inference.v1'
SEMANTICS = 'uncalibrated sigmoid score for original pairing provenance'
STAGES = ('media_validation','integrity_pre','audio_decode','audio_preprocess',
          'audio_encoder','video_decode','video_preprocess','video_encoder',
          'alignment','E3','A2','integrity_post')


class Timings:
    def __init__(self):
        self.values = {}
    def call(self, name, function):
        wall, cpu = time.perf_counter(), time.process_time()
        result = function()
        self.values[name] = {'wall_seconds':time.perf_counter()-wall,
                             'process_cpu_seconds':time.process_time()-cpu}
        return result


def unavailable():
    return ReadinessReport('MODEL_NOT_READY',(
        ReadinessIssue('raw_media_decoder','NOT_VERIFIED'),
        ReadinessIssue('raw_to_train_export_parity','NOT_VERIFIED')),False)


def error_result(code, status='REJECTED'):
    return {'schema':SCHEMA,'status':status,'predictions':None,'error_code':code,
            'production_predictions_enabled':False}


def probe_with_clocks(path, executable):
    command=[str(executable),'-v','error','-max_alloc',str(MAX_PROBE_ALLOCATION_BYTES),
        '-show_entries','format=format_name,start_time:format_tags=major_brand,compatible_brands:'
        'stream=codec_type,codec_name,duration,start_time,width,height,r_frame_rate',
        '-of','json',str(path)]
    code,stdout,_stderr=_run_bounded_process(command)
    if code:raise DeploymentError('UNSUPPORTED_UPLOAD')
    try:
        data=json.loads(stdout)
        streams=data['streams'];fmt=data['format']
        media=MediaInfo(tuple(fmt['format_name'].split(',')),tuple(sorted(
            (k.lower(),v.lower()) for k,v in fmt.get('tags',{}).items())),
            tuple(MediaStream(s['codec_type'],s['codec_name'],float(s['duration'])) for s in streams))
        if len(streams)!=2 or {s['codec_type'] for s in streams}!={'audio','video'}:
            raise ValueError
        starts=[float(fmt['start_time'])]+[float(s['start_time']) for s in streams]
        if any(not math.isfinite(x) or abs(x)>1e-6 for x in starts):
            raise DeploymentError('MODALITY_CLOCK_MISMATCH')
        video=next(s for s in streams if s['codec_type']=='video')
        width,height=video['width'],video['height']
        if (type(width) is not int or type(height) is not int or width<1 or height<1
                or width>4096 or height>2160):
            raise DeploymentError('UNSUPPORTED_VIDEO_DIMENSIONS')
        numerator,denominator=map(int,video['r_frame_rate'].split('/'))
        fps=numerator/denominator
        if not math.isfinite(fps) or fps<1 or fps>120:
            raise DeploymentError('UNSUPPORTED_FRAME_RATE')
        return media,{'audio_start_seconds':float(next(s for s in streams if s['codec_type']=='audio')['start_time']),
                      'video_start_seconds':float(video['start_time']),'video_fps':fps,
                      'video_width':width,'video_height':height}
    except DeploymentError:raise
    except (ValueError,KeyError,TypeError,AttributeError,ZeroDivisionError,StopIteration):
        raise DeploymentError('MALFORMED_MEDIA_METADATA') from None


class InternalTrainBackend:
    """Load authentic models once; serialize requests; retain full integrity checks.

    Only the explicitly permitted TRAIN IDs and SHA-bound bytes are accepted.
    This class is not imported or called by the user-facing Gradio application.
    """
    def __init__(self, models, inputs, manifest_sha256, ffmpeg, ffprobe, derived):
        self.models,self.inputs=Path(models),Path(inputs)
        self._lock=Lock();self._completed=0;self.initialization=Timings()
        self.manifest=self.initialization.call('manifest_authentication',lambda:load_manifest(
            self.inputs/'manifest.json',manifest_sha256,self.models,self.inputs))
        self.audio=self.initialization.call('audio_model_loading',lambda:VerifiedAudioPredictor(
            Artifact(str(self.models/'audio/model_card.json'),AUDIO_CARD),derived))
        self.video=self.initialization.call('video_model_loading',lambda:VerifiedVideoPredictor(
            Artifact(str(self.models/'video/model_card.json'),VIDEO_CARD)))
        self.ffmpeg,self.ffprobe=Path(ffmpeg),Path(ffprobe)
        self.decoder_artifacts=tuple(Artifact(str(p),hashlib.sha256(p.read_bytes()).hexdigest())
                                     for p in (self.ffmpeg,self.ffprobe))
        self.sources={s:Artifact(str(self.models/'video/parity_samples'/(s.rsplit('__',1)[0]+'.mp4')),
            self.manifest['model_files']['video/parity_samples/'+s.rsplit('__',1)[0]+'.mp4']) for s in PERMITTED_IDS}
        self.study=self.initialization.call('fusion_and_train_reference_loading',lambda:InternalTrainPairStudy(
            self.audio,self.video,{n:self.models/n for n in ('E3','A2')},
            self.models/'audio_train',self.models/'video_train',self.sources,self.decoder_artifacts[0]))
        self._models=tuple(x for x in (self.audio.model,self.video.model,
                                       *(self.study.adapters[n]._model for n in ('E3','A2'))))
        self.load_counts={'audio':1,'video':1,'E3':1,'A2':1}
        self._classes=(tuple(self.audio.class_names),tuple(self.video.class_names))
        self._execution_profile=(torch.get_num_threads(),torch.get_num_interop_threads(),torch.backends.mkldnn.enabled)
        self.config=DeploymentConfig(self.models,(),())

    def readiness(self):
        return {**unavailable().to_dict(),'predictions':None}

    def _integrity(self):
        if tuple((self.audio.model,self.video.model,*(self.study.adapters[n]._model for n in ('E3','A2'))))!=self._models:
            raise ContractError('loaded model object replaced')
        if (torch.get_num_threads(),torch.get_num_interop_threads(),torch.backends.mkldnn.enabled)!=self._execution_profile:
            raise ContractError('loaded execution profile changed')
        if (tuple(self.audio.class_names),tuple(self.video.class_names))!=self._classes:
            raise ContractError('class order changed')
        self.audio._card_artifact.verify();self.video._card_artifact.verify()
        self.audio.encoder.require_ready();self.video.encoder.require_ready()
        for artifact in self.decoder_artifacts:artifact.verify()
        for name,adapter in self.study.adapters.items():
            bind=CANDIDATES[name]
            for filename,digest in [('best.pt',bind.checkpoint_sha256),
                                    ('checkpoint_metadata.json',bind.metadata_sha256),
                                    ('frozen_operating_point.json',bind.operating_point_sha256)]:
                Artifact(str(self.models/name/filename),digest).verify()
            validate_live_model(adapter._model,adapter.metadata.model)
            if model_fingerprint(adapter._model)!=bind.model_sha256:
                raise ContractError('frozen fusion state changed')
        for model in self._models:
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ContractError('model is not frozen eval')

    def predict_internal(self, sample_id, content, filename, *, inference_mode=False, include_a2=True):
        if not self._lock.acquire(blocking=False):return error_result('RESOURCE_BUSY')
        started=time.perf_counter();timings=Timings()
        try:
            if sample_id not in self.sources:return error_result('UNAPPROVED_TRAIN_SOURCE')
            if type(inference_mode) is not bool or type(include_a2) is not bool:
                return error_result('INVALID_INTERNAL_CONFIGURATION')
            if not isinstance(content,bytes) or not content:return error_result('UNSUPPORTED_UPLOAD')
            if len(content)>self.config.max_upload_bytes:return error_result('UPLOAD_TOO_LARGE')
            if (not isinstance(filename,str) or Path(filename).name!=filename
                    or any(c in filename for c in ('/','\\','\x00'))):
                return error_result('UNSUPPORTED_FILENAME')
            if Path(filename).suffix.lower()!='.mp4':return error_result('UNSUPPORTED_EXTENSION')
            if filename!=Path(self.sources[sample_id].path).name or hashlib.sha256(content).hexdigest()!=self.sources[sample_id].sha256:
                return error_result('SOURCE_PROVENANCE_MISMATCH')
            return self._run(sample_id,content,filename,timings,started,inference_mode,include_a2)
        except DeploymentError as exc:
            safe={'UNSUPPORTED_UPLOAD','MALFORMED_MEDIA_METADATA','MODALITY_CLOCK_MISMATCH',
                'UNSUPPORTED_VIDEO_DIMENSIONS','UNSUPPORTED_FRAME_RATE','DECODER_DEPENDENCY_MISSING',
                'PROBE_TIMEOUT','PROBE_OUTPUT_TOO_LARGE','PROBE_OUTPUT_READ_FAILED','INVALID_DECODED_AUDIO'}
            return error_result(exc.code if exc.code in safe else 'INTERNAL_VALIDATION_FAILED')
        except Exception:
            return error_result('INTERNAL_INFERENCE_FAILED')
        finally:
            self._lock.release()

    def _run(self,sample_id,content,filename,timings,started,inference_mode,include_a2):
        clocks={}
        def probe(path):
            media,evidence=probe_with_clocks(path,self.ffprobe);clocks.update(evidence)
            return media
        validation=timings.call('media_validation',lambda:validate_upload(
            content,filename=filename,config=self.config,readiness=unavailable(),probe=probe))
        if validation.get('error_code')!='RAW_INFERENCE_GATE_CLOSED':
            return {**validation,'schema':SCHEMA,'production_predictions_enabled':False}
        timings.call('integrity_pre',self._integrity)
        with tempfile.TemporaryDirectory(prefix='aval-internal-request-') as temporary:
            source=Path(temporary)/filename;source.write_bytes(content)
            artifact=Artifact(str(source),self.sources[sample_id].sha256)
            wav=Path(temporary)/'audio.wav'
            def decode_audio():
                code,_,_=_run_bounded_process([str(self.ffmpeg),'-nostdin','-v','error','-y',
                    '-max_alloc',str(MAX_PROBE_ALLOCATION_BYTES),'-i',str(source),'-vn','-ac','1','-ar','16000',
                    '-t','10','-fs','322048',str(wav)])
                if code or not wav.is_file() or wav.stat().st_size>322048:
                    raise DeploymentError('INVALID_DECODED_AUDIO')
                waveform,rate=sf.read(wav,dtype='float32',always_2d=False)
                if (rate!=16000 or waveform.shape!=(160000,) or waveform.dtype!=np.float32
                        or not np.isfinite(waveform).all() or (np.abs(waveform)>1).any()):
                    raise DeploymentError('INVALID_DECODED_AUDIO')
                return waveform
            wave=timings.call('audio_decode',decode_audio)
            frames,video_timing=timings.call('video_decode',lambda:sample_train_frames(artifact))
            audio_input=np.zeros(160000,dtype=np.float32);audio_input[:]=wave
            audio_tensor=torch.from_numpy(audio_input)[None]
            def prepare_video():
                x=(frames.astype(np.float32)/255.).transpose(0,3,1,2)
                x=(x-np.array([.485,.456,.406],np.float32).reshape(1,3,1,1))/np.array([.229,.224,.225],np.float32).reshape(1,3,1,1)
                return torch.from_numpy(x).unsqueeze(0)
            video_tensor=timings.call('video_preprocess',prepare_video)
            context=torch.inference_mode if inference_mode else torch.no_grad
            with context(),torch.autocast('cpu',enabled=False):
                audio_started=time.perf_counter();cpu_started=time.process_time();split=[]
                def after_fbank(module,args):split.append((time.perf_counter(),time.process_time()))
                hook=self.audio.model.beats.patch_embedding.register_forward_pre_hook(after_fbank)
                try:
                    _,audio_emb=self.audio.model(audio_tensor)
                finally:hook.remove()
                audio_finished=time.perf_counter();cpu_finished=time.process_time()
                if len(split)!=1:raise ContractError('audio timing hook coverage differs')
                timings.values['audio_preprocess']={'wall_seconds':split[0][0]-audio_started,'process_cpu_seconds':split[0][1]-cpu_started}
                timings.values['audio_encoder']={'wall_seconds':audio_finished-split[0][0],'process_cpu_seconds':cpu_finished-split[0][1]}
                _,visual_emb=timings.call('video_encoder',lambda:self.video.model(video_tensor))
                def align():
                    for x,encoder in ((audio_emb,self.audio.encoder),(visual_emb,self.video.encoder)):
                        EncoderOutput(x,torch.ones(1,10,dtype=torch.bool),BIN_STARTS,encoder.provenance).validate(1,encoder.provenance)
                    for sec,frame in enumerate(video_timing['frames']):
                        if (frame['source_frame_index']!=int(sec*video_timing['fps']) or not frame['valid']
                                or abs(frame['timestamp_seconds']-frame['source_frame_index']/video_timing['fps'])>1e-5):
                            raise ContractError('achieved video timing differs from producer convention')
                    if abs(video_timing['fps']-clocks['video_fps'])>1e-6:
                        raise ContractError('probe and decode FPS differ')
                    return torch.ones(1,10,dtype=torch.bool)
                mask=timings.call('alignment',align)
                predictions={}
                for name in (('E3','A2') if include_a2 else ('E3',)):
                    logit=timings.call(name,lambda name=name:self.study.adapters[name]._model(audio_emb,visual_emb,mask).logits)
                    if logit.shape!=(1,1) or logit.dtype!=torch.float32 or not torch.isfinite(logit).all():
                        raise ContractError('invalid frozen fusion output')
                    z=float(logit[0,0]);score=float(np.exp(-np.logaddexp(0.,-np.float64(z))))
                    binding=CANDIDATES[name]
                    predictions[name]={'score':score,'logit':z,'decision':'original' if score>=binding.threshold else 'substituted',
                        'predicted_target':int(score>=binding.threshold),'score_semantics':SEMANTICS,
                        'threshold':binding.threshold,'checkpoint_sha256':binding.checkpoint_sha256,'model_sha256':binding.model_sha256}
            artifact.verify()
        timings.call('integrity_post',self._integrity)
        self._completed+=1
        return {'schema':SCHEMA,'request_id':uuid.uuid4().hex,'status':'EXPERIMENTAL_INTERNAL_TRAIN_ONLY',
            'predictions':predictions,'production_predictions_enabled':False,'public_readiness':self.readiness(),
            'event_predictions':None,'temporal_predictions':None,
            'timings':{**timings.values,'total':{'wall_seconds':time.perf_counter()-started}},
            'diagnostics':{'sample_id':sample_id,'completed_request_number':self._completed,
                'source_sha256':artifact.sha256,'canonical_waveform_sha256':hashlib.sha256(wave.tobytes()).hexdigest(),
                'video_timing':video_timing,'clock_metadata':clocks,'phase6_shared_clock_accepted':False,
                'timing_basis':'producer intended seconds with separately recorded achieved PTS',
                'inference_mode':inference_mode,'load_counts':dict(self.load_counts),
                'execution_profile':{'intra_op_threads':self._execution_profile[0],
                    'inter_op_threads':self._execution_profile[1],'mkldnn_enabled':self._execution_profile[2]},
                'encoder_stamps':{'audio':asdict(self.audio.encoder.provenance),'video':asdict(self.video.encoder.provenance)}},
            '_audit_features':(audio_emb.detach().numpy().copy(),visual_emb.detach().numpy().copy())}


def response_for_review(result):
    """JSON-safe response boundary; private audit tensors never enter the UI schema."""
    if type(result) is not dict or any(type(key) is not str for key in result):
        raise ContractError('invalid response object')
    response={key:value for key,value in result.items() if not key.startswith('_')}
    validate_internal_response(response)
    return response


def validate_internal_response(response):
    """Validate the internal ABI without granting any public inference readiness."""
    if (type(response) is not dict or response.get('schema')!=SCHEMA
            or response.get('production_predictions_enabled') is not False):
        raise ContractError('invalid internal response envelope')
    if response.get('status') in ('REJECTED','MODEL_NOT_READY'):
        if (response.get('predictions') is not None or type(response.get('error_code')) is not str
                or not re.fullmatch('[A-Z0-9_]+',response['error_code'])):
            raise ContractError('invalid sanitized error envelope')
        if set(response)-{'schema','status','predictions','error_code','issues','production_predictions_enabled'}:
            raise ContractError('unexpected error response field')
        issues=response.get('issues',[])
        if (type(issues) is not list or any(type(x) is not dict or set(x)!={'category','code'}
                or any(type(v) is not str or not re.fullmatch('[A-Za-z0-9_.:-]{1,96}',v) for v in x.values()) for x in issues)):
            raise ContractError('invalid sanitized readiness issues')
        return
    expected={'schema','request_id','status','predictions','production_predictions_enabled','public_readiness',
              'event_predictions','temporal_predictions','timings','diagnostics'}
    if (set(response)!=expected or response['status']!='EXPERIMENTAL_INTERNAL_TRAIN_ONLY'
            or response['event_predictions'] is not None or response['temporal_predictions'] is not None
            or type(response['request_id']) is not str or not re.fullmatch('[a-f0-9]{32}',response['request_id'])):
        raise ContractError('invalid experimental response contract')
    readiness=response['public_readiness']
    if (type(readiness) is not dict or readiness.get('status')!='MODEL_NOT_READY' or readiness.get('inference_enabled') is not False
            or readiness.get('predictions') is not None):
        raise ContractError('experimental result cannot open public readiness')
    if type(response['predictions']) is not dict or set(response['predictions']) not in ({'E3'},{'E3','A2'}):
        raise ContractError('mandatory frozen E3 result missing')
    for name,prediction in response['predictions'].items():
        binding=CANDIDATES[name]
        required={'score','logit','decision','predicted_target','score_semantics','threshold','checkpoint_sha256','model_sha256'}
        if (type(prediction) is not dict or set(prediction)!=required
                or prediction['threshold']!=binding.threshold or prediction['checkpoint_sha256']!=binding.checkpoint_sha256
                or prediction['model_sha256']!=binding.model_sha256 or prediction['score_semantics']!=SEMANTICS):
            raise ContractError('frozen result binding differs')
        if any(type(prediction[k]) not in (int,float) or not math.isfinite(prediction[k]) for k in ('score','logit')):
            raise ContractError('nonfinite correspondence result')
        score=prediction['score'];target=int(score>=binding.threshold)
        expected_score=float(np.exp(-np.logaddexp(0.,-np.float64(prediction['logit']))))
        if (not 0<=score<=1 or type(prediction['predicted_target']) is not int or prediction['predicted_target']!=target
                or prediction['decision']!=('original' if target else 'substituted') or score!=expected_score):
            raise ContractError('frozen decision rule differs')
    if type(response['diagnostics']) is not dict or response['diagnostics'].get('phase6_shared_clock_accepted') is not False:
        raise ContractError('unreviewed timing cannot be promoted')
    if type(response['timings']) is not dict or 'total' not in response['timings']:
        raise ContractError('missing timing diagnostic')
    for values in response['timings'].values():
        if type(values) is not dict or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values()):
            raise ContractError('invalid timing diagnostic')
