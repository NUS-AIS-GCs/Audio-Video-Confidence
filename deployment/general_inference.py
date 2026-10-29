"""Authentic general-media research service. Never authorizes public inference."""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
from threading import Lock
import uuid

import numpy as np
import torch

from deployment.auth import login_owner
from deployment.audio_bundle import VerifiedAudioPredictor
from deployment.video_bundle import VerifiedVideoPredictor
from deployment.general_decode import GeneralDecoder, PROFILE, require_research_sandbox
from deployment.provision_artifacts import artifact_catalog, verify_artifacts
from deployment.release_gate import unapproved_prediction
from deployment.runtime import DeploymentConfig, DeploymentError, PINNED_VERSIONS
from deployment.train_manifest import AUDIO_CARD, VIDEO_CARD
from fusion.encoder_interfaces import Artifact
from fusion.inference import CANDIDATES, FrozenFusionInference
from fusion.mlp_training import model_fingerprint
from fusion.model_interface import validate_live_model

SCHEMA = 'aval.general-media-research.v1'
STATUS = 'EXPERIMENTAL_GENERAL_MEDIA_NOT_APPROVED'


@dataclass(frozen=True)
class ResearchPrediction:
    request_id: str
    source_sha256: str
    profile_sha256: str
    score: float
    logit: float
    processing_seconds: float
    threshold: float = CANDIDATES['E3'].threshold
    model_sha256: str = CANDIDATES['E3'].model_sha256
    checkpoint_sha256: str = CANDIDATES['E3'].checkpoint_sha256
    schema: str = SCHEMA
    status: str = STATUS
    production_predictions_enabled: bool = False

    def __post_init__(self):
        if (not re.fullmatch('[a-f0-9]{32}', self.request_id)
                or any(not re.fullmatch('[a-f0-9]{64}', x) for x in (self.source_sha256,self.profile_sha256))
                or self.threshold != CANDIDATES['E3'].threshold or self.model_sha256 != CANDIDATES['E3'].model_sha256
                or self.checkpoint_sha256 != CANDIDATES['E3'].checkpoint_sha256
                or self.schema != SCHEMA or self.status != STATUS or self.production_predictions_enabled is not False):
            raise ValueError('invalid research identity or unauthorized scope')
        if (any(type(x) is not float or not math.isfinite(x) for x in (self.score,self.logit,self.processing_seconds))
                or not 0 <= self.score <= 1 or self.processing_seconds < 0
                or self.score != float(np.exp(-np.logaddexp(0.,-np.float64(self.logit))))):
            raise ValueError('invalid genuine correspondence result')

    def to_dict(self):
        return {**asdict(self), 'classification':'Original-like pairing' if self.score>=self.threshold else 'Audio-substituted-like pairing',
            'model_version':'E3 / frozen concatenation + MLP', 'score_semantics':'uncalibrated sigmoid correspondence score',
            'implementation_status':'IMPLEMENTED', 'numerical_temporal_release_validation':'BLOCKED',
            'independent_approval':None, 'event_predictions':None, 'temporal_predictions':None,
            'limitation':'Pairing correspondence does not prove physical sound-source causation.'}


def error(code):
    return {'schema':SCHEMA,'status':'REJECTED','predictions':None,'error_code':code,
            'production_predictions_enabled':False}


def response_for_research(result):
    """Only a validated success ABI or fixed sanitized error enters the research UI."""
    if result.get('status') == STATUS:
        fields = ResearchPrediction.__dataclass_fields__
        prediction = ResearchPrediction(**{k:result[k] for k in fields})
        expected = prediction.to_dict()
        if {k:v for k,v in result.items() if k not in ('diagnostics','_audit')} != expected:
            raise ValueError('unexpected research response fields')
        return {**expected, 'diagnostics':result['diagnostics']}
    if (set(result) != set(error('ERROR')) or result.get('schema') != SCHEMA or result.get('status') != 'REJECTED'
            or result.get('predictions') is not None or result.get('production_predictions_enabled') is not False
            or not re.fullmatch('[A-Z0-9_]{1,96}',result.get('error_code',''))):
        raise ValueError('invalid sanitized research error')
    return dict(result)


class GeneralResearchService:
    """One authentic worker, one nonblocking request slot; no TRAIN identity fabrication."""
    def __init__(self, models, derived):
        require_research_sandbox()
        self.models = Path(models).resolve()
        if not verify_artifacts(self.models)['ready']:
            raise DeploymentError('GENERAL_MODEL_ARTIFACTS_NOT_READY')
        self.config = DeploymentConfig(self.models, (), ())
        self.decoder = GeneralDecoder(self.config)
        self._catalog = artifact_catalog()
        self.audio = VerifiedAudioPredictor(Artifact(str(self.models/'audio/model_card.json'),AUDIO_CARD),Path(derived))
        self.video = VerifiedVideoPredictor(Artifact(str(self.models/'video/model_card.json'),VIDEO_CARD))
        self.fusion = FrozenFusionInference(self.models/'E3', candidate='E3')
        self._models = (self.audio.model,self.video.model,self.fusion._model)
        self._classes = (tuple(self.audio.class_names),tuple(self.video.class_names))
        if self._classes[0] != self._classes[1]:
            raise DeploymentError('GENERAL_CLASS_ORDER_MISMATCH')
        self._execution = (torch.get_num_threads(),torch.get_num_interop_threads(),torch.backends.mkldnn.enabled)
        self.load_counts = {'audio':1,'video':1,'E3':1}
        self._lock = Lock()
        self.upload_store = None
        self.profile = {'id':PROFILE, 'artifacts':self._catalog,
            'runtime':{n:importlib.metadata.version(n) for n in PINNED_VERSIONS},
            'decoder_binaries':{n:a.sha256 for n,a in self.decoder.executables.items()},
            'source_hashes':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ('general_decode.py','general_inference.py')},
            'sandbox':self.decoder.sandbox, 'torch_execution':self._execution,
            'audio':'FFmpeg mono 16k PCM_s16le -t 10; float32 /32768; inherited authentic BEATs preprocessing',
            'video':'CFR int(second*fps); FFmpeg BGR24; OpenCV BGR->RGB and INTER_LINEAR 224x224; inherited ImageNet normalization',
            'public_authorization':unapproved_prediction('upload')}
        self.profile_sha256 = hashlib.sha256(json.dumps(self.profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self._integrity()

    def _integrity(self):
        if hashlib.sha256(json.dumps(self.profile,sort_keys=True,separators=(',',':')).encode()).hexdigest() != self.profile_sha256:
            raise DeploymentError('GENERAL_RUNTIME_PROFILE_CHANGED')
        for name,digest in self.profile['source_hashes'].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
                raise DeploymentError('GENERAL_RUNTIME_PROFILE_CHANGED')
        if ((self.audio.model,self.video.model,self.fusion._model) != self._models
                or (tuple(self.audio.class_names),tuple(self.video.class_names)) != self._classes
                or (torch.get_num_threads(),torch.get_num_interop_threads(),torch.backends.mkldnn.enabled) != self._execution):
            raise DeploymentError('GENERAL_LOADED_STATE_CHANGED')
        for name,binding in self._catalog.items():
            Artifact(str(self.models/name),binding['sha256']).verify()
        self.audio.encoder.require_ready(); self.video.encoder.require_ready()
        validate_live_model(self.fusion._model,self.fusion.metadata.model)
        if model_fingerprint(self.fusion._model) != CANDIDATES['E3'].model_sha256:
            raise DeploymentError('GENERAL_FROZEN_MODEL_CHANGED')
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in self._models):
            raise DeploymentError('GENERAL_FROZEN_MODEL_CHANGED')

    def predict_research(self, content, filename):
        """Research harness only. Uploaded bytes, never a client path or TRAIN export claim."""
        if not self._lock.acquire(blocking=False):
            return error('RESOURCE_BUSY')
        start = time.perf_counter()
        try:
            self._integrity()
            decoded = self.decoder.decode(content,filename)
            score, logit, audio, visual = self._forward_window(decoded.waveform, decoded.frames)
            self._integrity()
            result = ResearchPrediction(uuid.uuid4().hex,decoded.source_sha256,self.profile_sha256,
                score,logit,float(time.perf_counter()-start)).to_dict()
            result['diagnostics'] = {'timing':decoded.timing,'waveform_sha256':decoded.waveform_sha256,
                'frames_sha256':decoded.frames_sha256,'audio_embedding_shape':list(audio.embeddings.shape),
                'video_embedding_shape':list(visual.embeddings.shape),'audio_probabilities_shape':list(audio.probs.shape),
                'video_probabilities_shape':list(visual.probs.shape),'load_counts':dict(self.load_counts)}
            result['_audit'] = (decoded.waveform,decoded.frames,audio.embeddings,visual.embeddings)
            response_for_research(result)
            return result
        except DeploymentError as exc:
            # DeploymentError codes are fixed program identifiers; never decoder stderr.
            return error(exc.code if re.fullmatch('[A-Z0-9_]{1,96}',exc.code) else 'GENERAL_INFERENCE_FAILED')
        except Exception:
            return error('GENERAL_INFERENCE_FAILED')
        finally:
            self._lock.release()

    def _forward_window(self, waveform, frames):
        """Shared authentic forward; caller owns the worker lock and integrity checks.

        This grants no admission, ownership, timeline or public-release authority.
        Both offline live replay and compressed-media research use these same
        already-loaded verified encoders and the frozen E3 model.
        """
        audio = self.audio.predict_waveform(waveform)
        visual = self.video.predict_frames(frames)
        with torch.no_grad(), torch.autocast('cpu', enabled=False):
            z = self.fusion._model(torch.from_numpy(audio.embeddings)[None],
                torch.from_numpy(visual.embeddings)[None], torch.ones(1,10,dtype=torch.bool)).logits
        if z.shape != (1,1) or z.dtype != torch.float32 or not torch.isfinite(z).all():
            raise DeploymentError('GENERAL_INVALID_FUSION_OUTPUT')
        logit = float(z[0,0])
        score = float(np.exp(-np.logaddexp(0.,-np.float64(logit))))
        return score, logit, audio, visual

    def submit_owned(self, handle, request):
        from deployment.uploads import UploadError
        raw = getattr(request,'request',None)
        owner = login_owner(raw)
        if owner is None or self.upload_store is None or raw.scope.get('aval.upload_guard') is not self.upload_store:
            return error('UPLOAD_AUTHENTICATION_REQUIRED')
        if not handle:
            return error('UPLOAD_REQUIRED')
        try:
            with self.upload_store.lease(handle,owner) as (path,filename):
                with path.open('rb') as stream:
                    content = stream.read(self.config.max_upload_bytes+1)
                result = self.predict_research(content,filename)
                if login_owner(raw) != owner:
                    return error('UPLOAD_AUTHENTICATION_REQUIRED')
                return response_for_research(result)
        except UploadError:
            return error('UPLOAD_NOT_FOUND')
        except Exception:
            return error('GENERAL_INFERENCE_FAILED')
