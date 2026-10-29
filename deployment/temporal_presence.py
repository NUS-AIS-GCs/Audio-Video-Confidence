"""Optional private, independently versioned Head 2. No public authorization."""
import argparse
from decimal import Decimal,localcontext,ROUND_HALF_EVEN,Underflow
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from deployment.provision_artifacts import artifact_catalog,_copy_bounded
from deployment.runtime import DeploymentError
from fusion.encoder_interfaces import Artifact,_strict_state
from fusion.mlp_training import model_fingerprint
from fusion.temporal_presence_model import VERSION,TemporalPresenceMLP

CATALOG_SHA256='91ef1b5211e7dd50fb7e0353b7584339d32875d28a86f5e0953dc9f7e06f87d1'
SCHEMA='aval.temporal-presence-research.v1'
STATUS='EXPERIMENTAL_AVE_PRESENCE_NOT_APPROVED'


def canonical_sigmoid(logit):
    """Deterministic Head-2-only serialization across Linux/Windows libm.

    A float64 NumPy exp differed by one ULP for an authentic logit across the
    worker/broker OS boundary. Compute the same correctly rounded mathematical
    sigmoid on both sides instead of relaxing result validation or thresholds.
    """
    if type(logit) is not float or not math.isfinite(logit):raise ValueError('finite Head 2 logit required')
    with localcontext() as context:
        context.prec=50;context.rounding=ROUND_HALF_EVEN
        context.Emin=-999999;context.Emax=999999;context.traps[Underflow]=False
        x=Decimal.from_float(logit);z=(-abs(x)).exp()
        return float(Decimal(1)/(Decimal(1)+z) if x>=0 else z/(Decimal(1)+z))


def catalog():
    raw=Path(__file__).with_name('temporal_presence_catalog.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CATALOG_SHA256:raise ValueError('Head 2 catalog changed')
    value=json.loads(raw)
    if value['version']!=VERSION or set(value['files'])!={'weights.pt','metadata.json'}:raise ValueError('invalid Head 2 identity')
    base=artifact_catalog()
    if any(base[name]['sha256']!=digest for name,digest in value['encoder_artifacts'].items()):
        raise ValueError('Head 2 frozen encoder binding changed')
    return value


def verify(root):
    root=Path(root);binding=catalog()
    if root.is_symlink() or not root.is_dir():raise ValueError('private Head 2 directory required')
    for name,row in binding['files'].items():
        path=root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes']:raise ValueError('Head 2 artifact missing')
        Artifact(str(path.resolve()),row['sha256']).verify()
    metadata=json.loads((root/'metadata.json').read_text())
    if (metadata['schema']!='gc2-head2-metadata-v1' or metadata['version']!=VERSION
            or metadata['integration_criteria_passed'] is not True or metadata['public_release_approved'] is not False
            or metadata['checkpoint_sha256']!=binding['files']['weights.pt']['sha256']
            or metadata['selection_sha256']!=binding['selection_sha256']
            or any(metadata[k]!=binding[k] for k in ('hidden','kernel','context','dropout','threshold','scope','limitations'))):
        raise ValueError('Head 2 evaluation or model binding invalid')
    return metadata


def provision(source,destination):
    source,destination=Path(source).resolve(),Path(destination).absolute()
    if destination.is_symlink():raise ValueError('private Head 2 destination required')
    destination=destination.resolve()
    if source==destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('separate Head 2 source and destination required')
    verify(source)
    if destination.exists():verify(destination);return
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.head2-',dir=destination.parent) as directory:
        stage=Path(directory)/'Head2';stage.mkdir(mode=0o700)
        for name,binding in catalog()['files'].items():_copy_bounded(source/name,stage/name,binding)
        verify(stage);stage.rename(destination)


def validate_prediction(value):
    if value is None:return
    binding=catalog()
    expected={'schema':SCHEMA,'status':STATUS,'model_version':VERSION,'threshold':binding['threshold'],
        'checkpoint_sha256':binding['files']['weights.pt']['sha256'],
        'metadata_sha256':binding['files']['metadata.json']['sha256'],
        'scope':binding['scope'],'limitations':binding['limitations'],'bin_starts':[float(i) for i in range(10)],
        'score_semantics':'uncalibrated trained binary sigmoid; not a calibrated probability',
        'production_predictions_enabled':False}
    if type(value) is not dict or set(value)!=set(expected)|{'scores','logits','presence'}:
        raise ValueError('invalid Head 2 result fields')
    if any(value[k]!=v for k,v in expected.items()):raise ValueError('invalid Head 2 result identity')
    for key in ('scores','logits','presence'):
        if type(value[key]) is not list or len(value[key])!=10:raise ValueError('ten Head 2 bins required')
    for score,logit,present in zip(value['scores'],value['logits'],value['presence']):
        if (type(score) is not float or type(logit) is not float or type(present) is not bool
                or not math.isfinite(score) or not math.isfinite(logit) or not 0<=score<=1
                or score!=canonical_sigmoid(logit)
                or present!=(score>=binding['threshold'])):raise ValueError('invalid trained Head 2 values')


class TemporalPresencePredictor:
    def __init__(self,root):
        self.root=Path(root);self.metadata=verify(root);binding=catalog()
        cfg={k:binding[k] for k in ('hidden','kernel','dropout','context')}
        payload=torch.load(self.root/'weights.pt',map_location='cpu',weights_only=True)
        if (type(payload) is not dict or set(payload)!={'schema','hidden','kernel','dropout','context','state_dict'}
                or payload['schema']!='gc2-head2-state-v1' or any(payload[k]!=v for k,v in cfg.items())):
            raise ValueError('invalid Head 2 checkpoint architecture')
        self.model=TemporalPresenceMLP(**cfg);_strict_state(self.model,payload['state_dict'])
        self.model.eval().requires_grad_(False)
        if not torch.all(self.model.scale>=1e-6):raise ValueError('invalid TRAIN normalization')
        self._fingerprint=model_fingerprint(self.model)
        self._structure=repr(self.model);self._config=cfg
        self._sources={Path(__file__):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        import fusion.temporal_presence_model as implementation
        path=Path(implementation.__file__);self._sources[path]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.require_ready()

    def require_ready(self):
        if (verify(self.root)!=self.metadata or model_fingerprint(self.model)!=self._fingerprint
                or repr(self.model)!=self._structure or self.model.training
                or any(p.requires_grad for p in self.model.parameters())
                or any(getattr(self.model,k)!=v for k,v in self._config.items())
                or any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in self._sources.items())):
            raise ValueError('frozen Head 2 state changed')

    def predict(self,audio,video):
        self.require_ready()
        for data,dim in ((audio,256),(video,512)):
            if type(data) is not np.ndarray or data.shape!=(10,dim) or data.dtype!=np.float32 or not np.isfinite(data).all():
                raise DeploymentError('TEMPORAL_PRESENCE_INPUT_INVALID')
        with torch.no_grad(),torch.autocast('cpu',enabled=False):
            logits=self.model(torch.from_numpy(np.concatenate((audio,video),axis=-1))[None])
        if logits.shape!=(1,10) or logits.dtype!=torch.float32 or not torch.isfinite(logits).all():
            raise DeploymentError('TEMPORAL_PRESENCE_OUTPUT_INVALID')
        raw=logits[0].tolist();scores=[canonical_sigmoid(value) for value in raw]
        binding=catalog()
        result={'schema':SCHEMA,'status':STATUS,'model_version':VERSION,'threshold':binding['threshold'],
            'checkpoint_sha256':binding['files']['weights.pt']['sha256'],'metadata_sha256':binding['files']['metadata.json']['sha256'],
            'scope':binding['scope'],'limitations':binding['limitations'],'bin_starts':[float(i) for i in range(10)],
            'score_semantics':'uncalibrated trained binary sigmoid; not a calibrated probability',
            'production_predictions_enabled':False,'scores':scores,'logits':raw,
            'presence':[score>=binding['threshold'] for score in scores]}
        validate_prediction(result);self.require_ready();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    try:provision(args.source,args.destination)
    except Exception:raise SystemExit('PRIVATE_HEAD2_PROVISIONING_FAILED') from None
    print('PRIVATE_HEAD2_ARTIFACTS_VERIFIED; public gates unchanged')


if __name__=='__main__':main()
