"""Bounded encoder evidence from the existing forward; no fused presence head.

The 29-class vocabulary is copied byte-for-byte from the verified audio bundle.
Video uses different JSON formatting, with the same verified class order.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from deployment.provision_artifacts import artifact_catalog
from deployment.runtime import DeploymentError

SCHEMA = 'aval.encoder-event-evidence.v1'
SEMANTICS = 'Per-second 29-class softmax scores; uncalibrated; background included.'
TEMPORAL_UNAVAILABLE = 'Temporal presence head — Not available (verified artifact not loaded).'


def class_names():
    data = Path(__file__).with_name('event_class_names.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact_catalog()['audio/class_names.json']['sha256']:
        raise ValueError('event vocabulary changed')
    names = json.loads(data)
    if (type(names) is not list or len(names) != 29 or names[-1] != 'background'
            or names[:-1] != sorted(names[:-1]) or len(set(names)) != 29):
        raise ValueError('invalid event vocabulary')
    return names


def provenance(modality):
    catalog = artifact_catalog()
    prefix = 'audio' if modality == 'audio' else 'video'
    names = ['weights.pt', 'model_card.json', 'class_names.json']
    if modality == 'audio':
        names += ['encoder.pt', 'norm_stats.json']
    return {'model_version': 'BEATs iter3 + LoRA / beats_lora_r8_v1' if modality == 'audio'
            else 'ResNet-18 / resnet18_v1',
            'artifact_sha256': {name: catalog[prefix+'/'+name]['sha256'] for name in names}}


def _scores(value):
    if (type(value) is not list or len(value) != 10
            or any(type(row) is not list or len(row) != 29 for row in value)
            or any(type(x) is not float for row in value for x in row)):
        raise ValueError('expected 10 by 29 float32 scores')
    data = np.asarray(value, dtype=np.float64)
    if (not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1)
            or not np.array_equal(data, data.astype(np.float32).astype(np.float64))
            # Softmax rounding bound only; this is not a release/parity tolerance.
            or np.any(np.abs(data.sum(axis=1)-1) > 29*np.finfo(np.float32).eps)):
        raise ValueError('invalid encoder softmax values')
    return data


def validate_event_evidence(evidence):
    if evidence is None:
        return
    if (type(evidence) is not dict or set(evidence) != {'schema','class_names','semantics','bin_starts','audio','video'}
            or evidence['schema'] != SCHEMA or evidence['class_names'] != class_names()
            or evidence['semantics'] != SEMANTICS
            or evidence['bin_starts'] != [float(i) for i in range(10)]):
        raise ValueError('invalid event evidence identity')
    for modality in ('audio','video'):
        row = evidence[modality]
        if (type(row) is not dict or set(row) != {'shape','dtype','scores','provenance'}
                or row['shape'] != [10,29] or row['dtype'] != 'float32'
                or row['provenance'] != provenance(modality)):
            raise ValueError('invalid event model binding')
        _scores(row['scores'])


def event_evidence(audio, visual):
    """Serialize genuine predictor outputs, never rerun or alter a model."""
    names = class_names()
    result = {'schema':SCHEMA, 'class_names':names, 'semantics':SEMANTICS,
              'bin_starts':[float(i) for i in range(10)]}
    try:
        for modality, prediction in (('audio',audio),('video',visual)):
            scores = prediction.probs
            if (list(prediction.class_names) != names or type(scores) is not np.ndarray
                    or scores.shape != (10,29) or scores.dtype != np.float32):
                raise ValueError('encoder shape or class order mismatch')
            result[modality] = {'shape':[10,29], 'dtype':'float32',
                'scores':scores.tolist(), 'provenance':provenance(modality)}
        validate_event_evidence(result)
    except (ValueError, TypeError, AttributeError):
        raise DeploymentError('GENERAL_INVALID_EVENT_EVIDENCE') from None
    return result


def summary(evidence, modality):
    """Mean original softmax values; exclude background only from event ranking."""
    scores = _scores(evidence[modality]['scores'])
    mean = scores.mean(axis=0)
    order = sorted(range(28), key=lambda i: (-mean[i], i))[:5]
    return [(evidence['class_names'][i], float(mean[i])) for i in order], float(mean[28])
