"""Opt-in sensitivity audit reusing preserved genuine TRAIN float32 features.

First require every one of the ten logits to equal the fresh compressed-media
audit. Retain exact feature digests; no tolerance or approval is created.
"""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from deployment.general_decode import require_research_sandbox
from deployment.general_validation import fusion_sensitivity
from deployment.train_manifest import PERMITTED_IDS, load_manifest
from fusion.inference import FrozenFusionInference


def run():
    require_research_sandbox()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    models=Path('/models');inputs=Path('/inputs');root=Path('/reference')
    manifest_sha='65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e'
    load_manifest(inputs/'manifest.json',manifest_sha,models,inputs)
    service=SimpleNamespace(fusion=FrozenFusionInference(models/'E3',candidate='E3'))
    raw={};historical={};digests={}
    for role in ('audio','video'):
        paths=[root/f'first_pass_{sid}_{role}.npy' for sid in PERMITTED_IDS]
        for p in paths:digests[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        raw[role]=[np.load(p,allow_pickle=False)[0] for p in paths]
        ref=models/(role+'_train')
        with (ref/'train_ids.csv').open() as stream:ids=[r['sample_id'] for r in csv.DictReader(stream)]
        arr=np.load(ref/'train_emb.npy',allow_pickle=False)
        historical[role]=[arr[ids.index(sid)] for sid in PERMITTED_IDS]
    rows=fusion_sensitivity(service,raw['audio'],raw['video'],historical['audio'],historical['video'])
    audit_path=Path('/results/validation.json');audit=json.loads(audit_path.read_text())
    assert len(rows)==10
    for row,case in zip(rows,audit['cases'][:10]):
        assert row['variants']['raw_float32']['logit']==case['logit']
        assert row['variants']['raw_float32']['score']==case['score']
    for name,digest in digests.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    report={'scope':'five prespecified TRAIN original/cyclic-swap pairs only; not an accuracy benchmark',
        'approved':False,'numerical_policy':'inactive; measurements only',
        'reference_feature_sha256':digests,'fresh_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'raw_logits_match_fresh_general_audit_exactly':True,'cases':rows}
    Path('/results/sensitivity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':run()
