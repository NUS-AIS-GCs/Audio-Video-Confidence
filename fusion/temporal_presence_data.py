"""Original-pair AVE temporal supervision, with source-disjoint partitions.

Swaps have no joint temporal target. Background means absence of an annotated
AVE joint event, not an exhaustive annotation of every audible/visible object.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from fusion.artifact_integration import _official_records
from fusion.contracts import OFFICIAL_COUNTS, SPLITS, _read_records
from fusion.verify_ave_reference import verify_reference
from deployment.provision_artifacts import artifact_catalog
from fusion.encoder_interfaces import Artifact


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def source_exclusions(splits):
    membership=defaultdict(set)
    for split,records in splits.items():
        for row in records:membership[row.clip_id].add(split)
    return {clip for clip,partitions in membership.items() if len(partitions)>1}


def temporal_targets(records, excluded=()):
    groups=defaultdict(list)
    for row in records:
        if row.clip_id not in excluded:groups[row.clip_id].append(row)
    result={}
    for clip,rows in groups.items():
        if any(row.start==row.end for row in rows):
            continue  # known empty/ambiguous interval is not a fabricated negative
        result[clip]=np.asarray([any(row.start<=t<row.end for row in rows) for t in range(10)],dtype=np.float32)
    return result


def swapped_temporal_target(*_):
    return None  # independent source annotations cannot label a recombined pair


class PresenceData:
    def __init__(self,audio_root,video_root,metadata_root,reference_root,e3_metadata):
        self.roots={'audio':Path(audio_root),'vision':Path(video_root)}
        self.reference=verify_reference(metadata_root,reference_root)
        self.names,self.records=_official_records(Path(metadata_root),OFFICIAL_COUNTS)
        self.excluded=source_exclusions(self.records)
        expected=artifact_catalog()['E3/checkpoint_metadata.json']['sha256']
        Artifact(str(Path(e3_metadata).resolve()),expected).verify()
        metadata=json.loads(Path(e3_metadata).read_text())
        if metadata['export_identities']!=[['audio','beats_lora_r8_v1 / beats_iter3'],['vision','resnet18_v1']]:
            raise ValueError('frozen encoder export identities differ')
        self.bindings=dict(metadata['data'][0]['input_export_hashes'])
        self.loaded={};self.inputs={}

    def load(self,split,*,test_selection=None):
        if split not in SPLITS:raise ValueError('unknown split')
        if split=='test' and not test_selection:raise ValueError('TEST requires a frozen selection digest')
        if split in self.loaded:return self.loaded[split]
        arrays={};indices={}
        for modality,root in self.roots.items():
            for name in ('class_names.json','SOURCE.txt','README.md',f'{split}_ids.csv',f'{split}_emb.npy'):
                role=modality+'/'+name;actual=sha256(root/name)
                if actual!=self.bindings[role]:raise ValueError('cache differs from authenticated frozen E3 export binding: '+role)
                self.inputs[role]=actual
            if tuple(json.loads((root/'class_names.json').read_text()))!=self.names:raise ValueError('class order mismatch')
            rows=_read_records(root/f'{split}_ids.csv')
            if (len(rows)!=len(self.records[split]) or len({r.sample_id for r in rows})!=len(rows)
                    or set(rows)!=set(self.records[split])):raise ValueError('official/cache annotation mismatch')
            a=np.load(root/f'{split}_emb.npy',mmap_mode='r',allow_pickle=False)
            dim=256 if modality=='audio' else 512
            if a.shape!=(len(rows),10,dim) or a.dtype!=np.float16 or not np.isfinite(a).all():
                raise ValueError('invalid frozen embedding tensor')
            arrays[modality]=a;indices[modality]={r.sample_id:i for i,r in enumerate(rows)}
        targets=temporal_targets(self.records[split],self.excluded)
        groups=defaultdict(list)
        for row in self.records[split]:groups[row.clip_id].append(row)
        clips=sorted(targets)
        # Choose one deterministic original-source representation, independent
        # of labels/predictions. Union targets avoid conflicting duplicate rows.
        representatives=[min(groups[c],key=lambda r:r.sample_id).sample_id for c in clips]
        audio=np.asarray(arrays['audio'][[indices['audio'][sid] for sid in representatives]],dtype=np.float32)
        visual=np.asarray(arrays['vision'][[indices['vision'][sid] for sid in representatives]],dtype=np.float32)
        x=np.concatenate((audio,visual),axis=-1);y=np.stack([targets[c] for c in clips])
        for a in arrays.values():a._mmap.close()
        self.loaded[split]=(x,y,clips)
        return self.loaded[split]

    def verify_inputs_unchanged(self):
        for role,expected in self.inputs.items():
            modality,name=role.split('/',1)
            if sha256(self.roots[modality]/name)!=expected:raise ValueError('training/evaluation input changed')
