"""Authenticate the permitted offline TRAIN bundle without importing a profiler."""
import hashlib
import json
from pathlib import Path

from fusion.contracts import ContractError

AUDIO_CARD = '3b98967d720e7fad3de3af37f57487ebd49cf48dc357cef8dbde394a86470bfc'
VIDEO_CARD = '12f069feab94d3324f80cf9de08390a5f1a6fdaa7acdb293ada1690262925a1b'
PERMITTED_IDS = ('c---zaDCTaE__08', '22DapbNwnUY__12', '0h8HxZaX9wQ__26',
                 '6VRcINAY7KE__07', 'NGeniwxDzJ0__21')
REQUIRED_MODEL_FILES = frozenset(
    ['audio/'+n for n in ('model_card.json','encoder.pt','weights.pt','norm_stats.json','class_names.json','environment.json')]
    + ['video/'+n for n in ('model_card.json','weights.pt','class_names.json','environment.json','README.md',
                           'parity_samples/README.md','parity_samples/emb.npy','parity_samples/probs.npy','parity_samples/samples.csv')]
    + ['video/parity_samples/'+s.rsplit('__',1)[0]+'.mp4' for s in PERMITTED_IDS]
    + [role+'/'+n for role in ('audio_train','video_train') for n in ('SOURCE.txt','README.md','class_names.json','train_emb.npy','train_probs.npy','train_ids.csv')]
    + [role+'/'+n for role in ('E3','A2') for n in ('best.pt','checkpoint_metadata.json','checkpoint_integrity.json','frozen_operating_point.json')])
REQUIRED_INPUT_FILES = frozenset({'waveforms.npy','frames.npy','audio_stage1.npy','audio_stage2b.npy','video_stage2b.npy'})


def verify_file(path, expected_hash):
    if (type(expected_hash) is not str or len(expected_hash)!=64
            or any(c not in '0123456789abcdef' for c in expected_hash)):
        raise ContractError('explicit lowercase SHA-256 required')
    try:
        with Path(path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    except OSError:
        raise ContractError('required audit artifact unavailable') from None
    if actual!=expected_hash:raise ContractError('audit artifact SHA-256 mismatch')


def pinned_child(root, relative):
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ContractError('relative audit path required')
    path=(Path(root)/relative).resolve()
    try:path.relative_to(Path(root).resolve())
    except ValueError:raise ContractError('audit path escapes mounted root') from None
    return path


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ContractError('duplicate audit manifest key')
        result[key]=value
    return result


def load_manifest(path, expected_hash, models, inputs):
    verify_file(path,expected_hash)
    try:meta=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_unique_object)
    except (OSError,ValueError,TypeError):raise ContractError('malformed audit manifest') from None
    if (type(meta) is not dict or meta.get('schema')!='aval.stage2c-permitted-train.v1'
            or type(meta.get('sample_ids')) is not list or tuple(meta['sample_ids'])!=PERMITTED_IDS
            or meta.get('audio_card_sha256')!=AUDIO_CARD or meta.get('video_card_sha256')!=VIDEO_CARD
            or type(meta.get('model_files')) is not dict or set(meta['model_files'])!=REQUIRED_MODEL_FILES
            or type(meta.get('input_files')) is not dict or set(meta['input_files'])!=REQUIRED_INPUT_FILES):
        raise ContractError('unsupported acquisition/identity bindings')
    for root,key in ((models,'model_files'),(inputs,'input_files')):
        for relative,digest in meta[key].items():verify_file(pinned_child(root,relative),digest)
    verify_file(path,expected_hash)
    return meta
