"""Numerical audit math and manifest rejection; no replacement encoder weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.parity_metrics import numerical_metrics, float16_cell_analysis
from deployment.stage2c_probe import (
    AUDIO_CARD, VIDEO_CARD, PERMITTED_IDS, REQUIRED_MODEL_FILES, REQUIRED_INPUT_FILES,
    load_manifest, pinned_child, normalize_video,
)
from fusion.contracts import ContractError


def test_metrics_use_float64_subtraction_without_changing_precision():
    reference=np.array([1.,-2.],np.float16)
    actual=np.array([1.25,-1.5],np.float32)
    m=numerical_metrics(actual,reference)
    assert m['max_absolute_error']==.5
    assert m['mae']==.375
    assert m['rmse']==pytest.approx(np.sqrt(.15625))
    assert m['actual_dtype']=='float32' and m['reference_dtype']=='float16'
    assert m['float16_differing_values']==2


@pytest.mark.parametrize('actual,reference',[
    (np.array([],np.float32),np.array([],np.float16)),
    (np.array([np.nan],np.float32),np.array([0],np.float16)),
    (np.array([np.inf],np.float32),np.array([0],np.float16)),
    (np.array([0],np.float32),np.array([np.inf],np.float16)),
    (np.zeros(2,np.float32),np.zeros(1,np.float16)),
    (np.zeros(1,np.int64),np.zeros(1,np.float16)),
    (np.array([1e10],np.float32),np.array([0],np.float16)),
])
def test_invalid_metrics_rejected(actual,reference):
    with pytest.raises(ValueError):numerical_metrics(actual,reference)


def test_rounding_cell_is_asymmetric_at_power_of_two():
    r=np.array([1.],np.float16)
    a=np.array([1.0001],np.float32)
    m=float16_cell_analysis(a,r)
    assert m['max_rounding_cell_excess']==0
    assert m['float16_array_equal'] is True
    down=np.nextafter(r,np.float16(-np.inf)).astype(np.float64)
    up=np.nextafter(r,np.float16(np.inf)).astype(np.float64)
    assert (1.-down)[0] * 2 == (up-1.)[0]


def test_tie_to_even_is_separate_from_closed_cell_excess():
    r=np.array([np.nextafter(np.float16(1),np.float16(2))],np.float16)
    nxt=np.nextafter(r,np.float16(2)).astype(np.float64)
    midpoint=((r.astype(np.float64)+nxt)/2).astype(np.float32)
    m=float16_cell_analysis(midpoint,r)
    assert m['max_rounding_cell_excess']==0
    assert m['float16_array_equal'] is False
    assert m['mismatch_coordinates']==[[0]]


def test_mismatch_locations_and_temporal_distributions_are_exact():
    r=np.ones((2,10,3),np.float16)
    a=r.astype(np.float32)
    a[1,7,2]=1.01
    m=float16_cell_analysis(a,r)
    assert m['mismatch_coordinates']==[[1,7,2]]
    assert m['mismatches_by_clip']==[0,1]
    assert m['mismatches_by_second']==[0,0,0,0,0,0,0,1,0,0]
    assert m['max_rounding_cell_excess']>0


def test_signed_zero_is_numeric_equality_not_a_bitwise_claim():
    m=float16_cell_analysis(np.array([-0.],np.float32),np.array([0.],np.float16))
    assert m['float16_array_equal'] is True and m['max_rounding_cell_excess']==0


@pytest.mark.parametrize('reference',[np.array([1.],np.float32),np.array([65504.],np.float16),np.array([-65504.],np.float16)])
def test_cell_requires_original_finite_interior_float16_reference(reference):
    with pytest.raises(ValueError):float16_cell_analysis(reference.astype(np.float32),reference)


@pytest.mark.parametrize('relative',['../escape','/absolute',''])
def test_audit_paths_cannot_escape_mount(tmp_path,relative):
    with pytest.raises(ContractError):pinned_child(tmp_path,relative)


def fixture_manifest(tmp_path):
    models=tmp_path/'models';inputs=tmp_path/'inputs'
    files={}
    for prefix,names in ((models,REQUIRED_MODEL_FILES),(inputs,REQUIRED_INPUT_FILES)):
        entries={}
        for relative in names:
            p=prefix/relative;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(relative.encode())
            entries[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
        files['model_files' if prefix==models else 'input_files']=entries
    m={'schema':'aval.stage2c-permitted-train.v1','sample_ids':list(PERMITTED_IDS),
       'audio_card_sha256':AUDIO_CARD,'video_card_sha256':VIDEO_CARD,**files}
    p=inputs/'manifest.json';p.write_text(json.dumps(m))
    return p,m,models,inputs


def bound(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_hash_is_external_and_tampering_fails(tmp_path):
    p,m,models,inputs=fixture_manifest(tmp_path)
    expected=bound(p)
    assert load_manifest(p,expected,models,inputs)['sample_ids']==list(PERMITTED_IDS)
    (models/'video/weights.pt').write_bytes(b'tampered')
    with pytest.raises(ContractError):load_manifest(p,expected,models,inputs)
    with pytest.raises(ContractError):load_manifest(p,'0'*64,models,inputs)


@pytest.mark.parametrize('key,value',[
    ('sample_ids',list(reversed(PERMITTED_IDS))),
    ('audio_card_sha256','0'*64),('video_card_sha256','0'*64),
])
def test_manifest_cannot_change_checkpoint_or_identity_binding(tmp_path,key,value):
    p,m,models,inputs=fixture_manifest(tmp_path);m[key]=value;p.write_text(json.dumps(m))
    with pytest.raises(ContractError):load_manifest(p,bound(p),models,inputs)


def test_missing_fusion_integrity_sidecar_fails_preflight(tmp_path):
    p,m,models,inputs=fixture_manifest(tmp_path)
    del m['model_files']['E3/checkpoint_integrity.json'];p.write_text(json.dumps(m))
    with pytest.raises(ContractError):load_manifest(p,bound(p),models,inputs)


def test_heldout_array_is_not_an_allowed_audit_manifest_role(tmp_path):
    p,m,models,inputs=fixture_manifest(tmp_path)
    m['model_files']['audio_train/test_emb.npy']='0'*64;p.write_text(json.dumps(m))
    with pytest.raises(ContractError):load_manifest(p,bound(p),models,inputs)


def test_torch_and_numpy_video_normalization_agree_before_kernel_variants():
    frames=np.arange(10*224*224*3,dtype=np.int64).astype(np.uint8).reshape(1,10,224,224,3)
    actual=normalize_video(frames).numpy()
    reference=(frames.astype(np.float32)/255).transpose(0,1,4,2,3)
    reference=(reference-np.array([.485,.456,.406],np.float32).reshape(1,1,3,1,1))/np.array([.229,.224,.225],np.float32).reshape(1,1,3,1,1)
    assert np.array_equal(actual,reference)
