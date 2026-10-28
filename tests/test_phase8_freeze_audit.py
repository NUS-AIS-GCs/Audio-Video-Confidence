"""Freeze fixes: import isolation, non-assert authentication, and response safety."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch

from deployment import internal_backend, seed_artifact_volume, train_manifest
from fusion.contracts import ContractError

ROOT=Path(__file__).resolve().parents[1]


def test_backend_import_does_not_import_profile_command():
    code="import sys;import deployment.internal_backend;sys.exit(1 if 'deployment.stage2c_probe' in sys.modules else 0)"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_seed_module_is_import_safe_without_container_files():
    with mock.patch.object(sys,'argv',['unrelated-command']),mock.patch('pathlib.Path.read_bytes',side_effect=AssertionError('no import I/O')):
        module=importlib.reload(seed_artifact_volume)
    assert callable(module.main) and callable(module.copy_verified_files)


def test_verified_copy_is_exclusive_and_preserves_bytes(tmp_path):
    source=tmp_path/'source';source.mkdir();cache=tmp_path/'cache';cache.mkdir()
    data=b'unit-only checksum fixture';(source/'weights.pt').write_bytes(data)
    bindings={'weights.pt':hashlib.sha256(data).hexdigest()}
    assert seed_artifact_volume.copy_verified_files(bindings,source,cache)==len(data)
    assert (cache/'weights.pt').read_bytes()==data
    with pytest.raises(FileExistsError):seed_artifact_volume.copy_verified_files(bindings,source,cache)
    assert (cache/'weights.pt').read_bytes()==data


def test_corrupt_source_is_rejected_before_copy(tmp_path):
    source=tmp_path/'source';source.mkdir();cache=tmp_path/'cache';cache.mkdir()
    (source/'weights.pt').write_bytes(b'corrupt')
    with pytest.raises(ContractError):seed_artifact_volume.copy_verified_files({'weights.pt':'0'*64},source,cache)
    assert not list(cache.iterdir())


def test_copy_guard_remains_active_with_python_optimization(tmp_path):
    source=tmp_path/'source';source.mkdir();cache=tmp_path/'cache';cache.mkdir()
    (source/'weights.pt').write_bytes(b'corrupt')
    code="""import sys
from deployment.seed_artifact_volume import copy_verified_files
from fusion.contracts import ContractError
try: copy_verified_files({'weights.pt':'0'*64},sys.argv[1],sys.argv[2])
except ContractError: sys.exit(0)
sys.exit(3)
"""
    result=subprocess.run([sys.executable,'-O','-B','-c',code,str(source),str(cache)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not list(cache.iterdir())


def test_postcopy_hash_detects_changed_copy(tmp_path):
    source=tmp_path/'source';source.mkdir();cache=tmp_path/'cache';cache.mkdir()
    data=b'unit-only valid source';(source/'weights.pt').write_bytes(data)
    with mock.patch('deployment.seed_artifact_volume.shutil.copyfileobj',side_effect=lambda i,o:o.write(b'altered')):
        with pytest.raises(ContractError):seed_artifact_volume.copy_verified_files({'weights.pt':hashlib.sha256(data).hexdigest()},source,cache)


@pytest.mark.parametrize('destination',['source','source/child'])
def test_snapshot_cannot_overlap_source(tmp_path,destination):
    (tmp_path/'source').mkdir()
    with pytest.raises(ContractError):seed_artifact_volume.copy_verified_files({},tmp_path/'source',tmp_path/destination)


def test_snapshot_path_escape_is_rejected(tmp_path):
    with pytest.raises(ContractError):seed_artifact_volume.copy_verified_files({'../outside':'0'*64},tmp_path/'s',tmp_path/'c')


@pytest.mark.parametrize('document',['[]','null','{"schema":1,"schema":2}'])
def test_manifest_rejects_wrong_shape_or_duplicate_keys(tmp_path,document):
    p=tmp_path/'manifest.json';p.write_text(document)
    with pytest.raises(ContractError):train_manifest.load_manifest(p,hashlib.sha256(p.read_bytes()).hexdigest(),tmp_path,tmp_path)


def genuine_response():
    p=ROOT/'_local/phase8_stage2d/profile-final/native-volume/run/profile.json'
    if not p.is_file():pytest.skip('genuine Stage 2D response unavailable')
    return json.loads(p.read_text())['requests'][0]['response']


@pytest.mark.parametrize('field,value',[('public_readiness',None),('diagnostics',[]),('timings',None),('timings',{})])
def test_malformed_nested_response_has_contract_error(field,value):
    response=genuine_response();response[field]=value
    with pytest.raises(ContractError):internal_backend.validate_internal_response(response)


def test_inconsistent_sigmoid_score_is_rejected():
    response=genuine_response();response['predictions']['E3']['logit']=0.
    with pytest.raises(ContractError):internal_backend.validate_internal_response(response)


def test_error_issues_cannot_leak_paths():
    response=internal_backend.error_result('NOT_READY')
    response['issues']=[{'category':'/private/artifact','code':'NOT_READY'}]
    with pytest.raises(ContractError):internal_backend.validate_internal_response(response)


@pytest.mark.parametrize('value',[None,[],{1:'invalid key'}])
def test_serializer_rejects_non_response_objects(value):
    with pytest.raises(ContractError):internal_backend.response_for_review(value)


def test_runtime_thread_drift_is_rejected_before_artifact_access():
    backend=object.__new__(internal_backend.InternalTrainBackend)
    backend.audio=mock.Mock();backend.video=mock.Mock();backend.study=mock.Mock()
    backend.study.adapters={name:mock.Mock() for name in ('E3','A2')}
    backend._models=(backend.audio.model,backend.video.model,*(backend.study.adapters[n]._model for n in ('E3','A2')))
    backend._execution_profile=(-1,-1,False)
    with pytest.raises(ContractError,match='execution profile'):backend._integrity()
