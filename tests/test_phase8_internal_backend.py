"""Boundary/lifecycle tests; genuine frozen output preservation runs in Docker."""
import hashlib
import copy
import json
from pathlib import Path
from threading import Lock
from unittest import mock

import pytest

from deployment import internal_backend as backend
from deployment.runtime import DeploymentConfig, DeploymentError
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact


SID='c---zaDCTaE__08'
FILENAME='c---zaDCTaE.mp4'
CONTENT=b'unit fixture, never a genuine prediction'


def unloaded_fixture(tmp_path):
    b=object.__new__(backend.InternalTrainBackend)
    b._lock=Lock();b._completed=0
    b.sources={SID:Artifact(str(tmp_path/FILENAME),hashlib.sha256(CONTENT).hexdigest())}
    b.config=DeploymentConfig(tmp_path,(),())
    b.ffmpeg=tmp_path/'ffmpeg';b.ffprobe=tmp_path/'ffprobe'
    return b


def test_repeated_dispatch_reuses_backend_and_never_calls_loader(tmp_path):
    b=unloaded_fixture(tmp_path);calls=[]
    # Exercise dispatch/lifecycle only; no substitute score or model forward.
    def dispatch(*args):
        calls.append(id(b));return backend.error_result('UNIT_TEST_NO_MODEL_FORWARD')
    b._run=dispatch
    with mock.patch('torch.load',side_effect=AssertionError('request must not reload models')):
        for _ in range(3):assert b.predict_internal(SID,CONTENT,FILENAME)['predictions'] is None
    assert calls==[id(b)]*3 and not b._lock.locked()


@pytest.mark.parametrize('sid,content,filename,expected',[
    ('unapproved',CONTENT,FILENAME,'UNAPPROVED_TRAIN_SOURCE'),
    (SID,None,FILENAME,'UNSUPPORTED_UPLOAD'),
    (SID,b'',FILENAME,'UNSUPPORTED_UPLOAD'),
    (SID,CONTENT,'../'+FILENAME,'UNSUPPORTED_FILENAME'),
    (SID,CONTENT,'x\\'+FILENAME,'UNSUPPORTED_FILENAME'),
    (SID,CONTENT,'clip.webm','UNSUPPORTED_EXTENSION'),
    (SID,b'different',FILENAME,'SOURCE_PROVENANCE_MISMATCH'),
    (SID,CONTENT,'other.mp4','SOURCE_PROVENANCE_MISMATCH'),
])
def test_bad_or_unapproved_media_never_reaches_decode(tmp_path,sid,content,filename,expected):
    b=unloaded_fixture(tmp_path)
    with mock.patch.object(b,'_run',side_effect=AssertionError('decode forbidden')):
        r=b.predict_internal(sid,content,filename)
    assert r['error_code']==expected and r['predictions'] is None
    assert r['production_predictions_enabled'] is False and not b._lock.locked()


def test_upload_byte_limit_precedes_any_probe(tmp_path):
    b=unloaded_fixture(tmp_path);b.config=DeploymentConfig(tmp_path,(),(),max_upload_bytes=4)
    assert b.predict_internal(SID,CONTENT,FILENAME)['error_code']=='UPLOAD_TOO_LARGE'


def test_concurrent_request_is_rejected_without_queueing(tmp_path):
    b=unloaded_fixture(tmp_path);b._lock.acquire()
    try:assert b.predict_internal(SID,CONTENT,FILENAME)['error_code']=='RESOURCE_BUSY'
    finally:b._lock.release()


def test_internal_readiness_never_exposes_predictions(tmp_path):
    r=unloaded_fixture(tmp_path).readiness()
    assert r['status']=='MODEL_NOT_READY' and r['predictions'] is None
    assert r['inference_enabled'] is False
    assert any(x['category']=='raw_to_train_export_parity' for x in r['issues'])


def valid_probe():
    return {'format':{'format_name':'mov,mp4','start_time':'0','tags':{'major_brand':'isom'}},
            'streams':[{'codec_type':'video','codec_name':'h264','duration':'10','start_time':'0',
                        'width':640,'height':360,'r_frame_rate':'25/1'},
                       {'codec_type':'audio','codec_name':'aac','duration':'10','start_time':'0'}]}


def test_probe_retains_clock_evidence_and_reuses_phase8_media_schema(tmp_path):
    with mock.patch.object(backend,'_run_bounded_process',return_value=(0,json.dumps(valid_probe()).encode(),b'')):
        media,clock=backend.probe_with_clocks(tmp_path/'input.mp4',tmp_path/'ffprobe')
    assert media.streams[0].codec_name=='h264'
    assert clock['video_fps']==25 and clock['audio_start_seconds']==0


@pytest.mark.parametrize('mutation,code',[
    (lambda d:d['streams'][0].update(start_time='0.1'),'MODALITY_CLOCK_MISMATCH'),
    (lambda d:d['streams'][0].update(width=8000),'UNSUPPORTED_VIDEO_DIMENSIONS'),
    (lambda d:d['streams'][0].update(r_frame_rate='240/1'),'UNSUPPORTED_FRAME_RATE'),
    (lambda d:d['streams'][0].update(r_frame_rate='25/0'),'MALFORMED_MEDIA_METADATA'),
    (lambda d:d.update(streams=[]),'MALFORMED_MEDIA_METADATA'),
    (lambda d:d['format'].update(format_name=[]),'MALFORMED_MEDIA_METADATA'),
    (lambda d:d['streams'][1].update(codec_type='video'),'MALFORMED_MEDIA_METADATA'),
    (lambda d:d['format'].update(start_time='nan'),'MODALITY_CLOCK_MISMATCH'),
])
def test_invalid_probe_metadata_is_sanitized(tmp_path,mutation,code):
    payload=valid_probe();mutation(payload)
    with mock.patch.object(backend,'_run_bounded_process',return_value=(0,json.dumps(payload).encode(),b'secret decoder path')):
        with pytest.raises(DeploymentError) as caught:backend.probe_with_clocks(tmp_path/'a.mp4',tmp_path/'ffprobe')
    assert caught.value.code==code and 'secret' not in str(caught.value)


def test_private_request_file_is_cleaned_after_decoder_timeout(tmp_path):
    b=unloaded_fixture(tmp_path);b._integrity=lambda:None;created=[]
    def timeout(command):
        source=Path(command[command.index('-i')+1]);created.append(source)
        assert source.read_bytes()==CONTENT
        raise DeploymentError('PROBE_TIMEOUT')
    with mock.patch.object(backend,'validate_upload',return_value={'error_code':'RAW_INFERENCE_GATE_CLOSED'}), \
         mock.patch.object(backend,'_run_bounded_process',side_effect=timeout):
        result=b.predict_internal(SID,CONTENT,FILENAME)
    assert result['error_code']=='PROBE_TIMEOUT' and result['predictions'] is None
    assert created and all(not p.parent.exists() for p in created)
    assert not b._lock.locked()


def test_failed_admission_stops_before_models_or_decoders(tmp_path):
    b=unloaded_fixture(tmp_path)
    with mock.patch.object(backend,'validate_upload',return_value={'status':'REJECTED','predictions':None,'error_code':'MISSING_OR_AMBIGUOUS_STREAMS'}):
        result=b.predict_internal(SID,CONTENT,FILENAME)
    assert result['error_code']=='MISSING_OR_AMBIGUOUS_STREAMS'


def test_internal_exception_details_do_not_leak(tmp_path):
    b=unloaded_fixture(tmp_path);b._run=mock.Mock(side_effect=RuntimeError('/private/checkpoint token=secret'))
    result=b.predict_internal(SID,CONTENT,FILENAME)
    assert result['error_code']=='INTERNAL_INFERENCE_FAILED'
    assert '/private' not in json.dumps(result) and 'secret' not in json.dumps(result)


def test_model_replacement_rejected_before_forward(tmp_path):
    b=unloaded_fixture(tmp_path);b.audio=mock.Mock();b.video=mock.Mock()
    b.study=mock.Mock();b.study.adapters={n:mock.Mock() for n in ('E3','A2')};b._models=()
    with pytest.raises(ContractError,match='replaced'):b._integrity()


def test_response_boundary_omits_audit_arrays_and_preserves_blocked_state():
    r=backend.error_result('SOURCE_PROVENANCE_MISMATCH');r['_audit_features']=object()
    visible=backend.response_for_review(r)
    assert '_audit_features' not in visible
    assert visible['predictions'] is None and visible['production_predictions_enabled'] is False


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_internal_execution_options_require_actual_booleans(tmp_path,value):
    b=unloaded_fixture(tmp_path)
    assert b.predict_internal(SID,CONTENT,FILENAME,inference_mode=value)['error_code']=='INVALID_INTERNAL_CONFIGURATION'


def genuine_response():
    path=Path(__file__).resolve().parents[1]/'_local/phase8_stage2d/profile-01/native-volume/run/profile.json'
    if not path.is_file():pytest.skip('genuine Docker Stage 2D response unavailable')
    return json.loads(path.read_text())['requests'][0]['response']


def test_genuine_frozen_docker_response_satisfies_internal_contract():
    response=genuine_response();backend.validate_internal_response(response)
    assert response['predictions']['E3']['score_semantics']==backend.SEMANTICS
    assert response['public_readiness']['inference_enabled'] is False


@pytest.mark.parametrize('mutation',[
    lambda r:r['predictions']['E3'].update(threshold=.5),
    lambda r:r['predictions']['E3'].update(score=float('nan')),
    lambda r:r['predictions']['E3'].update(score_semantics='calibrated probability'),
    lambda r:r.update(event_predictions=['unvalidated event']),
    lambda r:r['public_readiness'].update(inference_enabled=True),
    lambda r:r['predictions'].pop('E3'),
    lambda r:r['diagnostics'].update(phase6_shared_clock_accepted=True),
])
def test_contract_rejects_provenance_threshold_or_readiness_promotion(mutation):
    response=copy.deepcopy(genuine_response());mutation(response)
    with pytest.raises(ContractError):backend.validate_internal_response(response)


def test_resource_cleanup_removes_file_and_hook_after_model_failure(tmp_path):
    import numpy as np
    b=unloaded_fixture(tmp_path);b._integrity=lambda:None;paths=[]
    b.audio=mock.Mock();b.video=mock.Mock()
    hook=mock.Mock();b.audio.model.beats.patch_embedding.register_forward_pre_hook.return_value=hook
    b.audio.model.side_effect=RuntimeError('model failure with private details')
    def decode(command):
        source=Path(command[command.index('-i')+1]);paths.append(source)
        backend.sf.write(command[-1],np.zeros(160000,np.float32),16000,subtype='PCM_16')
        return 0,b'',b''
    with mock.patch.object(backend,'validate_upload',return_value={'error_code':'RAW_INFERENCE_GATE_CLOSED'}), \
         mock.patch.object(backend,'_run_bounded_process',side_effect=decode), \
         mock.patch.object(backend,'sample_train_frames',return_value=(np.zeros((10,224,224,3),np.uint8),{})):
        result=b.predict_internal(SID,CONTENT,FILENAME)
    assert result['error_code']=='INTERNAL_INFERENCE_FAILED'
    assert paths and all(not path.parent.exists() for path in paths)
    hook.remove.assert_called_once()
    assert not b._lock.locked()
