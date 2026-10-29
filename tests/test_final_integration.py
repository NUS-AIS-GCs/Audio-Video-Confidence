"""New provisioning, restricted upload and closed release-boundary tests.

Small bytes below are transport fixtures, never weights or model predictions.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from deployment import provision_artifacts as provisioning
from deployment.internal_backend import InternalTrainBackend
from deployment.prediction_adapter import DEMO_CASES, InternalPredictionAdapter
from deployment.release_gate import prediction_readiness, unapproved_prediction


def test_catalog_binds_original_acquisition_and_frozen_fusion():
    from fusion.inference import CANDIDATES
    from deployment.train_manifest import AUDIO_CARD, VIDEO_CARD
    files = provisioning.artifact_catalog(True)
    assert len(files) == 18 and len(provisioning.artifact_catalog(False)) == 14
    assert files['audio/model_card.json']['sha256'] == AUDIO_CARD
    assert files['video/model_card.json']['sha256'] == VIDEO_CARD
    for name in ('E3', 'A2'):
        assert files[name+'/best.pt']['sha256'] == CANDIDATES[name].checkpoint_sha256
        assert files[name+'/checkpoint_metadata.json']['sha256'] == CANDIDATES[name].metadata_sha256
        assert files[name+'/frozen_operating_point.json']['sha256'] == CANDIDATES[name].operating_point_sha256


@pytest.fixture
def small_catalog(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    (source/'file.bin').write_bytes(b'not a checkpoint')
    catalog = {'file.bin': {'bytes': 16, 'sha256': hashlib.sha256(b'not a checkpoint').hexdigest()}}
    monkeypatch.setattr(provisioning, 'artifact_catalog', lambda *a: catalog)
    return source


def test_atomic_local_provision_and_reuse_never_grants_release(tmp_path, small_catalog):
    dest = tmp_path/'models'
    r = provisioning.provision_artifacts(dest, source_root=small_catalog)
    assert r['ready'] and r['inference_enabled'] is False
    assert (dest/'file.bin').read_bytes() == b'not a checkpoint'
    assert not list(tmp_path.glob('.aval-provision-*'))
    # Fully verified reuse requires no network, even if offline.
    assert provisioning.provision_artifacts(dest)['ready']
    assert not unapproved_prediction('upload')['inference_enabled']


@pytest.mark.parametrize('bad', [b'different bytes', b'x'*17, b''])
def test_failed_copy_never_publishes_partial_model_directory(tmp_path, small_catalog, bad):
    (small_catalog/'file.bin').write_bytes(bad)
    with pytest.raises(provisioning.ProvisionError):
        provisioning.provision_artifacts(tmp_path/'models', source_root=small_catalog)
    assert not (tmp_path/'models').exists() and not list(tmp_path.glob('.aval-provision-*'))


def test_existing_modified_artifact_refuses_repair_or_startup(tmp_path, small_catalog):
    dest = tmp_path/'models'
    provisioning.provision_artifacts(dest, source_root=small_catalog)
    (dest/'file.bin').write_bytes(b'altered')
    with pytest.raises(provisioning.ProvisionError, match='EXISTING_ARTIFACTS'):
        provisioning.provision_artifacts(dest, source_root=small_catalog)
    assert (dest/'file.bin').read_bytes() == b'altered'


def test_disk_exhaustion_precedes_copy_or_network(tmp_path, small_catalog):
    with mock.patch.object(provisioning.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)), \
         mock.patch.object(provisioning, '_copy_bounded', side_effect=AssertionError('copy forbidden')):
        with pytest.raises(provisioning.ProvisionError, match='INSUFFICIENT_ARTIFACT_DISK'):
            provisioning.provision_artifacts(tmp_path/'models', source_root=small_catalog)


@pytest.mark.parametrize('repo,revision,token', [
    ('owner/repo', 'main', 'secret'), ('https://host/private?token=secret', 'a'*40, 'secret'),
    ('owner/repo', 'a'*40, None),
])
def test_remote_source_requires_pinned_private_repository_and_secret(repo, revision, token):
    with pytest.raises(provisioning.ProvisionError):
        provisioning._private_hf_source(repo, revision, token, {})


def test_remote_source_rejects_public_repo_and_sanitizes_sdk_errors():
    import huggingface_hub
    client = mock.Mock()
    with mock.patch.object(provisioning.importlib.metadata, 'version', return_value='2.3.0'), \
         mock.patch.object(huggingface_hub, 'HfApi', return_value=client):
        client.repo_info.return_value = SimpleNamespace(private=False, sha='a'*40)
        with pytest.raises(provisioning.ProvisionError, match='PRIVATE_IMMUTABLE_REPOSITORY_REQUIRED'):
            provisioning._private_hf_source('owner/repo', 'a'*40, 'secret', {})
        client.repo_info.side_effect = RuntimeError('Authorization Bearer secret /private/path')
        with pytest.raises(provisioning.ProvisionError) as exc:
            provisioning._private_hf_source('owner/repo', 'a'*40, 'secret', {})
        assert str(exc.value) == 'PRIVATE_ARTIFACT_SOURCE_UNAVAILABLE'


def test_runtime_bindings_preserve_original_artifact_paths_and_hashes(tmp_path):
    b = provisioning.runtime_bindings(tmp_path)
    assert b['AVAL_AUDIO_BASE_PATH'] == str(tmp_path/'audio/encoder.pt')
    assert b['AVAL_E3_DIR'] == str(tmp_path/'E3')
    assert b['AVAL_AUDIO_NORMALIZATION_SHA256'] == provisioning.artifact_catalog()['audio/norm_stats.json']['sha256']
    assert not any('TOKEN' in k or 'PASSWORD' in k for k in b)


@pytest.mark.parametrize('scope', ['upload', 'live'])
def test_release_cannot_be_promoted_by_environment_or_caller_flags(monkeypatch, scope):
    monkeypatch.setenv('AVAL_INFERENCE_ENABLED', 'true')
    monkeypatch.setenv('AVAL_RELEASE_APPROVED', 'true')
    status = prediction_readiness(scope)
    assert status.to_dict()['approved_release_id'] is None
    assert unapproved_prediction(scope)['predictions'] is None
    with pytest.raises(ValueError):
        replace(status, inference_enabled=True)
    with pytest.raises(ValueError):
        replace(status, approved_release_id='caller-claim')


def unloaded_adapter(tmp_path):
    from fusion.encoder_interfaces import Artifact
    b = object.__new__(InternalTrainBackend)
    content = b'TRAIN upload transport fixture; never decoded'
    b.config = SimpleNamespace(max_upload_bytes=1000)
    b.sources = {c.video_id: Artifact(str(tmp_path/(c.video_id+'.mp4')),
        hashlib.sha256(content if i == 0 else str(i).encode()).hexdigest())
        for i,c in enumerate(DEMO_CASES[:5])}
    return InternalPredictionAdapter(b), content


def test_uploaded_unknown_media_never_reaches_existing_decoder(tmp_path):
    a, content = unloaded_adapter(tmp_path)
    with mock.patch.object(a.backend, 'predict_internal', side_effect=AssertionError('decode forbidden'), create=True):
        assert a.predict_uploaded_train(b'unknown', 'clip.mp4')['error_code'] == 'UNAPPROVED_TRAIN_SOURCE'
        assert a.predict_uploaded_train(content, '../clip.mp4')['error_code'] == 'UNSUPPORTED_FILENAME'
        assert a.predict_uploaded_train(b'x'*1001, 'clip.mp4')['error_code'] == 'UPLOAD_TOO_LARGE'


def test_internal_upload_authentication_before_lease_and_path_read(tmp_path):
    a, _ = unloaded_adapter(tmp_path)
    a.upload_store = mock.Mock()
    for request in (None, SimpleNamespace(username='spoof'), SimpleNamespace(request=object())):
        assert a.submit_owned_upload('forged-handle', request)['error_code'] == 'INTERNAL_AUTHENTICATION_REQUIRED'
    a.upload_store.lease.assert_not_called()


def test_login_revocation_drops_even_internal_results(tmp_path):
    a, _ = unloaded_adapter(tmp_path)
    with mock.patch('deployment.prediction_adapter.login_owner', side_effect=['owner', None]), \
         mock.patch.object(a, 'predict_case', return_value={'private_marker':'must not escape'}):
        r = a.submit('original-1', SimpleNamespace(request=None))
    assert r['error_code'] == 'INTERNAL_AUTHENTICATION_REQUIRED' and 'private_marker' not in r


def test_catalog_modification_is_rejected_before_any_source_access():
    with mock.patch.object(provisioning, 'CATALOG_SHA256', '0'*64):
        with pytest.raises(provisioning.ProvisionError, match='ARTIFACT_CATALOG_MODIFIED'):
            provisioning.artifact_catalog()


def test_internal_upload_lease_is_login_specific_consumed_once_and_always_cleaned(tmp_path):
    from starlette.requests import Request
    from deployment.auth import owner_digest
    from deployment.uploads import UploadStore
    adapter, content = unloaded_adapter(tmp_path)
    store = UploadStore(tmp_path/'uploads', 1000)
    app = SimpleNamespace(auth=True, auth_dependency=None, tokens={'first':'alice','second':'alice'}, cookie_id='unit')
    store.start(); store.bind_auth(app); adapter.upload_store=store

    def request(token):
        return SimpleNamespace(request=Request({'type':'http','app':app,'aval.upload_guard':store,
            'headers':[(b'cookie',f'access-token-unsecure-unit={token}'.encode())]}))

    try:
        handle, entry = store.reserve(owner_digest('alice','first'))
        entry.path.write_bytes(content)
        filename = Path(adapter.backend.sources[DEMO_CASES[0].video_id].path).name
        store.publish(handle, filename); path=store._entries[handle].path
        with mock.patch.object(adapter, 'predict_uploaded_train', side_effect=RuntimeError('private detail')) as predict:
            assert adapter.submit_owned_upload(handle, request('second'))['error_code']=='UPLOAD_NOT_FOUND'
            assert path.is_file(); predict.assert_not_called()
            assert adapter.submit_owned_upload(handle, request('first'))['error_code']=='INTERNAL_INFERENCE_FAILED'
            predict.assert_called_once_with(content, filename)
        assert not path.exists() and not store._entries
        assert adapter.submit_owned_upload(handle,request('first'))['error_code']=='UPLOAD_NOT_FOUND'
    finally:
        store.close()


def test_validated_completed_live_window_reports_closed_prediction_scope():
    from deployment.live.service import LiveCaptureService
    svc = LiveCaptureService()
    try:
        started = svc.start('owner', {'sample_rate':16000})
        session = svc._sessions[started['session']]
        # Contract-only seam, no model score. Existing buffer tests exercise actual samples/timing.
        with mock.patch.object(session.buffer, 'snapshot', return_value={'timing':{}}), \
             mock.patch.object(session.buffer, 'to_raw_media_input', return_value=SimpleNamespace(
                 waveform=SimpleNamespace(shape=(1,160000)), frames=SimpleNamespace(shape=(1,10,224,224,3)))):
            r = svc.check('owner', {'session':started['session'],'tab_key':started['tab_key']})
        assert r['status'] == 'CONTRACT_VALID' and r['predictions'] is None
        assert r['prediction_readiness']['error_code'] == 'LIVE_INFERENCE_GATE_CLOSED'
        assert 'LIVE_INPUT_COMPATIBILITY_NOT_APPROVED' in r['prediction_readiness']['blockers']
    finally:
        svc.close()
