"""General candidate boundary tests. No invented scores or substitute models."""
import json
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from starlette.requests import Request

from deployment import general_decode as decode
from deployment.auth import owner_digest
from deployment.general_inference import GeneralResearchService, error, response_for_research
from deployment.runtime import DeploymentConfig, DeploymentError
from deployment.uploads import UploadStore


def media_clock():
    meta = {'format': {'start_time': '0'}, 'streams': [
        {'codec_type':'video','start_time':'0','width':224,'height':224,'pix_fmt':'yuv420p',
         'r_frame_rate':'25','avg_frame_rate':'25','time_base':'1/12800'},
        {'codec_type':'audio','start_time':'0','sample_rate':'16000','channels':1,'time_base':'1/16000'}]}
    frames = [{'media_type':'video','best_effort_timestamp':n*512,'width':224,'height':224,'pix_fmt':'yuv420p'} for n in range(250)]
    frames += [{'media_type':'audio','best_effort_timestamp':n*1600,'nb_samples':1600} for n in range(100)]
    return meta, frames


def test_clock_reports_achieved_pts_without_approval_or_repairs():
    plan = decode.clock_plan(*media_clock())
    assert plan['frame_indices'] == list(range(0,250,25))
    assert plan['achieved_frame_seconds'] == list(range(10))
    assert plan['historical_shared_clock_approved'] is False
    assert plan['repairs_applied'] is False


def test_fractional_fps_uses_producer_indices_and_reports_offset():
    meta, frames = media_clock()
    meta['streams'][0].update(r_frame_rate='30000/1001',avg_frame_rate='30000/1001',time_base='1/30000')
    video = [dict(frames[0],best_effort_timestamp=n*1001) for n in range(300)]
    p = decode.clock_plan(meta,video+frames[250:])
    assert p['frame_indices'] == [0,29,59,89,119,149,179,209,239,269]
    assert p['achieved_frame_pts'][1] == '29029/30000'
    assert p['frame_offset_seconds'][1] == float(decode.Fraction(-971,30000))


@pytest.mark.parametrize('change,code',[
    (lambda m,f:m['streams'].pop(),'MISSING_OR_AMBIGUOUS_STREAMS'),
    (lambda m,f:m['format'].update(start_time='0.1'),'MODALITY_CLOCK_MISMATCH'),
    (lambda m,f:m['streams'][1].update(start_time='0.01'),'MODALITY_CLOCK_MISMATCH'),
    (lambda m,f:m['streams'][0].update(width=1921),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(height=0),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(pix_fmt='yuv420p10le'),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(color_transfer='smpte2084'),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(side_data_list=[{'rotation':90}]),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(tags={'rotate':'180'}),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(sample_aspect_ratio='2:1'),'GENERAL_VIDEO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(avg_frame_rate='24'),'GENERAL_VFR_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(r_frame_rate='121',avg_frame_rate='121'),'GENERAL_VFR_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(time_base='1/100'),'GENERAL_CLOCK_RESOLUTION_UNSUPPORTED'),
    (lambda m,f:m['streams'][0].update(time_base='0'),'GENERAL_CLOCK_RESOLUTION_UNSUPPORTED'),
    (lambda m,f:m['streams'][1].update(channels=6),'GENERAL_AUDIO_PROFILE_UNSUPPORTED'),
    (lambda m,f:m['streams'][1].update(sample_rate='192000'),'GENERAL_AUDIO_PROFILE_UNSUPPORTED'),
    (lambda m,f:f[1].update(best_effort_timestamp=514),'GENERAL_VIDEO_CLOCK_MISMATCH'),
    (lambda m,f:f[20].update(width=225),'GENERAL_VIDEO_CLOCK_MISMATCH'),
    (lambda m,f:f[251].update(best_effort_timestamp=1602),'GENERAL_AUDIO_CLOCK_MISMATCH'),
    (lambda m,f:f[251].update(nb_samples=0),'GENERAL_AUDIO_CLOCK_MISMATCH'),
    (lambda m,f:f.pop(),'GENERAL_AUDIO_INCOMPLETE'),
    (lambda m,f:f.pop(249),'GENERAL_VIDEO_INCOMPLETE'),
    (lambda m,f:f.clear(),'GENERAL_FRAME_COUNT_INVALID'),
    (lambda m,f:m.clear(),'GENERAL_CLOCK_METADATA_INVALID'),
    (lambda m,f:m['streams'][0].update(r_frame_rate='NaN'),'GENERAL_CLOCK_METADATA_INVALID'),
])
def test_clock_and_profile_rejections(change,code):
    meta,frames = media_clock();change(meta,frames)
    with pytest.raises(DeploymentError) as e:decode.clock_plan(meta,frames)
    assert e.value.code == code


def test_subprocess_diagnostic_limit():
    with patch.object(decode,'MAX_OUTPUT',100):
        with pytest.raises(DeploymentError,match='GENERAL_DECODE_OUTPUT_LIMIT'):
            decode.bounded([sys.executable,'-c','import sys; sys.stderr.write("x"*100000)'])


def test_subprocess_timeout_and_sanitized_failure():
    with patch.object(decode,'WALL_SECONDS',.1):
        with pytest.raises(DeploymentError,match='GENERAL_DECODE_TIMEOUT'):
            decode.bounded([sys.executable,'-c','import time;time.sleep(20)'])
    with pytest.raises(DeploymentError,match='GENERAL_DECODE_FAILED'):
        decode.bounded([sys.executable,'-c','import sys;sys.stderr.write("PRIVATE SECRET PATH");sys.exit(1)'])


def test_child_never_inherits_credentials(monkeypatch):
    monkeypatch.setenv('AVAL_GRADIO_PASSWORD','private-test-value')
    result = decode.bounded([sys.executable,'-c','import os;print(os.getenv("AVAL_GRADIO_PASSWORD","absent"))'])
    assert result.strip() == b'absent'


def test_research_sandbox_refuses_missing_kernel_evidence():
    with patch.object(Path,'read_text',side_effect=OSError):
        with pytest.raises(DeploymentError,match='GENERAL_RESEARCH_SANDBOX_REQUIRED'):
            decode.require_research_sandbox()


def test_demuxer_is_forced_and_external_reference_support_disabled():
    for suffix in ('.mp4','.mov'):
        flags=decode.GeneralDecoder.input_flags(suffix)
        assert flags[flags.index('-protocol_whitelist')+1]=='file'
        assert flags[flags.index('-enable_drefs')+1]=='0'
        assert flags[flags.index('-use_absolute_path')+1]=='0'
        assert flags[flags.index('-f')+1]=='mov'
    assert decode.GeneralDecoder.input_flags('.webm')[-1]=='matroska'


def worker(tmp_path):
    service=object.__new__(GeneralResearchService)
    service.config=DeploymentConfig(tmp_path,(),());service._lock=Lock();service.upload_store=None
    return service


def test_concurrent_call_rejected_without_model_access(tmp_path):
    s=worker(tmp_path);s._lock.acquire()
    try:assert s.predict_research(b'','a.mp4')==error('RESOURCE_BUSY')
    finally:s._lock.release()


def test_errors_sanitized_and_lock_released(tmp_path):
    s=worker(tmp_path)
    s._integrity=lambda:(_ for _ in ()).throw(RuntimeError('secret weight path'))
    assert s.predict_research(b'','a.mp4')==error('GENERAL_INFERENCE_FAILED')
    assert not s._lock.locked()


def test_decode_failure_cleanup(tmp_path):
    d=object.__new__(decode.GeneralDecoder);d.config=DeploymentConfig(tmp_path,(),())
    meta,_=media_clock();created=[]
    def failed_scan(name,args):
        source=Path(args[-1]);created.append(source)
        assert source.read_bytes()==b'unit media bytes, not a prediction'
        raise DeploymentError('GENERAL_DECODE_TIMEOUT')
    d.run=failed_scan
    # Admission mocked only to exercise private-temp cleanup, no fake prediction.
    def admission(*args,**kwargs):
        # Populate the probe closure's metadata without running an unsafe decoder.
        with patch.object(d,'run',return_value=json.dumps({**meta,'format':{'format_name':'mov','start_time':'0'},
            'streams':[dict(s,codec_name='h264' if s['codec_type']=='video' else 'aac',duration='10') for s in meta['streams']]}).encode()):
            kwargs['probe'](tmp_path/'admission.mp4')
        return {'error_code':'RAW_INFERENCE_GATE_CLOSED'}
    with patch.object(decode,'validate_upload',side_effect=admission):
        with pytest.raises(DeploymentError,match='GENERAL_DECODE_TIMEOUT'):
            d.decode(b'unit media bytes, not a prediction','unit.mp4')
    assert created and not created[0].parent.exists()


def request(app,store,token):
    return SimpleNamespace(request=Request({'type':'http','app':app,'headers':[
        (b'cookie',('access-token-unit='+token).encode())],'aval.upload_guard':store}))


def test_cross_login_stale_handle_and_cleanup(tmp_path):
    s=worker(tmp_path);store=UploadStore(tmp_path,1024);s.upload_store=store
    app=SimpleNamespace(auth=True,auth_dependency=None,tokens={'a':'same-name','b':'same-name'},cookie_id='unit')
    store.bind_auth(app);store.start()
    try:
        handle,entry=store.reserve(owner_digest('same-name','a'));entry.path.write_bytes(b'unit')
        store.publish(handle,'unit.mp4')
        s.predict_research=lambda *args:error('UNIT_NO_MODEL_FORWARD')
        assert s.submit_owned(handle,request(app,store,'b'))==error('UPLOAD_NOT_FOUND')
        assert entry.path.exists()
        assert s.submit_owned(handle,request(app,store,'a'))==error('UNIT_NO_MODEL_FORWARD')
        assert not entry.path.exists()
        assert s.submit_owned(handle,request(app,store,'a'))==error('UPLOAD_NOT_FOUND')
        assert s.submit_owned('/models/E3/best.pt',request(app,store,'a'))==error('UPLOAD_NOT_FOUND')
    finally:store.close()


def test_login_revocation_during_work_suppresses_result_and_cleans(tmp_path):
    s=worker(tmp_path);store=UploadStore(tmp_path,1024);s.upload_store=store
    app=SimpleNamespace(auth=True,auth_dependency=None,tokens={'a':'user'},cookie_id='unit')
    store.bind_auth(app);store.start()
    try:
        handle,entry=store.reserve(owner_digest('user','a'));entry.path.write_bytes(b'unit');store.publish(handle,'unit.mp4')
        def revoke(*args):app.tokens.clear();return error('UNIT_NO_MODEL_FORWARD')
        s.predict_research=revoke
        assert s.submit_owned(handle,request(app,store,'a'))==error('UPLOAD_AUTHENTICATION_REQUIRED')
        assert not entry.path.exists()
    finally:store.close()


def test_authentication_before_any_lease(tmp_path):
    s=worker(tmp_path);s.upload_store=SimpleNamespace(lease=lambda *args:pytest.fail('lease before authentication'))
    assert s.submit_owned('handle',SimpleNamespace(request=None))==error('UPLOAD_AUTHENTICATION_REQUIRED')


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(score=.1),lambda r:r.update(error_code='/private/path'),
    lambda r:r.update(production_predictions_enabled=True),lambda r:r.update(predictions={}),
    lambda r:r.update(status='APPROVED'),lambda r:r.update(schema='wrong')])
def test_response_rejects_unauthorized_and_unsanitized_fields(mutation):
    r=error('UNIT_TEST');mutation(r)
    with pytest.raises(ValueError):response_for_research(r)


def test_default_app_has_no_research_dependency():
    # The optional entry point does not alter the default readiness or scope.
    import inspect
    from deployment.gradio_app import create_app
    assert inspect.signature(create_app).parameters['general_research'].default is None
    from deployment.release_gate import unapproved_prediction
    assert unapproved_prediction('upload')['inference_enabled'] is False
    assert unapproved_prediction('live')['inference_enabled'] is False


def test_research_ui_private_queue_and_cancellation():
    import gradio as gr
    from deployment.ui.general_research_section import build_general_research_section
    service=SimpleNamespace(submit_owned=lambda *_:error('UPLOAD_REQUIRED'))
    with gr.Blocks() as demo:build_general_research_section(gr,service)
    call=next(f for f in demo.fns.values() if getattr(f.fn,'__name__','')=='infer_general')
    assert call.api_visibility=='private' and call.concurrency_limit==1 and call.trigger_mode=='once'
    assert isinstance(call.inputs[0],gr.Textbox)
    assert any(f.cancels for f in demo.fns.values())
    loading,final=list(call.fn(None,None))
    assert loading[1] is None and loading[2]['interactive'] is False
    assert final[1]==error('UPLOAD_REQUIRED') and 'No score' in final[0]
    assert not final[2]['interactive']
