"""Independent checks for diagnostic statistics and sign-changing brackets."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deployment import runtime

spec=importlib.util.spec_from_file_location('compatibility_audit',Path(__file__).with_name('integration_inference_compatibility.py'))
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_distribution_preserves_fractional_errors_and_percentiles():
    actual=np.array([0.,1.,2.,4.],np.float32)
    result=audit.distribution(actual,np.zeros(4,np.float32))
    assert result['absolute_error_percentiles']['50']==1.5
    assert result['absolute_error_percentiles']['100']==4.
    assert result['different_values']==3 and result['values']==4
    assert result['mae']==1.75


def test_unsigned_pixel_subtraction_does_not_wrap():
    result=audit.distribution(np.array([0],np.uint8),np.array([255],np.uint8))
    assert result['max_absolute_error']==255.


@pytest.mark.parametrize('evaluate',[lambda x:x,lambda x:1-x,lambda x:x+.1*np.sin(20*x)])
def test_boundary_preserves_straddling_endpoints_without_monotonic_assumption(evaluate):
    row=audit.boundary_bracket(evaluate,.4)
    assert (row['logit_low']>=.4)!=(row['logit_high']>=.4)
    assert row['alpha_high']-row['alpha_low']==2**-24


@pytest.mark.parametrize('evaluate',[lambda x:1.,lambda x:float('nan'),lambda x:float('inf')])
def test_invalid_boundary_is_not_reported_as_a_success(evaluate):
    with pytest.raises(ValueError):audit.boundary_bracket(evaluate,.4)


@pytest.mark.parametrize('video,audio,expected',[
    (9.95,9.95,'RAW_INFERENCE_GATE_CLOSED'),
    (9.95,10.,'RAW_INFERENCE_GATE_CLOSED'),
    (10.05,10.,'RAW_INFERENCE_GATE_CLOSED'),
    (10.05,10.05,'RAW_INFERENCE_GATE_CLOSED'),
    (float(np.nextafter(9.95,-np.inf)),10.,'INVALID_DURATION'),
    (float(np.nextafter(10.05,np.inf)),10.,'INVALID_DURATION'),
    (9.96,10.04,'MODALITY_DURATION_MISMATCH'),
    (9.95,10.05,'MODALITY_DURATION_MISMATCH'),
    (float('nan'),10.,'INVALID_DURATION'),
    (float('inf'),10.,'INVALID_DURATION'),
])
def test_declared_inclusive_duration_endpoints_without_an_epsilon(tmp_path,video,audio,expected):
    response=runtime.validate_upload(b'fixture',filename='clip.mp4',
        config=runtime.DeploymentConfig(tmp_path,(),()),
        readiness=runtime.ReadinessReport('MODEL_NOT_READY',(),False),
        probe=lambda _:runtime.MediaInfo(('mp4',),(('major_brand','mp42'),),
            (runtime.MediaStream('video','h264',video),runtime.MediaStream('audio','aac',audio))))
    assert response['error_code']==expected
    assert response['predictions'] is None
