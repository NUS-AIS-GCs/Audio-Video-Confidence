"""Schema/security fixtures are synthetic; genuine execution is browser acceptance."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deployment.event_evidence import (class_names, event_evidence, provenance,
    summary, validate_event_evidence, TEMPORAL_UNAVAILABLE)
from deployment.general_inference import ResearchPrediction, response_for_research
from deployment.provision_artifacts import artifact_catalog
from deployment.runtime import DeploymentError
from deployment.ui.general_research_section import research_card
from deployment.ui.result_sections import empty_results, event_score


def predictor():
    scores=np.zeros((10,29),dtype=np.float32)
    scores[:,0]=.1;scores[:,1]=.1;scores[:,28]=.8
    return SimpleNamespace(probs=scores,class_names=class_names())


def result():
    # Unit schema fixture only; no model execution or performance claim.
    r=ResearchPrediction('a'*32,'b'*64,'c'*64,.5,0.,1.,
        event_predictions=event_evidence(predictor(),predictor())).to_dict()
    r['diagnostics']={}
    return r


def test_vocabulary_is_the_acquisition_bound_audio_class_file():
    data=Path('deployment/event_class_names.json').read_bytes()
    assert hashlib.sha256(data).hexdigest()==artifact_catalog()['audio/class_names.json']['sha256']
    assert len(class_names())==29 and class_names()[-1]=='background'
    assert class_names()[:-1]==sorted(class_names()[:-1])


def test_exact_scores_are_serialized_without_recalculation_or_background_renormalization():
    a,v=predictor(),predictor();before=(a.probs.tobytes(),v.probs.tobytes())
    r=event_evidence(a,v)
    assert r['audio']['scores']==a.probs.tolist() and r['video']['scores']==v.probs.tolist()
    assert (a.probs.tobytes(),v.probs.tobytes())==before
    top,background=summary(r,'audio')
    assert top[0]==('Accordion',float(np.float32(.1)))
    assert top[1][0]=='Acoustic guitar'  # stable class-index tie break
    assert background==float(np.float32(.8))
    assert r['audio']['provenance']==provenance('audio')
    assert r['video']['provenance']==provenance('video')


@pytest.mark.parametrize('mutation',[
    lambda p:setattr(p,'probs',p.probs.astype(np.float64)),
    lambda p:setattr(p,'probs',p.probs[:9]),
    lambda p:p.class_names.reverse(),
    lambda p:p.probs.__setitem__((0,0),np.nan),
    lambda p:p.probs.__setitem__((0,0),np.inf),
    lambda p:p.probs.__setitem__((0,0),-.1),
    lambda p:p.probs.__setitem__((0,0),1.1),
    lambda p:p.probs.__setitem__((0,0),.5),
])
def test_invalid_encoder_outputs_fail_closed(mutation):
    a=predictor();mutation(a)
    with pytest.raises(DeploymentError,match='GENERAL_INVALID_EVENT_EVIDENCE'):
        event_evidence(a,predictor())


@pytest.mark.parametrize('mutation',[
    lambda e:e['class_names'].reverse(),
    lambda e:e['audio']['provenance']['artifact_sha256'].update({'weights.pt':'0'*64}),
    lambda e:e['video'].update({'shape':[29,10]}),
    lambda e:e['audio'].update({'private_path':'/models/secret'}),
    lambda e:e['audio']['scores'][0].__setitem__(0,True),
    lambda e:e['video']['scores'].pop(),
    lambda e:e.update({'schema':'APPROVED'}),
])
def test_response_revalidates_event_schema_order_and_provenance(mutation):
    r=result();mutation(r['event_predictions'])
    with pytest.raises(ValueError):response_for_research(r)
    html=research_card(r)
    assert 'Actual raw E3 score' not in html and '/models/secret' not in html


def test_three_distinct_sections_and_explicit_untrained_presence_state():
    r=result();html=research_card(r)
    for name in ('events','temporal','correspondence'):
        assert f'data-head="{name}"' in html
    assert TEMPORAL_UNAVAILABLE in html
    assert r['temporal_predictions'] is None
    assert html.count('<tr>')==11
    assert 'Accordion' in html and 'background' in html and '0.100000' in html
    assert 'Actual raw E3 score' in html and str(r['threshold']) in html
    assert 'not a calibrated confidence percentage' in html
    assert '29-class' in r['event_predictions']['semantics']


def test_temporal_predictions_cannot_be_invented_from_event_evidence():
    r=result();r['temporal_predictions']=[1]*10
    with pytest.raises(ValueError):response_for_research(r)


def test_legacy_no_encoder_values_remains_explicitly_unavailable():
    r=result();r['event_predictions']=None
    html=research_card(r)
    assert 'Encoder event evidence is not available' in html
    assert 'Actual raw E3 score' in html


def test_reset_empty_state_has_no_previous_scores_or_event_labels():
    html=empty_results('Cancelled. Select a new file.')
    assert 'Accordion' not in html and 'Actual raw E3 score' not in html
    assert TEMPORAL_UNAVAILABLE in html and 'No score is shown' in html
    assert event_score(1e-10)=='1.000e-10' and event_score(0.)=='0.000000'


def test_extra_unverified_temporal_or_event_content_is_not_rendered():
    r=result();r['event_predictions']['class_names'][0]='<img src=x onerror=alert(1)>'
    assert '<img' not in research_card(r)


def test_evidence_round_trip_is_independent_of_caller_mutation():
    a=predictor();e=event_evidence(a,predictor());saved=copy.deepcopy(e)
    a.probs.fill(0);a.class_names.clear()
    assert e==saved
    validate_event_evidence(e)
