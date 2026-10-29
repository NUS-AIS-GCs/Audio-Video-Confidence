"""Real private Head 2 checkpoint checks; explicitly skip if unprovisioned."""
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from deployment.temporal_presence import TemporalPresencePredictor,validate_prediction,verify,provision,catalog,canonical_sigmoid
from deployment.general_inference import ResearchPrediction,response_for_research
from deployment.ui.general_research_section import research_card
from fusion.inference import CANDIDATES

ROOT=Path('_local/temporal_presence/run-gru-v1')


def test_cross_platform_sigmoid_uses_one_exact_reference_without_relaxing_thresholds():
    # Actual Linux/Windows libm results straddled this last float64 digit.
    value=canonical_sigmoid(0.2688511610031128)
    assert value==0.5668108449567416
    from decimal import Decimal,localcontext,ROUND_HALF_EVEN
    with localcontext() as context:
        context.prec=50;context.rounding=ROUND_HALF_EVEN
        expected=float(Decimal(1)/(Decimal(1)+(-Decimal.from_float(0.2688511610031128)).exp()))
    assert value==expected
    assert canonical_sigmoid(0.)==.5 and canonical_sigmoid(-1000.)==0. and canonical_sigmoid(1000.)==1.


@pytest.fixture(scope='module')
def genuine():
    if not (ROOT/'weights.pt').is_file():pytest.skip('Private trained Head 2 checkpoint not provisioned')
    p=TemporalPresencePredictor(ROOT)
    example=json.loads((ROOT/'example.json').read_text())
    values=[]
    for root in (Path('_local/embeddings/audio_final'),Path('_local/embeddings/vision_final/resnet18_v1')):
        with (root/'train_ids.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        candidates=[(r['sample_id'],i) for i,r in enumerate(rows) if r['clip_id']==example['clip_id']]
        index=min(candidates)[1];array=np.load(root/'train_emb.npy',mmap_mode='r',allow_pickle=False)
        values.append(np.array(array[index],dtype=np.float32));array._mmap.close()
    return p,values,example


def test_genuine_private_checkpoint_produces_ten_supervised_scores(genuine):
    predictor,values,example=genuine
    result=predictor.predict(*values);validate_prediction(result)
    assert len(result['scores'])==len(result['logits'])==len(result['presence'])==10
    assert np.allclose(result['scores'],example['scores'],rtol=0,atol=1e-6)
    assert result['presence']==[bool(v) for v in example['predictions']]
    assert result['threshold']==.25 and CANDIDATES['E3'].threshold==0.009579099009671126
    assert not result['production_predictions_enabled']


@pytest.mark.parametrize('mutation',[
    lambda r:r['scores'].__setitem__(0,float('nan')),
    lambda r:r['scores'].__setitem__(0,.5),
    lambda r:r['presence'].__setitem__(0,not r['presence'][0]),
    lambda r:r.update(threshold=.5),
    lambda r:r.update(checkpoint_sha256='0'*64),
    lambda r:r.update(production_predictions_enabled=True),
    lambda r:r.update(private_path='/models/Head2/weights.pt'),
])
def test_tampered_temporal_result_is_rejected(genuine,mutation):
    p,values,_=genuine;result=p.predict(*values);mutation(result)
    with pytest.raises(ValueError):validate_prediction(result)


def test_head2_scores_reach_all_three_validated_result_sections(genuine):
    p,values,_=genuine;temporal=p.predict(*values)
    r=ResearchPrediction('a'*32,'b'*64,'c'*64,.5,0.,1.,temporal_predictions=temporal).to_dict()
    r['diagnostics']={};response_for_research(r);html=research_card(r)
    assert html.count('class="av-presence-cell"')==10
    assert 'H2-AVE-presence-v1' in html and 'Head 2 threshold: 0.25' in html
    assert 'verified artifact not loaded' not in html and r['threshold']==CANDIDATES['E3'].threshold


def test_artifact_corruption_is_rejected_before_model_loading(genuine,tmp_path,monkeypatch):
    root=tmp_path/'Head2';root.mkdir()
    for name in ('weights.pt','metadata.json'):shutil.copyfile(ROOT/name,root/name)
    with (root/'weights.pt').open('r+b') as stream:stream.seek(100);stream.write(b'CHANGED')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('must verify hash before checkpoint load'))
    with pytest.raises(ValueError):TemporalPresencePredictor(root)


def test_private_provisioning_is_atomic_and_does_not_overwrite_bad_destinations(genuine,tmp_path):
    dest=tmp_path/'Head2';provision(ROOT,dest);verify(dest)
    original=(dest/'weights.pt').read_bytes()
    (dest/'metadata.json').write_text('{}')
    with pytest.raises(ValueError):provision(ROOT,dest)
    assert (dest/'weights.pt').read_bytes()==original


def test_loaded_head2_weight_mutation_fails_closed(genuine):
    p,values,_=genuine;parameter=next(p.model.parameters());original=parameter.detach().clone()
    try:
        with torch.no_grad():parameter.add_(.01)
        with pytest.raises(ValueError,match='frozen Head 2'):p.predict(*values)
    finally:
        with torch.no_grad():parameter.copy_(original)
    p.require_ready()
