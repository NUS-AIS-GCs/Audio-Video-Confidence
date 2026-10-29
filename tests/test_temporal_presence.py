"""Synthetic contract fixtures only; real Head 2 metrics come from AVE training."""
import numpy as np
import pytest
import torch

from fusion.contracts import Record
from fusion.temporal_presence_data import PresenceData,source_exclusions,temporal_targets,swapped_temporal_target
from fusion.temporal_presence_model import TemporalPresenceMLP,classification_metrics,boundary_metrics
from fusion.train_temporal_presence import choose_threshold,accepted


def record(clip,start,end,label=0):return Record(f'{clip}__{label:02d}',clip,'unit event',start,end,label)


def test_union_preserves_real_intervals_and_deduplicates_sources():
    y=temporal_targets([record('one',1,3),record('one',5,7,1),record('empty',0,0)])
    assert set(y)=={'one'}
    assert y['one'].tolist()==[0,1,1,0,0,1,1,0,0,0]


def test_cross_split_source_is_removed_from_every_partition():
    data={'train':[record('shared',0,2),record('train',2,4)],'val':[record('shared',4,6,1)],'test':[]}
    blocked=source_exclusions(data)
    assert blocked=={'shared'}
    assert set(temporal_targets(data['train'],blocked))=={'train'}
    assert temporal_targets(data['val'],blocked)=={}


def test_swaps_have_unknown_joint_ground_truth():
    assert swapped_temporal_target(record('a',0,10),record('b',0,10,1)) is None
    assert swapped_temporal_target(record('a',0,10),record('b',0,10)) is None


def test_test_feature_access_is_closed_before_selection():
    data=object.__new__(PresenceData)
    with pytest.raises(ValueError,match='frozen selection'):data.load('test')


def test_metrics_include_both_classes_and_explicit_confusion_order():
    r=classification_metrics([0,0,1,1],[.1,.8,.7,.2],.5)
    assert r['confusion_matrix']==[[1,1],[1,1]]
    assert r['macro_f1']==r['accuracy']==r['precision']==r['recall']==.5
    assert r['per_class_recall']==[.5,.5]


def test_boundary_errors_match_transition_direction_and_count_missing_edges():
    y=np.array([[0,0,1,1,1,0,0,0,0,0]])
    scores=np.array([[0,0,0,1,1,1,0,0,0,0]],dtype=float)
    r=boundary_metrics(y,scores,.5)
    assert r['nearest_transition_mae_seconds']==1
    assert r['missed_boundary_count']==r['extra_boundary_count']==0
    missing=boundary_metrics(y,np.ones_like(scores),.5)
    assert missing['nearest_transition_mae_seconds'] is None and missing['missed_boundary_count']==2


def test_model_is_per_second_and_uses_separate_normalization_buffers():
    torch.manual_seed(42);model=TemporalPresenceMLP(64).eval()
    x=torch.zeros((2,10,768),dtype=torch.float32)
    assert model(x).shape==(2,10)
    assert set(dict(model.named_buffers()))=={'mean','scale'}
    assert not any('encoder' in name or 'correspondence' in name for name,_ in model.named_parameters())
    with pytest.raises(ValueError):model(x[:,:9])


def test_threshold_is_selected_from_supplied_validation_scores_only():
    metrics,threshold=choose_threshold(np.array([0,0,1,1]),np.array([.1,.2,.3,.4]),[.25,.5])
    assert threshold==.25 and metrics['macro_f1']==1


def test_acceptance_requires_both_class_recalls_not_accuracy_alone():
    protocol={'integration_acceptance':{'validation_macro_f1_min':.7,'validation_each_class_recall_min':.6}}
    assert not accepted({'macro_f1':.9,'per_class_recall':[0.,1.]},protocol,'validation')
    assert accepted({'macro_f1':.8,'per_class_recall':[.7,.9]},protocol,'validation')
