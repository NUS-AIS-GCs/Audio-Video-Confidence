"""Separately trained per-second head; existing encoders and E3 are untouched."""
import numpy as np
from itertools import combinations
import torch
from torch import nn

VERSION='H2-AVE-presence-v1'


class TemporalPresenceMLP(nn.Module):
    def __init__(self,hidden=64,kernel=1,dropout=.1,context='local'):
        super().__init__()
        if type(hidden) is not int or hidden not in (64,128):raise ValueError('unsupported temporal architecture')
        if kernel not in (1,3) or not 0<=dropout<1 or context not in ('local','gru'):raise ValueError('unsupported temporal context')
        self.hidden,self.kernel,self.dropout,self.context=hidden,kernel,dropout,context
        self.register_buffer('mean',torch.zeros(768,dtype=torch.float32))
        self.register_buffer('scale',torch.ones(768,dtype=torch.float32))
        if context=='gru':
            self.layers=nn.Sequential(nn.Linear(768,hidden),nn.ReLU(),nn.Dropout(dropout))
            self.temporal=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
            self.output=nn.Linear(hidden,1)
        else:self.layers=(nn.Sequential(nn.Linear(768,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1)) if kernel==1
            else nn.Sequential(nn.Conv1d(768,hidden,3,padding=1),nn.ReLU(),nn.Dropout(dropout),nn.Conv1d(hidden,1,1)))

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(10,768) or x.dtype!=torch.float32:
            raise ValueError('expected float32 (batch,10,768) aligned embeddings')
        x=(x-self.mean)/self.scale
        if self.context=='gru':return self.output(self.temporal(self.layers(x))[0]).squeeze(-1)
        return self.layers(x).squeeze(-1) if self.kernel==1 else self.layers(x.transpose(1,2)).squeeze(1)


def classification_metrics(y,scores,threshold):
    y=np.asarray(y,dtype=np.int64).reshape(-1);pred=(np.asarray(scores).reshape(-1)>=threshold).astype(np.int64)
    if len(y)!=len(pred) or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('invalid evaluation targets')
    cm=np.bincount(2*y+pred,minlength=4).reshape(2,2)
    precision=[];recall=[];f1=[]
    for c in (0,1):
        p=float(cm[c,c]/cm[:,c].sum()) if cm[:,c].sum() else 0.
        r=float(cm[c,c]/cm[c].sum()) if cm[c].sum() else 0.
        precision.append(p);recall.append(r);f1.append(2*p*r/(p+r) if p+r else 0.)
    return {'accuracy':float(np.trace(cm)/cm.sum()),'precision':precision[1],'recall':recall[1],
            'macro_precision':float(np.mean(precision)),'macro_recall':float(np.mean(recall)),
            'macro_f1':float(np.mean(f1)),'per_class_precision':precision,'per_class_recall':recall,
            'per_class_f1':f1,'confusion_matrix':cm.tolist(),'seconds':int(len(y)),
            'positive_seconds':int(y.sum()),'background_seconds':int(len(y)-y.sum())}


def boundary_metrics(y,scores,threshold):
    errors=[];missed=0;extra=0;target_count=0;pred_count=0
    for truth,score in zip(y,scores):
        for direction in (-1,1):
            actual=np.flatnonzero(np.diff(truth)==direction)+1
            predicted=np.flatnonzero(np.diff((score>=threshold).astype(np.int64))==direction)+1
            target_count+=len(actual);pred_count+=len(predicted)
            if len(actual) and len(predicted):
                small,large=(actual,predicted) if len(actual)<=len(predicted) else (predicted,actual)
                matching=min(combinations(large,len(small)),key=lambda row:sum(abs(a-b) for a,b in zip(small,row)))
                errors.extend(float(abs(a-b)) for a,b in zip(small,matching))
            missed+=max(0,len(actual)-len(predicted));extra+=max(0,len(predicted)-len(actual))
    return {'definition':'minimum-distance one-to-one same-direction in-window transitions; unmatched boundaries reported separately; 1-second grid',
            'nearest_transition_mae_seconds':float(np.mean(errors)) if errors else None,
            'annotated_boundaries':int(target_count),'predicted_boundaries':int(pred_count),
            'missed_boundary_count':int(missed),'extra_boundary_count':int(extra)}
