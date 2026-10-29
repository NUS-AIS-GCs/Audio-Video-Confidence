"""Train only Head 2 from authenticated frozen exports; TEST opens after selection.

No encoder construction/training, tracking upload, synthetic target or public
release authorization. Checkpoints and run evidence must stay Git-ignored.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from fusion.temporal_presence_data import PresenceData,sha256
from fusion.temporal_presence_model import (VERSION,TemporalPresenceMLP,
    classification_metrics,boundary_metrics)


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def scores_for(model,x,batch=128):
    model.eval();chunks=[]
    with torch.no_grad(),torch.autocast('cpu',enabled=False):
        for i in range(0,len(x),batch):chunks.append(model(torch.from_numpy(x[i:i+batch])).numpy())
    logits=np.concatenate(chunks)
    if logits.shape!=(len(x),10) or not np.isfinite(logits).all():raise ValueError('invalid trained temporal output')
    return logits,np.exp(-np.logaddexp(0.,-logits.astype(np.float64)))


def choose_threshold(y,scores,grid,min_recall=None):
    choices=[(classification_metrics(y,scores,t),t) for t in grid]
    eligible=[row for row in choices if min(row[0]['per_class_recall'])>=min_recall] if min_recall is not None else choices
    choices=eligible or choices
    return max(choices,key=lambda row:(row[0]['macro_f1'],-abs(row[1]-.5),-row[1]))


def accepted(metrics,protocol,split):
    criteria=protocol['integration_acceptance']
    return (metrics['macro_f1']>=criteria[split+'_macro_f1_min'] and
            min(metrics['per_class_recall'])>=criteria[split+'_each_class_recall_min'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',default='_local/embeddings/audio_final')
    p.add_argument('--video',default='_local/embeddings/vision_final/resnet18_v1')
    p.add_argument('--metadata',default='_local/datasets/AVE')
    p.add_argument('--reference',default='_local/datasets/AVE_reference')
    p.add_argument('--e3-metadata',default='_local/phase8_stage2c/models/E3/checkpoint_metadata.json')
    p.add_argument('--protocol',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();out=args.output.resolve();repo=Path.cwd().resolve()
    if out.exists() or not out.is_relative_to(repo):raise ValueError('fresh repository-local ignored output required')
    if subprocess.run(['git','check-ignore','--quiet','--no-index',str(out)],check=False).returncode:
        raise ValueError('private training output must be Git-ignored')
    protocol=json.loads(args.protocol.read_text());cfg=protocol['search']
    if protocol['version']!=VERSION or cfg['hidden_dims']!=[64,128] or cfg['epochs']!=40:
        raise ValueError('unexpected preregistered protocol')
    out.mkdir(parents=True);write(out/'protocol.json',protocol)
    protocol_digest=sha256(out/'protocol.json')
    source={n:sha256(Path(__file__).with_name(n)) for n in
        ('temporal_presence_data.py','temporal_presence_model.py','train_temporal_presence.py')}
    write(out/'source-sha256.json',source)
    (out/'source').mkdir()
    for name,digest in source.items():
        payload=Path(__file__).with_name(name).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('source changed before training')
        (out/'source'/name).write_bytes(payload)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data=PresenceData(args.audio,args.video,args.metadata,args.reference,args.e3_metadata)
    train,yt,train_ids=data.load('train');val,yv,val_ids=data.load('val')
    if set(train_ids)&set(val_ids):raise ValueError('source leakage')
    mean=train.astype(np.float64).mean(axis=(0,1)).astype(np.float32)
    scale=np.maximum(train.astype(np.float64).std(axis=(0,1)),1e-6).astype(np.float32)
    pos_weight=float((yt.size-yt.sum())/yt.sum())
    if not 0<pos_weight<float('inf'):raise ValueError('both supervised classes required')
    inventory={'source_policy':protocol['split_policy'],'excluded_shared_sources':sorted(data.excluded),
               'train':{'clips':len(train_ids),'seconds':int(yt.size),'positive':int(yt.sum()),'background':int(yt.size-yt.sum())},
               'val':{'clips':len(val_ids),'seconds':int(yv.size),'positive':int(yv.sum()),'background':int(yv.size-yv.sum())},
               'source_ids':{'train':train_ids,'val':val_ids},'inputs':dict(data.inputs),
               'test_embeddings_opened':False,'source_reference':data.reference}
    write(out/'training-inputs.json',inventory)
    xt=torch.from_numpy(train);targets=torch.from_numpy(yt)
    history=[];best=None;started=time.monotonic()
    variants=cfg.get('variants',[{'hidden':h,'kernel':1,'dropout':cfg['dropout'],'weight_decay':cfg['weight_decay']} for h in cfg['hidden_dims']])
    for variant in variants:
        hidden=variant['hidden'];kernel=variant['kernel'];dropout=variant['dropout'];context=variant.get('context','local')
        torch.manual_seed(cfg['seed']);model=TemporalPresenceMLP(hidden,kernel,dropout,context)
        model.mean.copy_(torch.from_numpy(mean));model.scale.copy_(torch.from_numpy(scale))
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=variant['weight_decay'])
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        generator=torch.Generator().manual_seed(cfg['seed'])
        for epoch in range(cfg['epochs']):
            model.train();order=torch.randperm(len(train),generator=generator);losses=[]
            for indices in order.split(cfg['batch_clips']):
                optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(xt[indices]),targets[indices])
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss.detach()))
            _,sv=scores_for(model,val);metrics,threshold=choose_threshold(yv,sv,cfg['threshold_grid'],
                protocol['integration_acceptance']['validation_each_class_recall_min'] if 'variants' in cfg else None)
            key=(accepted(metrics,protocol,'validation'),metrics['macro_f1'],-abs(threshold-.5),-epoch,-hidden)
            row={'hidden':hidden,'kernel':kernel,'dropout':dropout,'context':context,'epoch':epoch+1,'loss':float(np.mean(losses)),'threshold':threshold,'validation':metrics}
            history.append(row)
            if best is None or key>best['key']:
                best={'key':key,'state':copy.deepcopy(model.state_dict()),'hidden':hidden,'kernel':kernel,'dropout':dropout,'context':context,'epoch':epoch+1,
                      'threshold':threshold,'validation':metrics}
            if (epoch+1)%10==0:print(json.dumps(row),flush=True)
    write(out/'history.json',history)
    data.verify_inputs_unchanged()
    model=TemporalPresenceMLP(best['hidden'],best['kernel'],best['dropout'],best['context']);model.load_state_dict(best['state'],strict=True);model.eval().requires_grad_(False)
    torch.save({'schema':'gc2-head2-state-v1','hidden':best['hidden'],'kernel':best['kernel'],'dropout':best['dropout'],'context':best['context'],'state_dict':model.state_dict()},out/'weights.pt')
    frozen={'schema':'gc2-head2-frozen-selection-v1','version':VERSION,'hidden':best['hidden'],'kernel':best['kernel'],'dropout':best['dropout'],'context':best['context'],'epoch':best['epoch'],
            'threshold':best['threshold'],'validation':best['validation'],'checkpoint_sha256':sha256(out/'weights.pt'),
            'protocol_sha256':protocol_digest,'source_sha256':source,'inputs':dict(data.inputs)}
    write(out/'selection.json',frozen)
    if not accepted(best['validation'],protocol,'validation'):
        write(out/'rejected.json',{'reason':'VALIDATION_ACCEPTANCE_NOT_MET','best_validation':best['validation'],'checkpoint_sha256':frozen['checkpoint_sha256']})
        raise SystemExit('Head 2 validation criteria failed; rejected checkpoint retained; TEST was not opened.')
    selection_digest=sha256(out/'selection.json')
    # Selection is durable and hashed before the first TEST feature load.
    with (out/'test-opened-once.json').open('x') as stream:
        json.dump({'selection_sha256':selection_digest,'checkpoint_sha256':frozen['checkpoint_sha256']},stream)
    test,ye,test_ids=data.load('test',test_selection=selection_digest)
    if set(test_ids)&(set(train_ids)|set(val_ids)):raise ValueError('TEST source leakage')
    test_logits,test_scores=scores_for(model,test)
    test_metrics=classification_metrics(ye,test_scores,best['threshold'])
    results={}
    for split,x,y in (('train',train,yt),('val',val,yv),('test',test,ye)):
        scores=test_scores if split=='test' else scores_for(model,x)[1]
        results[split]={**classification_metrics(y,scores,best['threshold']),
                       'boundaries':boundary_metrics(y,scores,best['threshold']),
                       'majority_baseline':classification_metrics(y,np.ones_like(y),.5)}
    data.verify_inputs_unchanged()
    if sha256(out/'weights.pt')!=frozen['checkpoint_sha256'] or sha256(out/'selection.json')!=selection_digest:
        raise ValueError('selection or checkpoint changed during TEST evaluation')
    passed=accepted(test_metrics,protocol,'test')
    report={'schema':'gc2-head2-evaluation-v1','version':VERSION,'selection_sha256':selection_digest,
            'checkpoint_sha256':frozen['checkpoint_sha256'],'threshold':best['threshold'],'metrics':results,
            'inputs':dict(data.inputs),'source_ids':{'train':train_ids,'val':val_ids,'test':test_ids},
            'integration_criteria_passed':passed,'elapsed_seconds':time.monotonic()-started,
            'test_used_for_selection':False,'swapped_pair_targets':'UNKNOWN_NOT_EVALUATED',
            'ground_truth_scope':protocol['task'],'limitations':protocol['scope_limitations']}
    write(out/'evaluation.json',report)
    # Prespecified first TRAIN clip with both annotated background and event bins.
    example_index=next(i for i,y in enumerate(yt) if y.min()==0 and y.max()==1)
    logits,prob=scores_for(model,train[example_index:example_index+1])
    write(out/'example.json',{'clip_id':train_ids[example_index],'split':'train','labels':yt[example_index].astype(int).tolist(),
        'scores':prob[0].tolist(),'predictions':(prob[0]>=best['threshold']).astype(int).tolist(),
        'positive_bin':int(np.flatnonzero(yt[example_index]==1)[0]),'negative_bin':int(np.flatnonzero(yt[example_index]==0)[0]),
        'note':'functional TRAIN example; not held-out performance'})
    metadata={'schema':'gc2-head2-metadata-v1','version':VERSION,'hidden':best['hidden'],'kernel':best['kernel'],'dropout':best['dropout'],'context':best['context'],'threshold':best['threshold'],
              'input_shape':[10,768],'audio_dim':256,'video_dim':512,'output_shape':[10],
              'score_semantics':'uncalibrated sigmoid of a trained binary logit for annotated AVE joint presence',
              'checkpoint_sha256':frozen['checkpoint_sha256'],'selection_sha256':selection_digest,
              'protocol_sha256':protocol_digest,'evaluation_sha256':sha256(out/'evaluation.json'),
              'integration_criteria_passed':passed,'public_release_approved':False,
              'training_runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__},
              'normalization':'TRAIN-only mean/std buffers; scale lower bound 1e-6',
              'scope':protocol['task'],'limitations':protocol['scope_limitations']}
    write(out/'metadata.json',metadata)
    print(json.dumps({'version':VERSION,'criteria_passed':passed,'hidden':best['hidden'],'epoch':best['epoch'],
                      'threshold':best['threshold'],'metrics':results,'output':str(out)}),flush=True)
    if not passed:raise SystemExit('TEST acceptance failed; no retuning or deployment is authorized.')


if __name__=='__main__':main()
