"""Genuine cold/warm profiling for the isolated internal TRAIN backend."""
import argparse
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from deployment.internal_backend import InternalTrainBackend, response_for_review
from deployment.parity_metrics import numerical_metrics
from deployment.stage2c_probe import PERMITTED_IDS, resource_usage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--compare-inference-mode',action='store_true')
    args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    process_start=resource_usage();start=time.perf_counter()
    calls=[];original_load=torch.load
    def restricted(*a,**kw):
        if kw.get('weights_only') is not True:raise RuntimeError('unrestricted checkpoint load attempted')
        calls.append(str(a[0]));return original_load(*a,**kw)
    with patch('torch.load',side_effect=restricted):
        backend=InternalTrainBackend(args.models,args.inputs,args.manifest_sha256,
            shutil.which('ffmpeg'),shutil.which('ffprobe'),args.output/'derived')
        initialization_seconds=time.perf_counter()-start
        load_count=len(calls);identities=tuple(id(model) for model in backend._models)
        raw_reference={n:np.load(args.baseline/('raw_'+n+'.npy'),allow_pickle=False) for n in ('audio','video')}
        records=[];features={};requests=[]
        passes=['first_pass','warm_repeat']+(['inference_mode'] if args.compare_inference_mode else [])
        for mode in passes:
            for i,sid in enumerate(PERMITTED_IDS):
                filename=sid.rsplit('__',1)[0]+'.mp4'
                source=args.models/'video/parity_samples'/filename
                content=source.read_bytes()
                boundary_started=time.perf_counter()
                result=backend.predict_internal(sid,content,filename,inference_mode=mode=='inference_mode')
                boundary_seconds=time.perf_counter()-boundary_started
                if result.get('status')!='EXPERIMENTAL_INTERNAL_TRAIN_ONLY':
                    (args.output/'failed-response.json').write_text(json.dumps(response_for_review(result),indent=2))
                    raise RuntimeError('genuine internal request rejected; inspect failed-response.json')
                actual=result.pop('_audit_features')
                serialization_started=time.perf_counter()
                visible=response_for_review(result)
                json.dumps(visible)
                serialization_seconds=time.perf_counter()-serialization_started
                record={'pass':mode,'sample_id':sid,'response':visible,
                    'request_boundary_wall_seconds':boundary_seconds,
                    'response_validation_and_json_seconds':serialization_seconds,
                    'same_model_objects':tuple(id(model) for model in backend._models)==identities,
                    'torch_load_calls_unchanged':len(calls)==load_count,
                    'vs_stage2c_linux_raw':{n:numerical_metrics(x,raw_reference[n][i:i+1])
                                           for n,x in zip(('audio','video'),actual)}}
                if mode=='first_pass':
                    features[sid]=tuple(x.copy() for x in actual)
                    requests.append(result['predictions'])
                else:
                    record['vs_first_pass']={n:numerical_metrics(x,y) for n,x,y in zip(('audio','video'),actual,features[sid])}
                    record['score_differences_vs_first_pass']={n:result['predictions'][n]['score']-requests[i][n]['score'] for n in result['predictions']}
                for n,x in zip(('audio','video'),actual):np.save(args.output/(mode+'_'+sid+'_'+n+'.npy'),x,allow_pickle=False)
                records.append(record)
                print(mode,sid,round(result['timings']['total']['wall_seconds'],4),'seconds',flush=True)
        # Optional A2 comparison never changes the mandatory frozen E3 result.
        sid=PERMITTED_IDS[0];filename=sid.rsplit('__',1)[0]+'.mp4'
        e3_only=backend.predict_internal(sid,(args.models/'video/parity_samples'/filename).read_bytes(),filename,include_a2=False)
        if set(e3_only.get('predictions',{}))!={'E3'} or e3_only['predictions']['E3']!=requests[0]['E3']:
            raise RuntimeError('optional A2 altered E3 output')
        if len(calls)!=load_count or identities!=tuple(id(x) for x in backend._models):
            raise RuntimeError('models were reloaded during requests')
    backend._integrity()
    finish=resource_usage()
    report={'schema':'aval.stage2d-runtime-profile.v1','status':'EXPERIMENTAL_INTERNAL_TRAIN_ONLY',
        'inside_docker':True,'storage_profile':args.profile,'production_predictions_enabled':False,
        'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'initialization_seconds':initialization_seconds,'initialization_stages':backend.initialization.values,
        'restricted_checkpoint_load_calls':calls,'checkpoint_load_calls_after_all_requests':len(calls),
        'load_counts':backend.load_counts,'requests':records,'e3_only_response':response_for_review(e3_only),
        'readiness':backend.readiness(),'resources':{'start':process_start,'finish':finish,
        'wall_seconds':finish['wall_clock']-process_start['wall_clock'],
        'process_cpu_seconds':finish['process_cpu_seconds']-process_start['process_cpu_seconds']}}
    ae=np.concatenate([features[s][0] for s in PERMITTED_IDS]);ve=np.concatenate([features[s][1] for s in PERMITTED_IDS])
    report['frozen_export_sensitivity']=backend.study.compare(PERMITTED_IDS,ae,ve)
    for n,x,b,lookup in (('audio',ae,backend.study.audio,backend.study.lookups[0]),('video',ve,backend.study.video,backend.study.lookups[1])):
        report[n+'_strict_export_comparison']=numerical_metrics(x,b.splits['train'].embeddings[[lookup[s] for s in PERMITTED_IDS]])
    # Keep audit arrays entirely separate from the JSON response contract.
    report['e3_only_response'].pop('_audit_features',None)
    (args.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'initialization_seconds':initialization_seconds,'request_count':len(records)+1,
        'loads':len(calls),'peak_rss_bytes':finish.get('process_peak_rss_bytes')}),flush=True)


if __name__=='__main__':main()
