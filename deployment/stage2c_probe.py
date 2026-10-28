"""Reproducible offline encoder audit. Never launches a server or admits uploads."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from deployment.audio_bundle import VerifiedAudioPredictor
from deployment.video_bundle import VerifiedVideoPredictor
from deployment.paired_diagnostics import InternalTrainPairStudy
from deployment.parity_metrics import numerical_metrics, float16_cell_analysis
from deployment.runtime import PINNED_VERSIONS
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact


from deployment.train_manifest import (
    AUDIO_CARD, VIDEO_CARD, PERMITTED_IDS, REQUIRED_MODEL_FILES, REQUIRED_INPUT_FILES,
    load_manifest, pinned_child,
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_usage():
    result = {'process_cpu_seconds': time.process_time(), 'wall_clock': time.perf_counter()}
    try:
        import resource
        result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)
    except ImportError:
        result['process_peak_rss_bytes'] = None
    for name in ('memory.current', 'memory.peak', 'memory.max', 'cpu.stat', 'cpu.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            result['cgroup_' + name] = path.read_text().strip()
    return result


def timed(call):
    wall, cpu = time.perf_counter(), time.process_time()
    result = call()
    seconds, cpu_seconds = time.perf_counter()-wall, time.process_time()-cpu
    return result, {'wall_seconds': seconds, 'process_cpu_seconds': cpu_seconds,
                    'process_cpu_percent_one_core_basis': 100*cpu_seconds/seconds}


def normalize_video(frames):
    value = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float() / 255.
    return (value-torch.tensor([.485,.456,.406]).view(1,1,3,1,1)) / torch.tensor([.229,.224,.225]).view(1,1,3,1,1)


def describe_outputs(audio, visual, ap, vp, refs, reference_probs, sample_ids):
    result = {}
    for name, value, probabilities, reference, rp in zip(
            ('audio','video'), (audio,visual), (ap,vp), refs, reference_probs):
        result[name] = {'embeddings': float16_cell_analysis(value, reference),
            'probabilities': numerical_metrics(probabilities, rp),
            'same_second_labels': bool(np.array_equal(probabilities.argmax(-1), rp.argmax(-1))),
            'per_clip': [{'sample_id': sid, 'embeddings': numerical_metrics(value[i], reference[i]),
                          'probabilities': numerical_metrics(probabilities[i], rp[i]),
                          'mismatches_by_second': (value[i].astype(np.float16) != reference[i]).sum(1).tolist(),
                          'absolute_error_quantiles': np.quantile(np.abs(value[i].astype(np.float64)-reference[i]), [0,.5,.9,.99,1]).tolist()}
                         for i,sid in enumerate(sample_ids)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--interop-threads', type=int, default=1)
    parser.add_argument('--variants', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= 32 or not 1 <= args.interop_threads <= 32:
        parser.error('threads must be in 1..32')
    args.output.mkdir(parents=True, exist_ok=False)
    start = resource_usage()
    meta = load_manifest(args.inputs/'manifest.json', args.manifest_sha256, args.models, args.inputs)
    versions = {n: importlib.metadata.version(n) for n in PINNED_VERSIONS}
    if versions != PINNED_VERSIONS:
        raise ContractError('runtime versions differ from candidate deployment pins')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    audio = VerifiedAudioPredictor(Artifact(str(args.models/'audio/model_card.json'), AUDIO_CARD), args.output/'derived')
    video = VerifiedVideoPredictor(Artifact(str(args.models/'video/model_card.json'), VIDEO_CARD))
    executable = args.ffmpeg or Path(shutil.which('ffmpeg') or '')
    if not executable.is_file():
        raise ContractError('explicit FFmpeg executable unavailable')
    sources = {sid: Artifact(str(args.models/'video/parity_samples'/f"{sid.rsplit('__',1)[0]}.mp4"),
                            meta['model_files'][f"video/parity_samples/{sid.rsplit('__',1)[0]}.mp4"])
               for sid in PERMITTED_IDS}
    study = InternalTrainPairStudy(audio,video,{n:args.models/n for n in ('E3','A2')},
        args.models/'audio_train',args.models/'video_train',sources,Artifact(str(executable),sha(executable)))
    refs = [b.splits['train'].embeddings[[look[s] for s in PERMITTED_IDS]]
            for b,look in zip((study.audio,study.video),study.lookups)]
    rprobs = [b.splits['train'].probabilities[[look[s] for s in PERMITTED_IDS]]
              for b,look in zip((study.audio,study.video),study.lookups)]
    waves = np.load(args.inputs/'waveforms.npy',allow_pickle=False)
    frames = np.load(args.inputs/'frames.npy',allow_pickle=False)
    if waves.shape != (5,160000) or waves.dtype != np.float32 or frames.shape != (5,10,224,224,3) or frames.dtype != np.uint8:
        raise ContractError('canonical input geometry differs')
    raw_audio, raw_video, raw_ap, raw_vp, evidence = [],[],[],[],[]
    for i,sid in enumerate(PERMITTED_IDS):
        (a,v,ev), timing = timed(lambda:study.encode(sid))
        ev.update(sample_id=sid,latency=timing,
                  waveform_matches_host_canonical=ev['canonical_waveform_sha256']==hashlib.sha256(waves[i].tobytes()).hexdigest(),
                  selected_frames_match_host_canonical=all(f['rgb_sha256']==hashlib.sha256(frames[i,j].tobytes()).hexdigest()
                                                           for j,f in enumerate(ev['video']['frames'])))
        raw_audio.append(a.embeddings);raw_video.append(v.embeddings);raw_ap.append(a.probs);raw_vp.append(v.probs);evidence.append(ev)
        print(sid+' genuine raw execution complete',flush=True)
    raw = tuple(np.stack(v) for v in (raw_audio,raw_video,raw_ap,raw_vp))

    def canonical():
        ae,ve,ap,vp,latency=[],[],[],[],[]
        for i,sid in enumerate(PERMITTED_IDS):
            a,at=timed(lambda:audio.predict_waveform(waves[i]))
            v,vt=timed(lambda:video.predict_frames(frames[i]))
            ae.append(a.embeddings);ve.append(v.embeddings);ap.append(a.probs);vp.append(v.probs)
            latency.append({'sample_id':sid,'audio':at,'video':vt})
        return tuple(np.stack(x) for x in (ae,ve,ap,vp)),latency

    canon, canonical_latency=canonical()
    report={'schema':'aval.stage2c-encoder-audit.v1','status':'EXPERIMENTAL_INTERNAL_TRAIN_ONLY',
        'production_predictions_enabled':False,'strict_gate_changed':False,'candidate_policy_active':False,
        'manifest_sha256':args.manifest_sha256,'sample_ids':list(PERMITTED_IDS),
        'runtime':{'python':sys.version,'platform':platform.platform(),'versions':versions,
                   'threads':args.threads,'interop_threads':torch.get_num_interop_threads(),
                   'mkldnn_enabled':torch.backends.mkldnn.enabled,'torch_build':torch.__config__.show()},
        'decoder':{'path':str(executable),'sha256':sha(executable),
                   'version':subprocess.run([str(executable),'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]},
        'clips':evidence,'canonical_latency':canonical_latency,
        'raw_to_export':describe_outputs(*raw,refs,rprobs,PERMITTED_IDS),
        'canonical_to_export':describe_outputs(*canon,refs,rprobs,PERMITTED_IDS),
        'raw_vs_canonical':{n:numerical_metrics(a,b) for n,a,b in zip(('audio','video','audio_probs','video_probs'),raw,canon)},
        'raw_fusion':study.compare(PERMITTED_IDS,*raw[:2]),
        'canonical_fusion':study.compare(PERMITTED_IDS,*canon[:2]),
        'host_stage2b_comparison':{n:numerical_metrics(a,np.load(args.inputs/(n+'_stage2b.npy'),allow_pickle=False))
                                  for n,a in zip(('audio','video'),canon[:2])}}
    report['normalization']={'audio_mean_exact':bool(torch.equal(audio.model.mean,torch.tensor(json.loads((args.models/'audio/norm_stats.json').read_text())['mean']))),
                            'audio_std_exact':bool(torch.equal(audio.model.std,torch.tensor(json.loads((args.models/'audio/norm_stats.json').read_text())['std'])))}
    normalized=normalize_video(frames)
    np_norm=(frames.astype(np.float32)/255.).transpose(0,1,4,2,3)
    np_norm=(np_norm-np.array([.485,.456,.406],np.float32).reshape(1,1,3,1,1))/np.array([.229,.224,.225],np.float32).reshape(1,1,3,1,1)
    report['normalization']['video_numpy_vs_torch']=numerical_metrics(normalized.numpy(),np_norm)
    report['float16_conversion_numpy_vs_torch']={n:bool(np.array_equal(a.astype(np.float16),torch.from_numpy(a).half().numpy()))
                                                 for n,a in zip(('audio','video'),canon[:2])}
    for scope,values in (('raw',raw),('canonical',canon)):
        for name,value in zip(('audio','video','audio_probs','video_probs'),values):
            np.save(args.output/(scope+'_'+name+'.npy'),value,allow_pickle=False)
    if args.variants:
        variants={}
        for threads,mkldnn,name in ((20,True,'threads20'),(1,True,'threads1'),(4,False,'mkldnn_off')):
            torch.set_num_threads(threads);torch.backends.mkldnn.enabled=mkldnn
            values,latency=canonical()
            variants[name]={'threads':threads,'mkldnn':mkldnn,'latency':latency,
                'to_export':describe_outputs(*values,refs,rprobs,PERMITTED_IDS),
                'vs_canonical':{n:numerical_metrics(a,b) for n,a,b in zip(('audio','video'),values[:2],canon[:2])},
                'fusion':study.compare(PERMITTED_IDS,*values[:2])}
            if name=='threads20':
                variants[name]['vs_audio_stage1']=numerical_metrics(values[0],np.load(args.inputs/'audio_stage1.npy',allow_pickle=False))
            for n,v in zip(('audio','video'),values[:2]):np.save(args.output/(name+'_'+n+'.npy'),v,allow_pickle=False)
            print(name+' controlled numerical variant complete',flush=True)
        torch.set_num_threads(args.threads);torch.backends.mkldnn.enabled=True
        with torch.no_grad(),torch.autocast('cpu',enabled=False):
            ae,ap=[],[]
            for i in range(0,5,2):
                logits,embedding=audio.model(torch.from_numpy(waves[i:i+2]))
                ae.append(embedding.numpy());ap.append(torch.softmax(logits,-1).numpy())
            logits,embedding=video.model(normalized.contiguous())
            values=(np.concatenate(ae),embedding.numpy(),np.concatenate(ap),torch.softmax(logits,-1).numpy())
            variants['batch2_audio_batch5_video']={'to_export':describe_outputs(*values,refs,rprobs,PERMITTED_IDS),
                'vs_canonical':{n:numerical_metrics(a,b) for n,a,b in zip(('audio','video'),values[:2],canon[:2])},
                'fusion':study.compare(PERMITTED_IDS,*values[:2])}
            logits,embedding=video.model(normalized[:1].contiguous())
            variants['video_contiguous_clip0']={'vs_canonical':numerical_metrics(embedding.numpy(),canon[1][:1])}
        audio.encoder.require_ready();video.encoder.require_ready()
        report['variants']=variants
    load_manifest(args.inputs/'manifest.json',args.manifest_sha256,args.models,args.inputs)
    finish=resource_usage()
    report['resources']={'start':start,'finish':finish,'elapsed_seconds':finish['wall_clock']-start['wall_clock'],
        'process_cpu_seconds':finish['process_cpu_seconds']-start['process_cpu_seconds']}
    (args.output/'measurements.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['resources']['elapsed_seconds'],
                      'audio_mismatches':report['raw_to_export']['audio']['embeddings']['float16_differing_values'],
                      'video_mismatches':report['raw_to_export']['video']['embeddings']['float16_differing_values']}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
