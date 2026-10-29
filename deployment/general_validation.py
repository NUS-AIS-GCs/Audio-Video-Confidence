"""Offline genuine TRAIN-only research audit; no release authorization is written.

Run inside the same isolated container as the research worker. The explicit
acquisition manifest is authenticated before any reference arrays are read.
Generated swaps/remuxes stay in a private temporary directory. Output contains
measurements and identities, never model tensors, credentials or media.
"""
import argparse
import csv
import json
from pathlib import Path
import resource
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from deployment.general_inference import GeneralResearchService, STATUS, response_for_research
from deployment.parity_metrics import numerical_metrics
from deployment.runtime import ReadinessReport, validate_upload
from deployment.train_manifest import PERMITTED_IDS, load_manifest
from deployment.train_video_diagnostics import sample_train_frames
from fusion.encoder_interfaces import Artifact
from fusion.inference import CANDIDATES


def fusion_score(service, audio, video):
    with torch.no_grad():
        z = float(service.fusion._model(torch.from_numpy(audio)[None],
            torch.from_numpy(video)[None], torch.ones(1, 10, dtype=torch.bool)).logits[0, 0])
    return {'logit': z, 'score': float(np.exp(-np.logaddexp(0., -np.float64(z))))}


def fusion_sensitivity(service, audio, video, historical_audio, historical_video):
    """Prespecified ten-pair storage/runtime comparison; never fit a tolerance."""
    threshold = CANDIDATES['E3'].threshold
    rows = []
    for kind, offset in (('original', 0), ('swapped', 1)):
        for i, sid in enumerate(PERMITTED_IDS):
            j = (i+offset) % 5
            variants = {'raw_float32': fusion_score(service, audio[j], video[i]),
                'raw_float16_roundtrip': fusion_score(service,
                    audio[j].astype(np.float16).astype(np.float32), video[i].astype(np.float16).astype(np.float32)),
                'historical_float16_exports': fusion_score(service,
                    historical_audio[j].astype(np.float32), historical_video[i].astype(np.float32))}
            rows.append({'kind':kind,'video':sid,'audio':PERMITTED_IDS[j],'threshold':threshold,
                'variants':variants,'decisions':{n:r['score']>=threshold for n,r in variants.items()},
                'margins':{n:abs(r['score']-threshold) for n,r in variants.items()}})
    return rows


def audit(models, inputs, manifest_sha256, output):
    manifest = load_manifest(inputs/'manifest.json', manifest_sha256, models, inputs)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    report = {'schema': 'aval.general-validation.v1', 'scope': 'five prespecified TRAIN sources only; no accuracy benchmark',
              'independent_reviewer': None, 'independent_signoff': None, 'approved': False,
              'manifest_sha256': manifest_sha256, 'cases': [], 'compatibility': [], 'checks': {}}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='aval-general-audit-') as directory:
        work = Path(directory)
        with patch('torch.load', wraps=torch.load) as loads:
            service = GeneralResearchService(models, work/'derived')
            startup_loads = loads.call_count
            report.update(profile=service.profile, profile_sha256=service.profile_sha256,
                          startup_seconds=time.perf_counter()-start, startup_torch_load_calls=startup_loads)
            sources = [models/'video/parity_samples'/(sid.rsplit('__', 1)[0]+'.mp4') for sid in PERMITTED_IDS]
            originals = []
            canonical_wave = np.load(inputs/'waveforms.npy', allow_pickle=False)
            canonical_frames = np.load(inputs/'frames.npy', allow_pickle=False)
            exports = {}
            for role in ('audio', 'video'):
                root = models/(role+'_train')
                ids = [r['sample_id'] for r in csv.DictReader((root/'train_ids.csv').open())]
                arr = np.load(root/'train_emb.npy', allow_pickle=False)
                exports[role] = np.stack([arr[ids.index(sid)] for sid in PERMITTED_IDS])

            def run(path, label, kind):
                result = service.predict_research(path.read_bytes(), path.name)
                clean = response_for_research(result)
                if result['status'] != STATUS:
                    raise AssertionError((label, clean))
                report['cases'].append({'case': label, 'kind': kind, **clean})
                print(json.dumps({'case': label, 'score': result['score'], 'seconds': result['processing_seconds']}), flush=True)
                output.write_text(json.dumps(report, indent=2, allow_nan=False))
                return result

            for i, (sid, source) in enumerate(zip(PERMITTED_IDS, sources)):
                result = run(source, sid, 'original TRAIN')
                originals.append(result)
                wave, frames, a, v = result['_audit']
                reference_frames, _ = sample_train_frames(Artifact(str(source), manifest['model_files'][str(source.relative_to(models))]))
                # Separately re-execute the producer's audio command on the same
                # compressed bytes (not the canonical Windows WAV acquisition).
                wav = work/'producer.wav'
                service.decoder.run('ffmpeg', ['-nostdin','-v','error','-y','-i',str(source),
                    '-vn','-ac','1','-ar','16000','-t','10','-c:a','pcm_s16le',str(wav)])
                import soundfile as sf
                ref_wave, sr = sf.read(wav, dtype='float32'); assert sr == 16000
                ca = service.audio.predict_waveform(canonical_wave[i]).embeddings
                cv = service.video.predict_frames(canonical_frames[i]).embeddings
                reference = fusion_score(service, ca, cv)
                report['compatibility'].append({'sample': sid,
                    'same_runtime_producer_audio': numerical_metrics(wave, ref_wave),
                    'canonical_audio': numerical_metrics(wave, canonical_wave[i]),
                    'producer_pixels_exact': bool(np.array_equal(frames, reference_frames)),
                    'canonical_pixels_exact': bool(np.array_equal(frames, canonical_frames[i])),
                    'audio_vs_canonical_float32': numerical_metrics(a, ca),
                    'video_vs_canonical_float32': numerical_metrics(v, cv),
                    'audio_vs_historical_float16': numerical_metrics(a, exports['audio'][i]),
                    'video_vs_historical_float16': numerical_metrics(v, exports['video'][i]),
                    'canonical_fusion': reference,
                    'canonical_logit_delta': result['logit']-reference['logit'],
                    'canonical_score_delta': result['score']-reference['score'],
                    'same_canonical_decision': (result['score']>=result['threshold']) == (reference['score']>=result['threshold'])})

            # Prespecified cyclic audio substitution, stream copied without any
            # re-encoding, threshold fitting or selection based on scores.
            for i, sid in enumerate(PERMITTED_IDS):
                swapped = work/f'swap-{i}.mp4'
                service.decoder.run('ffmpeg', ['-nostdin','-v','error','-y','-i',str(sources[i]),
                    '-i',str(sources[(i+1)%5]),'-map','0:v:0','-map','1:a:0','-c','copy',str(swapped)])
                r = run(swapped, sid+' + audio '+PERMITTED_IDS[(i+1)%5], 'audio-swapped TRAIN')
                expected = fusion_score(service, originals[(i+1)%5]['_audit'][2], originals[i]['_audit'][3])
                assert r['logit'] == expected['logit'], 'stream-copy swap did not preserve features'
                report['cases'][-1]['paired_feature_logit_exact'] = True
                report['cases'][-1]['threshold_margin'] = abs(r['score']-r['threshold'])

            remux = work/'new-unapproved.mp4'
            service.decoder.run('ffmpeg', ['-nostdin','-v','error','-y','-i',str(sources[0]),
                '-map','0','-c','copy','-metadata','comment=unapproved general-path engineering remux',str(remux)])
            new = run(remux, 'new bytes / metadata remux', 'new admitted unapproved bytes; same TRAIN content')
            assert new['source_sha256'] not in {r['source_sha256'] for r in originals}
            assert new['logit'] == originals[0]['logit']
            repeat = run(sources[0], 'repeat first original', 'consistency')
            assert repeat['logit'] == originals[0]['logit']
            assert all(np.array_equal(a,b) for a,b in zip(repeat['_audit'],originals[0]['_audit']))
            service._lock.acquire()
            try: assert service.predict_research(b'', 'busy.mp4')['error_code'] == 'RESOURCE_BUSY'
            finally: service._lock.release()
            report['checks']['concurrency_rejected_without_decode'] = True
            invalid = [('corrupt.mp4',b'corrupt'),('bad.avi',b'bad')]
            for label, args in (
                ('missing-audio', ['-an','-c:v','copy']),
                ('missing-video', ['-vn','-c:a','copy']),
                ('short', ['-t','9','-c','copy']),
                ('shifted-clock', ['-c','copy','-output_ts_offset','0.1'])):
                path = work/(label+'.mp4')
                service.decoder.run('ffmpeg',['-nostdin','-v','error','-y','-i',str(sources[0]),*args,str(path)])
                invalid.append((path.name,path.read_bytes()))
            invalid.append(('oversize.mp4',bytes(service.config.max_upload_bytes+1)))
            for name, data in invalid:
                r = service.predict_research(data,name)
                assert r['status']=='REJECTED' and r['predictions'] is None, (name,r)
                report['checks'][name] = r['error_code']
            public = validate_upload(remux.read_bytes(),filename=remux.name,config=service.config,
                readiness=ReadinessReport('MODEL_NOT_READY',(),False))
            assert public['status']=='MODEL_NOT_READY' and public['predictions'] is None
            report['checks']['public_even_after_real_models_loaded'] = public
            assert loads.call_count == startup_loads
            report['checks']['no_request_torch_load_calls'] = True
            report['checks']['repeat_arrays_and_logit_exact'] = True
            report['fusion_sensitivity'] = fusion_sensitivity(service,
                [r['_audit'][2] for r in originals], [r['_audit'][3] for r in originals],
                exports['audio'], exports['video'])
            report['checks']['no_decoded_request_temporaries'] = not list(Path('/tmp').glob('aval-general-*')) or all(
                p == work for p in Path('/tmp').glob('aval-general-*'))
            assert report['checks']['no_decoded_request_temporaries']
            service._integrity()
    load_manifest(inputs/'manifest.json',manifest_sha256,models,inputs)
    report['checks']['all_40_model_and_5_input_hashes_unchanged'] = True
    report['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['total_seconds'] = time.perf_counter()-start
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    audit(args.models,args.inputs,args.manifest_sha256,args.output)
