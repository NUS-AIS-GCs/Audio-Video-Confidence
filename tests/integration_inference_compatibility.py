"""Opt-in offline TRAIN/generated-media experiments; never an acceptance gate.

Run in the existing restricted research Docker runtime with read-only /models,
/inputs and /fixtures, writable /results. No devices, HTTP, TEST data or policy
activation. The five acquisition-authorized identities and cyclic swaps are fixed
before execution. Feature interpolation is explicitly artificial, not new media.
"""
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np
import soundfile as sf
import torch

from deployment.general_inference import GeneralResearchService
from deployment.live.service import decode_frame
from deployment.parity_metrics import numerical_metrics, float16_cell_analysis
from deployment.release_gate import unapproved_prediction
from deployment.runtime import DeploymentError
from deployment.train_manifest import load_manifest, PERMITTED_IDS
from fusion.inference import CANDIDATES

MANIFEST = '65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e'
PERCENTILES = (0, 50, 90, 95, 99, 99.9, 100)


def distribution(actual, reference):
    """Descriptive errors; no tolerance, success threshold or approval value."""
    metrics = numerical_metrics(np.asarray(actual, np.float32), np.asarray(reference, np.float32))
    errors = np.abs(np.asarray(actual, np.float64) - np.asarray(reference, np.float64))
    return {**metrics, 'different_values': int(np.count_nonzero(errors)), 'values': errors.size,
            'absolute_error_percentiles': dict(zip(map(str, PERCENTILES), np.percentile(errors, PERCENTILES).tolist()))}


def boundary_bracket(evaluate, target, iterations=24):
    """Bisect an endpoint sign change; no monotonicity or unique root claim.

    Preserve a sign-changing bracket even for a nonmonotone function. The output
    is an artificial feature diagnostic, never a numerical rejection margin.
    """
    lo, hi = 0., 1.
    flo, fhi = evaluate(lo), evaluate(hi)
    if not all(math.isfinite(x) for x in (target, flo, fhi)) or (flo >= target) == (fhi >= target):
        raise ValueError('finite straddling endpoints required')
    for _ in range(iterations):
        mid = (lo + hi) / 2
        value = evaluate(mid)
        if not math.isfinite(value):
            raise ValueError('nonfinite boundary evaluation')
        if (value >= target) == (flo >= target):
            lo, flo = mid, value
        else:
            hi, fhi = mid, value
    return {'alpha_low':lo, 'alpha_high':hi, 'logit_low':flo, 'logit_high':fhi,
            'iterations':iterations, 'target_logit':target}


def run():
    from deployment.general_validation import fusion_score
    models, inputs, fixtures, output = map(Path, ('/models','/inputs','/fixtures','/results'))
    load_manifest(inputs/'manifest.json', MANIFEST, models, inputs)
    torch.set_num_threads(4); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    threshold = CANDIDATES['E3'].threshold
    target = math.log(threshold / (1-threshold))
    report = {'schema':'aval.compatibility-confirmation.v1', 'approved':False,
              'scope':'five authorized TRAIN sources, fixed cyclic substitutions, synthetic controls only',
              'manifest_sha256':MANIFEST, 'threshold':threshold, 'threshold_logit':target,
              'protocol':{'audio_circular_offsets_seconds':[-1.,-.1,.1,1.], 'frame_circular_shifts_bins':[-1,1],
                          'synthetic_native_sample_rates':[22050,48000],
                          'boundary_bisections':24, 'feature_probe_alpha_offsets':[-.0001,-.000001,0.,.000001,.0001],
                          'runtime_controls':[[4,True],[1,True],[4,False]], 'no_test_data':True},
              'historical':[], 'pairs':[], 'runtime_controls':[], 'generated_media':[], 'temporal_controls':[],
              'boundaries':[]}

    def save():
        (output/'experiments.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    with tempfile.TemporaryDirectory(prefix='aval-compatibility-') as tmp:
        service = GeneralResearchService(models, Path(tmp)/'derived')
        report.update(runtime=service.profile, runtime_profile_sha256=service.profile_sha256)
        wave = np.load(inputs/'waveforms.npy', allow_pickle=False)
        frames = np.load(inputs/'frames.npy', allow_pickle=False)
        historical = {}
        for role in ('audio','video'):
            folder = models/(role+'_train')
            with (folder/'train_ids.csv').open() as stream:
                ids = [r['sample_id'] for r in csv.DictReader(stream)]
            exported = np.load(folder/'train_emb.npy', allow_pickle=False)
            historical[role] = np.stack([exported[ids.index(sid)] for sid in PERMITTED_IDS])
        audio, video, canonical_audio, decoded_wave = [], [], [], []
        for i, sid in enumerate(PERMITTED_IDS):
            source = models/'video/parity_samples'/(sid.rsplit('__',1)[0]+'.mp4')
            decoded = service.decoder.decode(source.read_bytes(), source.name)
            raw = service._forward_window(decoded.waveform, decoded.frames)
            canonical = service.audio.predict_waveform(wave[i])
            audio.append(raw[2].embeddings); video.append(raw[3].embeddings)
            canonical_audio.append(canonical.embeddings); decoded_wave.append(decoded.waveform)
            report['historical'].append({'sample':sid, 'source_sha256':decoded.source_sha256,
                'waveform':distribution(decoded.waveform, wave[i]), 'pixels_exact':bool(np.array_equal(decoded.frames,frames[i])),
                'raw_vs_canonical_audio':distribution(raw[2].embeddings,canonical.embeddings),
                'canonical_audio_vs_historical':distribution(canonical.embeddings,historical['audio'][i]),
                'raw_audio_vs_historical':distribution(raw[2].embeddings,historical['audio'][i]),
                'video_vs_historical':distribution(raw[3].embeddings,historical['video'][i]),
                'audio_rounding_cell_max_excess':float16_cell_analysis(canonical.embeddings,historical['audio'][i])['max_rounding_cell_excess'],
                'video_rounding_cell_max_excess':float16_cell_analysis(raw[3].embeddings,historical['video'][i])['max_rounding_cell_excess'],
                'timing':decoded.timing})
            print('TRAIN decode/forward',sid,flush=True); save()
        audio, video, canonical_audio = map(np.stack,(audio,video,canonical_audio))
        for name, array in [('audio',audio),('video',video),('canonical_audio',canonical_audio)]:
            np.save(output/(name+'.npy'),array,allow_pickle=False)
        report['array_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.npy')}
        for offset, kind in ((0,'original'),(1,'cyclic_audio_swap')):
            for i, sid in enumerate(PERMITTED_IDS):
                j=(i+offset)%5
                variants = {'raw':fusion_score(service,audio[j],video[i]),
                            'canonical_audio':fusion_score(service,canonical_audio[j],video[i]),
                            'float16_roundtrip':fusion_score(service,audio[j].astype(np.float16).astype(np.float32),video[i].astype(np.float16).astype(np.float32)),
                            'historical_exports':fusion_score(service,historical['audio'][j].astype(np.float32),historical['video'][i].astype(np.float32))}
                report['pairs'].append({'kind':kind,'video':sid,'audio':PERMITTED_IDS[j], 'variants':variants,
                                        'decisions':{k:v['score']>=threshold for k,v in variants.items()}})

        # Preserved real Chromium preprocessing on TRAIN source zero. Verify all
        # byte bindings from the prior metrology before interpreting the arrays.
        previous = json.loads((fixtures/'metrology.json').read_text())
        for name, expected in previous['fixture_sha256'].items():
            assert hashlib.sha256((fixtures/name).read_bytes()).hexdigest()==expected, name
        ref, sr = sf.read(fixtures/'reference-ffmpeg.wav', dtype='float32')
        assert sr==16000 and ref.shape==(160000,)
        # This FFmpeg reference was reconstructed from the shared 48 kHz
        # stereo float32 intervention input, not decoded directly AAC -> 16 kHz.
        # Measure that extra conversion separately instead of conflating paths.
        refa=service.audio.predict_waveform(ref).embeddings
        browser_audio=np.frombuffer((fixtures/'browser-audio-s16.bin').read_bytes(),'<i2').astype(np.float32)/32768
        browser_frames=np.stack([decode_frame(base64.b64encode((fixtures/f'browser-live-{i}.jpg').read_bytes()).decode()) for i in range(10)])
        ba=service.audio.predict_waveform(browser_audio).embeddings
        bv=service.video.predict_frames(browser_frames).embeddings
        report['live']={'prior_evidence_sha256':hashlib.sha256((fixtures/'metrology.json').read_bytes()).hexdigest(),
                        'reconstructed_ffmpeg_vs_direct_aac_waveform':distribution(ref,decoded_wave[0]),
                        'reconstructed_ffmpeg_vs_direct_aac_embeddings':distribution(refa,audio[0]),
                        'waveform':distribution(browser_audio,ref),
                        'pixels':distribution(browser_frames,frames[0]),
                        'audio_embeddings':distribution(ba,refa), 'video_embeddings':distribution(bv,video[0]), 'pairs':[]}
        for vi,ai in ((0,0),(0,1),(4,0)):
            reference_audio=refa if ai==0 else audio[ai]
            baseline=fusion_score(service,reference_audio,video[vi])
            for name, aa, vv in [('audio_only',ba if ai==0 else reference_audio,video[vi]),
                                 ('video_only',reference_audio,bv if vi==0 else video[vi]),
                                 ('joint',ba if ai==0 else reference_audio,bv if vi==0 else video[vi])]:
                result=fusion_score(service,aa,vv)
                report['live']['pairs'].append({'audio_index':ai,'video_index':vi,'intervention':name,
                    'baseline':baseline,'result':result,'same_decision':(baseline['score']>=threshold)==(result['score']>=threshold)})

        # Direct audio sample and frame-index controls: circular shifts avoid
        # padding but are artificial discontinuous signals, not valid captures.
        for seconds in report['protocol']['audio_circular_offsets_seconds']:
            shifted=service.audio.predict_waveform(np.roll(decoded_wave[0],int(seconds*16000))).embeddings
            report['temporal_controls'].append({'kind':'circular_audio_shift','seconds':seconds,
                'result':fusion_score(service,shifted,video[0]),'embeddings':distribution(shifted,audio[0])})
        for bins in report['protocol']['frame_circular_shifts_bins']:
            report['temporal_controls'].append({'kind':'circular_frame_shift','bins':bins,
                'result':fusion_score(service,audio[0],np.roll(video[0],bins,axis=0))})

        # Controlled sample-rate round trips on the prespecified closest TRAIN
        # cyclic negative. This is not measured hardware/browser resampling.
        import torchaudio.functional as AF
        from deployment.live.buffer import _resample
        report['sample_rate_controls']=[]
        for rate in report['protocol']['synthetic_native_sample_rates']:
            native=AF.resample(torch.from_numpy(decoded_wave[4]),16000,rate).numpy()
            pcm=np.clip(np.floor(native*32768+.5),-32768,32767).astype('<i2')
            restored=_resample(pcm.astype(np.float32)/32768,rate)
            clip_count=int(np.count_nonzero(np.abs(restored)>1))
            restored=np.clip(restored,-1,1).astype(np.float32)
            encoded=service.audio.predict_waveform(restored).embeddings
            report['sample_rate_controls'].append({'native_rate':rate,'synthetic_round_trip_only':True,
                'transport_saturation_samples':int(np.count_nonzero(np.abs(native)>1)),
                'server_clipped_samples':clip_count,'waveform':distribution(restored,decoded_wave[4]),
                'embeddings':distribution(encoded,audio[4]),'baseline':fusion_score(service,audio[4],video[3]),
                'result':fusion_score(service,encoded,video[3])})

        # Identical canonical inputs, declared arithmetic controls. Restore the
        # serving execution profile before its mandatory integrity recheck.
        try:
            for threads, mkldnn in report['protocol']['runtime_controls']:
                torch.set_num_threads(threads); torch.backends.mkldnn.enabled=mkldnn
                r=service._forward_window(wave[0],frames[0])
                report['runtime_controls'].append({'threads':threads,'mkldnn':mkldnn,
                    'score':r[0],'logit':r[1], 'audio_vs_canonical':distribution(r[2].embeddings,canonical_audio[0]),
                    'video_vs_raw':distribution(r[3].embeddings,video[0]),
                    'arithmetic_control_only_not_serving_profile':True})
        finally:
            torch.set_num_threads(4); torch.backends.mkldnn.enabled=True

        for i, sid in enumerate(PERMITTED_IDS):
            j=(i+1)%5
            mix=lambda alpha: np.asarray((1-alpha)*audio[i]+alpha*audio[j],np.float32)
            bracket=boundary_bracket(lambda alpha:fusion_score(service,mix(alpha),video[i])['logit'],target)
            center=(bracket['alpha_low']+bracket['alpha_high'])/2
            probes=[]
            for step in report['protocol']['feature_probe_alpha_offsets']:
                alpha=float(np.clip(center+step,0,1)); aa=mix(alpha)
                raw=fusion_score(service,aa,video[i])
                quant=fusion_score(service,aa.astype(np.float16).astype(np.float32),video[i].astype(np.float16).astype(np.float32))
                probe={'alpha':alpha,'raw':raw,'float16_roundtrip':quant,
                       'decision_changed':(raw['score']>=threshold)!=(quant['score']>=threshold)}
                if i==0:
                    probe['measured_browser_residual']=fusion_score(service,aa+(1-alpha)*(ba-refa),bv)
                    probe['browser_residual_decision_changed']=(raw['score']>=threshold)!=(probe['measured_browser_residual']['score']>=threshold)
                probes.append(probe)
            row={'sample':sid,'synthetic_feature_only':True,'bracket':bracket,'probes':probes}
            if i==0:
                aa=mix(center)
                row['measured_browser_residual_control']={'baseline':fusion_score(service,aa,video[0]),
                    'result':fusion_score(service,aa+(1-center)*(ba-refa),bv),
                    'synthetic_combination_not_natural_capture':True}
            report['boundaries'].append(row)
        save()

        # Fresh encoded content, real ffprobe/frame scans/decoder, fixed cases.
        cases=[('mp4_25_16k','mp4','25',16000,'10'),('mov_25_48k','mov','25',48000,'10'),
               ('mp4_ntsc_44k','mp4','30000/1001',44100,'10'),('mp4_film_48k','mp4','24000/1001',48000,'10'),
               ('mp4_25_8k','mp4','25',8000,'10'),('mp4_25_96k','mp4','25',96000,'10')]
        cases += [('duration_'+d,'mp4','25',16000,d) for d in ('9.94','9.95','10.05','10.06')]
        cases += [('duration_exact_upper','mp4','20',16000,'10.05'),('duration_exact_lower','mp4','20',16000,'9.95')]
        for name, ext, fps, rate, duration in cases:
            p=Path(tmp)/(name+'.'+ext)
            service.decoder.run('ffmpeg',['-nostdin','-v','error','-y','-f','lavfi','-i',f'testsrc2=size=96x64:rate={fps}',
                '-f','lavfi','-i',f'sine=frequency=997:sample_rate={rate}','-t',duration,'-c:v','libx264','-threads','1',
                '-pix_fmt','yuv420p','-c:a','aac','-b:a','64k',str(p)])
            row={'case':name,'extension':ext,'requested_fps':fps,'requested_sample_rate':rate,'requested_duration':duration,
                 'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            row['reported_metadata']=json.loads(service.decoder.run('ffprobe',['-v','error','-show_entries',
                'stream=codec_type,codec_name,duration,start_time,time_base,sample_rate:format=duration,start_time','-of','json',str(p)]))
            try:
                d=service.decoder.decode(p.read_bytes(),p.name)
                row.update(status='DECODED',timing=d.timing,waveform_samples=d.waveform.size)
            except DeploymentError as e:
                row.update(status='REJECTED',error_code=e.code)
            report['generated_media'].append(row); save()
        original=Path(tmp)/'mp4_25_16k.mp4'
        for name, args, ext in [('shifted_audio',['-itsoffset','0.1','-i',str(original),'-i',str(original),'-map','1:v:0','-map','0:a:0'],'mp4'),
                                ('shifted_video',['-itsoffset','0.1','-i',str(original),'-i',str(original),'-map','0:v:0','-map','1:a:0'],'mp4'),
                                ('mkv',['-i',str(original)],'mkv')]:
            p=Path(tmp)/(name+'.'+ext)
            print('Generated rejection control',name,flush=True)
            service.decoder.run('ffmpeg',['-nostdin','-v','error','-y',*args,'-c','copy',str(p)])
            try:
                service.decoder.decode(p.read_bytes(),p.name)
                row={'case':name,'status':'DECODED'}
            except DeploymentError as e:
                row={'case':name,'status':'REJECTED','error_code':e.code}
            assert row['status']=='REJECTED', row
            report['generated_media'].append(row)
        webm=Path(tmp)/'unsupported.webm'
        service.decoder.run('ffmpeg',['-nostdin','-v','error','-y','-i',str(original),'-c:v','libvpx-vp9','-threads','1','-c:a','libopus',str(webm)])
        try:
            service.decoder.decode(webm.read_bytes(),webm.name)
            raise AssertionError('unexpected WebM admission; review scope')
        except DeploymentError as e:
            report['generated_media'].append({'case':'webm','status':'REJECTED','error_code':e.code})
        service._integrity()
        report['gates']={scope:unapproved_prediction(scope) for scope in ('upload','live')}
    load_manifest(inputs/'manifest.json',MANIFEST,models,inputs)
    report['all_original_artifact_hashes_preserved']=True
    report['temporary_media_cleaned']=not list(Path('/tmp').glob('aval-compatibility-*')) and not list(Path('/tmp').glob('aval-general-*'))
    assert report['temporary_media_cleaned']
    save(); print('Compatibility experiments complete; no policy activated',flush=True)


if __name__=='__main__':
    run()
