"""Opt-in isolated Docker audit of actual Chromium transformations on TRAIN fixtures.

Requires authenticated acquisition at /models and /inputs, browser comparison
outputs at /fixtures, writable /results. These are controlled interventions on
one TRAIN source, not benchmark accuracy, personal media or release approval.
"""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import soundfile as sf
import torch

from deployment.general_inference import GeneralResearchService
from deployment.live.buffer import _resample
from deployment.live.service import decode_frame
from deployment.parity_metrics import numerical_metrics
from deployment.train_manifest import load_manifest
from fusion.inference import CANDIDATES
from vision.infer import IMAGENET_MEAN, IMAGENET_STD


def pixels(actual, reference):
    d=actual.astype(np.float64)-reference.astype(np.float64)
    return {'different_values':int(np.count_nonzero(d)),'values':d.size,
            'max_absolute':float(np.abs(d).max()),'mae':float(np.abs(d).mean()),
            'rmse':float(np.sqrt(np.square(d).mean()))}


def run():
    models=Path('/models');inputs=Path('/inputs');fixtures=Path('/fixtures')
    manifest_sha='65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e'
    load_manifest(inputs/'manifest.json',manifest_sha,models,inputs)
    profile=json.loads((fixtures/'browser-comparison-profile.json').read_text())
    reference_audio,sr=sf.read(fixtures/'reference-ffmpeg.wav',dtype='float32');assert sr==16000
    browser_audio=np.frombuffer((fixtures/'browser-audio-s16.bin').read_bytes(),dtype='<i2').astype(np.float32)/32768
    assert reference_audio.shape==browser_audio.shape==(160000,)
    reference_frames=np.load(inputs/'frames.npy',allow_pickle=False)[0]
    browser_frames=np.stack([decode_frame(base64.b64encode((fixtures/f'browser-live-{k}.jpg').read_bytes()).decode()) for k in range(10)])
    report={'scope':'one prespecified TRAIN source, actual Chromium audio/canvas intervention; not an accuracy benchmark',
        'approved':False,'browser_profile':profile,'manifest_sha256':manifest_sha,
        'audio_vs_ffmpeg':numerical_metrics(browser_audio,reference_audio),
        'audio_pcm_differing_samples':int(np.count_nonzero(browser_audio!=reference_audio)),
        'video_rgb224':pixels(browser_frames,reference_frames),'image_probes':[]}
    norm=lambda a:(a.astype(np.float32)/255).transpose(0,3,1,2)
    bn=(norm(browser_frames)-IMAGENET_MEAN)/IMAGENET_STD
    rn=(norm(reference_frames)-IMAGENET_MEAN)/IMAGENET_STD
    report['normalized_pixels']=numerical_metrics(bn,rn)
    for i in (0,2):
        original=np.load(fixtures/f'source-{i}.npy')
        frame=next(f for f in profile['frames'] if f['source']==i)
        h,w=frame['height'],frame['width']
        canvas=np.frombuffer((fixtures/f'browser-{i}.rgba').read_bytes(),np.uint8).reshape(h,w,4)[...,:3]
        jpeg=decode_frame(base64.b64encode((fixtures/f'browser-{i}.jpg').read_bytes()).decode())
        browser224=np.frombuffer((fixtures/f'browser-{i}-resize.rgba').read_bytes(),np.uint8).reshape(224,224,4)[...,:3]
        ref224=cv2.resize(original,(224,224),interpolation=cv2.INTER_LINEAR)
        report['image_probes'].append({'source':i,
            'canvas_vs_opencv_at_canvas_size':pixels(canvas,cv2.resize(original,(w,h),interpolation=cv2.INTER_LINEAR)),
            'jpeg_then_actual_server_vs_producer':pixels(jpeg,ref224),
            'browser_resize_vs_opencv_without_jpeg':pixels(browser224,cv2.resize(canvas,(224,224),interpolation=cv2.INTER_LINEAR))})
    # A separate native-rate fallback route: PCM quantization precedes server
    # torchaudio resampling rather than browser resampling preceding PCM.
    stereo=np.frombuffer((fixtures/'stereo-48000.f32').read_bytes(),'<f4').reshape(-1,2)
    mono=stereo.mean(axis=1,dtype=np.float32)
    pcm=np.clip(np.floor(mono*32768+.5),-32768,32767).astype('<i2')
    native=pcm.astype(np.float32)/32768
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    server_audio=_resample(native,48000)
    report['server_native_48000_vs_ffmpeg']=numerical_metrics(np.clip(server_audio,-1,1).astype(np.float32),reference_audio)
    report['server_resample_clipped_samples']=int(np.count_nonzero(np.abs(server_audio)>1))
    with tempfile.TemporaryDirectory(prefix='aval-live-metrology-') as derived:
        worker=GeneralResearchService(models,Path(derived))
        with tempfile.TemporaryDirectory(prefix='aval-pcm-rounding-') as directory:
            pcm_root=Path(directory)
            probe=np.asarray(profile['audio']['clip_round_probe_input'],np.float32)
            (pcm_root/'input.f32').write_bytes(probe.tobytes())
            worker.decoder.run('ffmpeg',['-nostdin','-v','error','-y','-f','f32le','-ar','16000','-ac','1',
                '-i',str(pcm_root/'input.f32'),'-c:a','pcm_s16le','-f','s16le',str(pcm_root/'output.s16')])
            report['pcm_clipping_rounding_probe']={'input':probe.tolist(),
                'browser_int16':np.frombuffer(base64.b64decode(profile['audio']['clip_round_probe_pcm']),'<i2').tolist(),
                'ffmpeg_int16':np.frombuffer((pcm_root/'output.s16').read_bytes(),'<i2').tolist()}
        with patch('torch.load',wraps=torch.load) as loads:
            with worker._lock:
                worker._integrity()
                raw=worker._forward_window(reference_audio,reference_frames)
                browser=worker._forward_window(browser_audio,browser_frames)
                worker._integrity()
            assert loads.call_count==0
        threshold=CANDIDATES['E3'].threshold
        report['genuine_models']={'reference_score':raw[0],'reference_logit':raw[1],
            'browser_score':browser[0],'browser_logit':browser[1],'threshold':threshold,
            'same_decision':(raw[0]>=threshold)==(browser[0]>=threshold),
            'score_delta':browser[0]-raw[0], 'logit_delta':browser[1]-raw[1],
            'audio_embeddings':numerical_metrics(browser[2].embeddings,raw[2].embeddings),
            'video_embeddings':numerical_metrics(browser[3].embeddings,raw[3].embeddings),
            'request_checkpoint_load_calls':loads.call_count,'model_sha256':CANDIDATES['E3'].model_sha256,
            'checkpoint_sha256':CANDIDATES['E3'].checkpoint_sha256,'runtime':worker.profile}
    load_manifest(inputs/'manifest.json',manifest_sha,models,inputs)
    report['all_40_model_and_5_input_hashes_preserved']=True
    names=['browser-comparison-profile.json','browser-audio-s16.bin','reference-ffmpeg.wav']+[f'browser-live-{k}.jpg' for k in range(10)]
    report['fixture_sha256']={n:hashlib.sha256((fixtures/n).read_bytes()).hexdigest() for n in names}
    Path('/results/metrology.json').write_text(json.dumps(report,indent=2))
    print('Actual browser preprocessing and genuine encoder/E3 comparison completed',flush=True)


if __name__=='__main__':run()
