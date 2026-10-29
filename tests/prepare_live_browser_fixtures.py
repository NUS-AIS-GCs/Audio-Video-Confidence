"""Opt-in Docker preparation of manifest-authenticated TRAIN-only browser fixtures."""

from pathlib import Path
import hashlib,json,subprocess
import cv2,numpy as np
from deployment.train_manifest import load_manifest,PERMITTED_IDS
models=Path('/models');inputs=Path('/inputs');out=Path('/results')
sha='65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e'
load_manifest(inputs/'manifest.json',sha,models,inputs)
def ff(args):subprocess.run(['ffmpeg','-nostdin','-v','error','-y',*args],check=True,capture_output=True,timeout=40)
src=models/'video/parity_samples/c---zaDCTaE.mp4'
ff(['-i',str(src),'-t','10','-ac','2','-ar','48000','-c:a','pcm_f32le','-f','f32le',str(out/'stereo-48000.f32')])
ff(['-f','f32le','-ar','48000','-ac','2','-i',str(out/'stereo-48000.f32'),'-ac','1','-ar','16000','-t','10','-c:a','pcm_s16le',str(out/'reference-ffmpeg.wav')])
refs=[];canonical=np.load(inputs/'frames.npy')
for idx,w,h in [(0,640,360),(2,1280,720)]:
 sid=PERMITTED_IDS[idx];src=models/'video/parity_samples'/(sid.rsplit('__',1)[0]+'.mp4')
 raw=out/f'pixels-{idx}.bgr'
 ff(['-i',str(src),'-map','0:v:0','-an','-vf','select=eq(n\,0)','-frames:v','1','-threads','1','-pix_fmt','bgr24','-f','rawvideo',str(raw)])
 bgr=np.frombuffer(raw.read_bytes(),np.uint8).reshape(h,w,3);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 assert np.array_equal(cv2.resize(rgb,(224,224),interpolation=cv2.INTER_LINEAR),canonical[idx,0])
 cv2.imwrite(str(out/f'source-{idx}.png'),bgr);np.save(out/f'source-{idx}.npy',rgb)
 refs.append({'source':sid,'width':w,'height':h,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
 'original_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'producer_rgb224_sha256':hashlib.sha256(canonical[idx,0].tobytes()).hexdigest()})
(out/'fixture-identities.json').write_text(json.dumps({'manifest_sha256':sha,'train_only':True,'images':refs,
 'stereo_f32_sha256':hashlib.sha256((out/'stereo-48000.f32').read_bytes()).hexdigest()},indent=2))
print('Authenticated TRAIN source fixtures prepared')

src=models/'video/parity_samples/c---zaDCTaE.mp4'
raw=out/'ten-frames.bgr'
select='+'.join('eq(n\\,'+str(k*25)+')' for k in range(10))
ff(['-i',str(src),'-map','0:v:0','-an','-vf','select='+select,'-vsync','0','-frames:v','10','-threads','1','-pix_fmt','bgr24','-f','rawvideo',str(raw)])
frames=np.frombuffer(raw.read_bytes(),np.uint8).reshape(10,360,640,3)
for k,bgr in enumerate(frames):
 assert np.array_equal(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(224,224)),canonical[0,k])
 cv2.imwrite(str(out/f'source-live-{k}.png'),bgr)
