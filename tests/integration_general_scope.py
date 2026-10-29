"""Opt-in Docker scope test using newly generated content and authentic models.

The deterministic color/sine input is a transport fixture, never an accuracy
example. All reported scores come from the actual frozen pipeline.
"""
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from deployment.general_inference import GeneralResearchService, STATUS, response_for_research
from deployment.runtime import DeploymentError


def run():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    report={'scope':'generated color/sine engineering fixture, no benchmark or TRAIN/TEST attribution',
            'approved':False,'formats':{}}
    with tempfile.TemporaryDirectory(prefix='aval-generated-scope-') as directory:
        root=Path(directory);worker=GeneralResearchService('/models',root/'derived')
        source=root/'new-generated.mp4'
        worker.decoder.run('ffmpeg',['-nostdin','-v','error','-y','-f','lavfi','-i','color=c=gray:s=224x224:r=25',
            '-f','lavfi','-i','sine=frequency=1000:sample_rate=16000','-t','10','-c:v','libx264','-pix_fmt','yuv420p',
            '-threads','1','-c:a','aac',str(source)])
        result=worker.predict_research(source.read_bytes(),source.name)
        assert result['status']==STATUS,result
        report['genuine_new_content_result']=response_for_research(result)
        report['profile_sha256']=worker.profile_sha256
        for extension in ('mov','mkv','webm'):
            dest=root/('fixture.'+extension)
            codec=['-c','copy'] if extension!='webm' else [
                '-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','8','-threads','1','-c:a','libopus']
            worker.decoder.run('ffmpeg',['-nostdin','-v','error','-y','-i',str(source),*codec,str(dest)])
            try:
                decoded=worker.decoder.decode(dest.read_bytes(),dest.name)
                report['formats'][extension]={'status':'DECODED', 'timing':decoded.timing,
                    'same_mp4_waveform':bool(np.array_equal(decoded.waveform,result['_audit'][0])),
                    'same_mp4_pixels':bool(np.array_equal(decoded.frames,result['_audit'][1]))}
            except DeploymentError as exc:
                report['formats'][extension]={'status':'REJECTED','error_code':exc.code}
        assert report['formats']['mov']['status']=='DECODED'
        worker._integrity()
    Path('/results/generated-scope.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':run()
