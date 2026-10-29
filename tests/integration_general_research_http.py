"""Opt-in genuine HTTP smoke, run in the documented isolated research container.

Requires /models, /results and the authentic catalog; ordinary pytest does not
load models. This script exercises real private Gradio queue/ownership paths.
"""
import json
import os
from pathlib import Path
import secrets
import tempfile

import httpx
import torch


def run():
    from deployment.general_inference import GeneralResearchService, STATUS
    from deployment.provision_artifacts import runtime_bindings
    from deployment.gradio_app import create_app
    from deployment.live.security import live_app_kwargs
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    os.environ.update(runtime_bindings(Path('/models')))
    os.environ.update(AVAL_ACCESS_MODE='gradio_auth',AVAL_GRADIO_USERNAME='research-smoke',
                      AVAL_GRADIO_PASSWORD=secrets.token_urlsafe(32),GRADIO_ANALYTICS_ENABLED='False')
    checks={}
    with tempfile.TemporaryDirectory(prefix='aval-research-http-') as directory:
        worker=GeneralResearchService('/models',Path(directory)/'derived')
        demo,config=create_app(general_research=worker)
        try:
            demo.launch(server_name='127.0.0.1',server_port=7860,share=False,auth=config.gradio_auth,
                prevent_thread_lock=True,quiet=True,show_error=False,enable_monitoring=False,
                blocked_paths=['/models'],max_file_size=config.max_upload_bytes,
                app_kwargs=live_app_kwargs(demo.aval_live_service))
            base='http://127.0.0.1:7860'
            with httpx.Client(base_url=base,timeout=120) as anon, httpx.Client(base_url=base,timeout=120) as a, httpx.Client(base_url=base,timeout=120) as b:
                for client in (a,b):
                    assert client.post('/login',data={'username':config.auth_username,'password':config.auth_password}).status_code==200
                cfg=a.get('/config').json()
                raw_id=next(c['id'] for c in cfg['components'] if c['type']=='json' and c['props'].get('label','').startswith('Research response'))
                research=next(d for d in cfg['dependencies'] if raw_id in d['outputs'] and d.get('queue'))
                public_id=next(c['id'] for c in cfg['components'] if c['type']=='json' and c['id']!=raw_id)
                public=next(d for d in cfg['dependencies'] if public_id in d['outputs'] and d.get('queue'))
                assert research['api_visibility']=='private'
                checks['research_queue_private']=True
                assert anon.get('/config').status_code==401
                assert anon.post('/gradio_api/upload',files={'files':('x.mp4',b'x')}).status_code==401
                assert anon.post('/gradio_api/queue/join',json={'fn_index':research['id'],'data':['x'],'session_hash':'x'}).status_code==401
                checks['unauthenticated_config_upload_queue_denied']=True

                def verify(client,handle,dep):
                    session=secrets.token_hex(12)
                    joined=client.post('/gradio_api/queue/join',json={'fn_index':dep['id'],'data':[handle],'session_hash':session})
                    assert joined.status_code==200,joined.text
                    event=joined.json()['event_id'];loading=False
                    with client.stream('GET','/gradio_api/queue/data',params={'session_hash':session}) as stream:
                        for line in stream.iter_lines():
                            if not line.startswith('data: '):continue
                            row=json.loads(line[6:])
                            if row.get('event_id')!=event:continue
                            loading |= row.get('msg')=='process_generating'
                            if row.get('msg')=='process_completed':
                                assert row['success'],row
                                return row['output']['data'][1],loading
                    raise AssertionError('no completion')

                data=Path('/models/video/parity_samples/c---zaDCTaE.mp4').read_bytes()
                def upload(client):
                    r=client.post('/gradio_api/upload',files={'files':('arbitrary-research-name.mp4',data,'video/mp4')})
                    assert r.status_code==200,r.text
                    return r.json()[0]
                handle=upload(a)
                assert len(handle)==64 and '/' not in handle
                p=demo.aval_upload_store.root/('payload-'+handle+'.mp4')
                for kind in ('file=','file/','proxy=','stream/'):
                    assert a.get('/gradio_api/'+kind+p.as_posix()).status_code==403
                rejected,_=verify(b,handle,research)
                assert rejected['error_code']=='UPLOAD_NOT_FOUND' and p.exists()
                checks['cross_login_cannot_consume_or_download']=True
                result,loading=verify(a,handle,research)
                assert loading and result['status']==STATUS and not p.exists()
                assert result['production_predictions_enabled'] is False and result['independent_approval'] is None
                assert '_audit' not in result and 'train_signature' not in json.dumps(result)
                checks['genuine_authenticated_research_result']=result
                stale,_=verify(a,handle,research)
                assert stale['error_code']=='UPLOAD_NOT_FOUND'
                checks['stale_handle_rejected']=True
                gated,_=verify(a,upload(a),public)
                assert gated['status']=='MODEL_NOT_READY' and gated['predictions'] is None
                assert 'score' not in gated
                checks['public_result_stays_gated_with_loaded_models']=gated
                assert not list(demo.aval_upload_store.root.glob('payload-*'))
                checks['all_owned_payloads_cleaned']=True
        finally:
            demo.close();demo.aval_live_service.close();demo.aval_upload_store.close()
    Path('/results/http-smoke.json').write_text(json.dumps(checks,indent=2))
    print('Genuine research HTTP smoke passed',flush=True)


if __name__=='__main__':run()
