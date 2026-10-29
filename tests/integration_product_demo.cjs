// Functional acceptance on the separately launched private demo. Fake devices ONLY.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(path.join(process.cwd(),'_local/premium_ui/browser/node_modules/playwright'));
if(process.argv.includes('--physical'))throw Error('Physical devices are not authorized');
const root=process.cwd(),base=process.env.AVAL_PRODUCT_TEST_URL||'http://127.0.0.1:7862';
const privateDir=path.resolve(process.env.AVAL_PRODUCT_PRIVATE_DIR||path.join(root,'_local/product_demo/private')),
  out=path.resolve(process.env.AVAL_PRODUCT_TEST_OUTPUT||path.join(root,'_local/product_demo/evidence'));
fs.mkdirSync(out,{recursive:true});
const login=JSON.parse(fs.readFileSync(path.join(privateDir,'login.json'),'utf8'));
const source=path.join(root,'_local/phase8_stage2c/models/video/parity_samples/c---zaDCTaE.mp4');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const report={fake_devices_only:true};let browser,activePage;
const requireHead2=process.env.AVAL_PRODUCT_REQUIRE_HEAD2==='1';
const head2Catalog=JSON.parse(fs.readFileSync(path.join(root,'deployment/temporal_presence_catalog.json')));
const save=()=>fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));
const classNames=JSON.parse(fs.readFileSync(path.join(root,'deployment/event_class_names.json')));
const catalog=JSON.parse(fs.readFileSync(path.join(root,'deployment/artifact_catalog.json'))).files;
function checkEvents(result) {
  const e=result.event_predictions;
  assert.equal(e.schema,'aval.encoder-event-evidence.v1');assert.deepEqual(e.class_names,classNames);
  if(requireHead2){
    const t=result.temporal_predictions;assert.equal(t.schema,'aval.temporal-presence-research.v1');
    assert.equal(t.model_version,head2Catalog.version);assert.equal(t.checkpoint_sha256,head2Catalog.files['weights.pt'].sha256);
    assert.equal(t.threshold,head2Catalog.threshold);assert.equal(t.production_predictions_enabled,false);
    assert.equal(t.scores.length,10);assert.equal(t.logits.length,10);assert.equal(t.presence.length,10);
    for(let i=0;i<10;i++){
      assert.ok(Number.isFinite(t.scores[i])&&Number.isFinite(t.logits[i])&&t.scores[i]>=0&&t.scores[i]<=1);
      assert.equal(t.presence[i],t.scores[i]>=t.threshold);
    }
  }else assert.equal(result.temporal_predictions,null);
  assert.deepEqual(e.bin_starts,[0,1,2,3,4,5,6,7,8,9]);
  for(const key of ['audio','video']){
    const m=e[key];assert.deepEqual(m.shape,[10,29]);assert.equal(m.dtype,'float32');assert.equal(m.scores.length,10);
    for(const row of m.scores){assert.equal(row.length,29);assert.ok(row.every(x=>Number.isFinite(x)&&x>=0&&x<=1&&Math.fround(x)===x));assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<=29*2**-23);}
    assert.equal(m.provenance.artifact_sha256['weights.pt'],catalog[key+'/weights.pt'].sha256);
    assert.equal(m.provenance.artifact_sha256['class_names.json'],catalog[key+'/class_names.json'].sha256);
  }
  return e;
}
async function checkRenderedHeads(container,e,temporal) {
  assert.equal(await container.locator('[data-head]').count(),3);
  if(requireHead2){
    assert.equal(await container.locator('.av-presence-cell').count(),10);
    assert.ok((await container.locator('[data-head=temporal]').innerText()).includes('H2-AVE-presence-v1'));
    for(let i=0;i<10;i++){
      const cell=container.locator('.av-presence-cell').nth(i);
      assert.equal(await cell.getAttribute('data-state'),temporal.presence[i]?'present':'background');
      assert.ok(Math.abs(Number(await cell.locator('span').last().innerText())-temporal.scores[i])<=.000000500001);
    }
  }else assert.ok((await container.locator('[data-head=temporal]').innerText()).includes('verified artifact not loaded'));
  for(const [j,key] of ['audio','video'].entries()){
    const means=e.class_names.map((_,i)=>e[key].scores.reduce((s,row)=>s+row[i],0)/10);
    const order=Array.from({length:28},(_,i)=>i).sort((a,b)=>means[b]-means[a]||a-b).slice(0,5);
    const items=container.locator('.av-event-list').nth(j).locator('li');assert.equal(await items.count(),5);
    for(let i=0;i<5;i++){
      assert.equal(await items.nth(i).locator('span').innerText(),e.class_names[order[i]]);
      assert.ok(Math.abs(Number(await items.nth(i).locator('strong').innerText())-means[order[i]])<=.000000500001);
    }
  }
  const timeline=container.locator('.av-event-details');await timeline.locator('summary').click();
  assert.equal(await timeline.locator('tbody tr').count(),10);
  await timeline.locator('summary').click();
}
(async()=>{
 try{
  browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  const a=await browser.newContext({permissions:['camera','microphone'],viewport:{width:1440,height:1080}});
  // Observe real SSE packets as they arrive; a persistent queue stream need
  // not finish when its generator has displayed the final response.
  await a.addInitScript(()=>{
    window.__avObservedResults=[];
    const observe=packet=>{
      const value=packet.output?.data?.[1];
      if(value?.schema==='aval.general-media-research.v1'){
        window.__avObservedResults.push(value);
        if(window.__avObservedResults.length>20)window.__avObservedResults.shift();
      }
    };
    const original=window.fetch;
    window.fetch=async function(...args){
      const response=await original.apply(this,args);
      if(new URL(response.url,location.href).pathname.includes('/queue/data')&&response.body){
        const reader=response.clone().body.getReader(),decoder=new TextDecoder();
        (async()=>{let pending='';try{for(;;){const {done,value}=await reader.read();if(done)break;
          pending+=decoder.decode(value,{stream:true});if(pending.length>512*1024)break;
          let n;while((n=pending.indexOf('\n'))>=0){const line=pending.slice(0,n);pending=pending.slice(n+1);
            if(line.startsWith('data: ')){try{observe(JSON.parse(line.slice(6)));}catch{}}
          }
        }}finally{await reader.cancel().catch(()=>{});}})().catch(()=>{});
      }
      return response;
    };
    if(window.EventSource){const Original=window.EventSource;window.EventSource=class extends Original{
      constructor(...args){super(...args);this.addEventListener('message',event=>{try{observe(JSON.parse(event.data));}catch{}});}
    };}
  });
  await a.route('**/*',r=>{const u=new URL(r.request().url());return ['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)?r.abort():r.continue();});
  await a.addInitScript(()=>{
    window.__avDeviceTracks=[];window.__avLiveCaptureProgress=[];
    const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia=async constraints=>{
      const stream=await original(constraints);window.__avDeviceTracks.push(...stream.getTracks());return stream;
    };
  });
  const page=await a.newPage(),completed=[],errors=[],progress=[];let sealed,seals=0;activePage=page;
  page.on('pageerror',e=>errors.push(e.message));
  page.on('requestfinished',async req=>{
   try{
    const u=new URL(req.url());
    if(u.pathname.includes('/queue/data')){
     const text=await (await req.response()).text();
     for(const line of text.split('\n'))if(line.startsWith('data: ')){
      const r=JSON.parse(line.slice(6));
      if(r.output?.data?.[1]?.schema==='aval.general-media-research.v1')completed.push(r.output.data[1]);
     }
    }
    if(u.pathname.includes('component_server')){
     const r=await(await req.response()).json();if(r.capture_progress)progress.push(r.capture_progress);
     if(r.status==='RESEARCH_CAPTURE_SEALED'){seals++;sealed={ticket:r.ticket,cancel_key:r.cancel_key};}
    }
   }catch{}
  });
  await page.goto(base,{waitUntil:'networkidle'});
  await page.getByLabel('username').fill(login.username);await page.getByLabel('password').fill(login.password);
  await page.getByRole('button',{name:'Login',exact:true}).click();
  await page.getByRole('tab',{name:'Upload Video',exact:true}).click();
  const file=page.locator('input[type=file]').first(),card=page.locator('#av-general-result');
  const selectFile=async value=>{
    const component=page.locator('#av-general-research-source');
    const clear=component.getByRole('button',{name:'Clear',exact:true});
    if(await clear.count())await clear.click();
    await component.locator('input[type=file]').setInputFiles(value);
  };
  const readUpload=async()=>{
    const raw=card.locator('[data-av-raw-result]');
    let actual;
    if(await raw.count()){
      actual=JSON.parse(await raw.textContent());
      assert.equal(await card.locator('.av-raw-score').innerText(),String(actual.score));
      assert.equal(await card.locator('[data-head=correspondence] h4').innerText(), actual.score>=actual.threshold?'MATCH':'MISMATCH');
      assert.ok((await card.locator('[data-head=correspondence]').innerText()).includes(String(actual.threshold)));
    }else{
      await page.waitForFunction(()=>window.__avObservedResults?.some(r=>r.status==='REJECTED'&&document.querySelector('#av-general-result')?.textContent.includes(r.error_code)),{},{timeout:10000});
      actual=await page.evaluate(()=>window.__avObservedResults.findLast(r=>r.status==='REJECTED'&&document.querySelector('#av-general-result')?.textContent.includes(r.error_code)));
    }
    assert.ok(!JSON.stringify(actual).includes('"cancel_key"')&&!('owner' in actual)&&!('_audit' in actual));
    const result={};
    for(const key of ['schema','status','request_id','source_sha256','profile_sha256','score','logit','processing_seconds',
      'threshold','model_sha256','checkpoint_sha256','classification','score_semantics','production_predictions_enabled','predictions','error_code'])
      if(key in actual)result[key]=actual[key];
    return result;
  };
  await selectFile(source);await page.getByRole('button',{name:'Verify',exact:true}).click();
  await page.waitForFunction(()=>{const t=document.querySelector('#av-general-result')?.textContent||'';return t.includes('Actual raw E3 score')||t.includes('UNABLE TO VERIFY');},{},{timeout:180000});
  assert.ok((await card.innerText()).includes('Actual raw E3 score'),await card.innerText());
  report.upload=await readUpload();assert.equal(report.upload.status,'EXPERIMENTAL_GENERAL_MEDIA_NOT_APPROVED');
  assert.equal(report.upload.threshold,0.009579099009671126);assert.equal(report.upload.production_predictions_enabled,false);
  for(const key of ['_audit','ticket','owner','cancel_key'])assert.ok(!(key in report.upload));
  save();
  await page.waitForFunction(id=>window.__avObservedResults?.some(r=>r.request_id===id),report.upload.request_id,{timeout:10000});
  const uploadResponse=await page.evaluate(id=>window.__avObservedResults.find(r=>r.request_id===id),report.upload.request_id);
  assert.equal(uploadResponse.schema,'aval.general-media-research.v1');
  assert.equal(uploadResponse.score,report.upload.score);
  // The nested event schema appears before the root schema in the JSON tree.
  assert.ok((await card.locator('[data-av-raw-result]').textContent()).includes(uploadResponse.schema));
  report.upload.schema=uploadResponse.schema;
  report.upload_events=checkEvents(uploadResponse);report.upload_temporal=uploadResponse.temporal_predictions;
  await checkRenderedHeads(card,report.upload_events,report.upload_temporal);
  await card.screenshot({path:path.join(out,'upload-three-heads.png')});
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+2),'responsive page must not overflow');
  await card.screenshot({path:path.join(out,'upload-three-heads-mobile.png')});
  await page.setViewportSize({width:1440,height:1080});report.upload_three_heads_verified=true;
  await page.screenshot({path:path.join(out,'upload.png'),fullPage:true});
  await selectFile(path.join(root,'_local/product_demo/fixtures/cyclic-negative.mp4'));
  await page.getByRole('button',{name:'Verify',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#av-general-result')?.textContent.includes('Actual raw E3 score'),{},{timeout:180000});
  report.negative_upload=await readUpload();
  const negativeResponse=await page.evaluate(id=>window.__avObservedResults.find(r=>r.request_id===id),report.negative_upload.request_id);
  checkEvents(negativeResponse);await checkRenderedHeads(card,negativeResponse.event_predictions,negativeResponse.temporal_predictions);
  report.negative_three_heads_verified=true;
  assert.equal(report.negative_upload.classification,'Audio-substituted-like pairing');
  assert.ok(report.negative_upload.score<report.negative_upload.threshold);save();
  await page.screenshot({path:path.join(out,'negative-upload.png'),fullPage:true});
  await selectFile({name:'invalid.mp4',mimeType:'video/mp4',buffer:Buffer.from('invalid media')});
  await page.getByRole('button',{name:'Verify',exact:true}).click();
  await card.getByRole('heading',{name:/UNABLE TO VERIFY|Verification unavailable/}).waitFor({timeout:60000});
  report.invalid=await readUpload();assert.equal(report.invalid.status,'REJECTED');assert.equal(report.invalid.predictions,null);
  save();
  assert.ok(!(await card.innerText()).includes('Actual raw E3 score'));

  // Known foreign handle cannot be consumed by a different login, even same username.
  const b=await browser.newContext();await b.request.post(base+'/login',{form:login});
  const cfg=await(await a.request.get(base+'/config')).json();
  const json=cfg.components.find(c=>c.props.elem_id==='av-general-response');
  const dep=cfg.dependencies.find(d=>d.outputs.includes(json.id)&&d.queue);
  const uploaded=await a.request.post(base+'/gradio_api/upload',{multipart:{files:{name:'foreign.mp4',mimeType:'video/mp4',buffer:fs.readFileSync(source)}}});
  const handle=(await uploaded.json())[0];
  const session='product-foreign-'+Date.now();
  assert.equal((await b.request.post(base+'/gradio_api/queue/join',{data:{fn_index:dep.id,data:[handle],session_hash:session}})).status(),200);
  const bpage=await b.newPage();await bpage.goto(base,{waitUntil:'networkidle'});
  const consume=async function(session){
    const response=await fetch('/gradio_api/queue/data?session_hash='+encodeURIComponent(session));
    const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';
    for(;;){const {done,value}=await reader.read();if(done)throw Error('No queued completion');pending+=decoder.decode(value,{stream:true});
      let nl;while((nl=pending.indexOf('\n'))>=0){const line=pending.slice(0,nl);pending=pending.slice(nl+1);
        if(line.startsWith('data: ')){const row=JSON.parse(line.slice(6));if(row.msg==='process_completed'){await reader.cancel();return row;}}
      }
    }
  };
  const foreign=await bpage.evaluate(consume,session);
  assert.equal(foreign.output.data[1].error_code,'UPLOAD_NOT_FOUND');report.foreign_upload_denied=true;
  const ownerSession=session+'-owner';
  await a.request.post(base+'/gradio_api/queue/join',{data:{fn_index:dep.id,data:[handle],session_hash:ownerSession}});
  const ownerResult=await page.evaluate(consume,ownerSession);
  assert.equal(ownerResult.output.data[1].status,'EXPERIMENTAL_GENERAL_MEDIA_NOT_APPROVED');report.owner_can_consume_protected_upload=true;
  const anon=await browser.newContext();assert.equal((await anon.request.get(base+'/config')).status(),401);await anon.close();
  save();
  // Cancel a real upload after its signed job appears. No stale score may render.
  await selectFile({name:'cancel.mp4',mimeType:'video/mp4',buffer:fs.readFileSync(source)});
  await page.getByRole('button',{name:'Verify',exact:true}).click();
  for(let i=0;i<200&&!fs.readdirSync(path.join(privateDir,'upload-spool')).some(n=>n.endsWith('.grant.json'));i++)await wait(50);
  assert.ok(fs.readdirSync(path.join(privateDir,'upload-spool')).some(n=>n.endsWith('.grant.json')));
  await page.getByRole('button',{name:'Cancel / reset result',exact:true}).click();await wait(2000);
  assert.ok(!fs.readdirSync(path.join(privateDir,'upload-spool')).some(n=>/\.(grant\.json|upload\.bin|result\.json)$/.test(n)));
  assert.ok(!(await card.innerText()).includes('Actual raw E3 score'));report.upload_cancelled_no_stale_result=true;
  assert.equal(await card.locator('.av-event-list').count(),0);
  assert.equal(await card.locator('.av-presence-cell').count(),0);
  save();

  await page.getByRole('tab',{name:'Live Verification',exact:true}).click();
  const live=page.locator('.av-live').last();
  await live.locator('.av-record').click();assert.ok((await live.locator('.av-msg').innerText()).includes('Consent'));
  await live.locator('.av-research-consent').check();await live.locator('.av-record').click();
  await page.waitForFunction(()=>{const v=document.querySelector('.av-preview');return v?.srcObject&&v.videoWidth>0&&v.readyState>=2;},{},{timeout:10000});
  report.preview_working=await live.locator('.av-preview').evaluate(v=>{
    const c=document.createElement('canvas');c.width=32;c.height=24;const ctx=c.getContext('2d');ctx.drawImage(v,0,0,32,24);
    return ctx.getImageData(0,0,32,24).data.some((x,i)=>i%4!==3&&x>20);
  });assert.equal(report.preview_working,true);
  await page.waitForFunction(()=>document.querySelector('.av-research-result')?.textContent.includes('EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED')||document.querySelector('.av-msg')?.dataset.kind==='error',{}, {timeout:180000});
  const display=await live.locator('.av-research-result').innerText();
  const result=JSON.parse(await live.locator('[data-av-raw-result]').textContent());
  assert.equal(result.status,'EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED');
  assert.equal(result.schema,'aval.live-input-research.v1');assert.ok(display.includes(result.classification));
  report.live={score:result.score,logit:result.logit,classification:result.classification,threshold:result.threshold,
    checkpoint_sha256:result.checkpoint_sha256,model_sha256:result.model_sha256,profile_sha256:result.profile_sha256,
    processing_seconds:result.processing_seconds,timing:result.live_input.timing};
  assert.equal(result.production_predictions_enabled,false);assert.ok(sealed);
  assert.equal(seals,1);report.sealed_exactly_once=true;
  assert.equal(progress.at(-1).accepted_audio_samples,160000);
  assert.ok(progress.every(p=>p.accepted_audio_samples<=160000&&p.accepted_clock_observations<=64));
  report.live_capture_progress=progress.at(-1);
  assert.equal(await page.evaluate(()=>window.__avDeviceTracks.every(t=>t.readyState==='ended')),true);
  report.tracks_released_before_results=true;
  assert.equal(await live.locator('.av-timer').innerText(),'10.0 s / 10 s');
  assert.equal(await live.locator('[data-head=correspondence] h4').innerText(),result.score>=result.threshold?'MATCH':'MISMATCH');
  report.live_events=checkEvents(result);report.live_temporal=result.temporal_predictions;
  await checkRenderedHeads(live.locator('.av-research-result'),report.live_events,report.live_temporal);
  const liveTimeline=live.locator('.av-event-details');await liveTimeline.locator('summary').click();
  await wait(1200);assert.ok(await liveTimeline.evaluate(e=>e.open),'polling must preserve expanded evidence');
  await liveTimeline.locator('summary').click();
  await live.locator('.av-research-result').screenshot({path:path.join(out,'live-three-heads.png')});
  report.live_three_heads_verified=true;
  save();
  assert.ok(!fs.existsSync(path.join(privateDir,'spool',result.ticket+'.capture.json')));report.live_media_deleted=true;
  await page.screenshot({path:path.join(out,'live.png'),fullPage:true});
  const component=cfg.components.find(c=>(c.props.html_template||'').includes('av-research-consent'));
  for(const method of ['research_poll','research_cancel']){
   const r=await b.request.post(base+'/gradio_api/component_server',{data:{component_id:component.id,session_hash:'foreign',fn_name:method,data:sealed}});
   assert.equal((await r.json()).status,'REJECTED');
  }
  report.foreign_live_denied=true;
  await live.locator('.av-reset').click();await wait(1000);
  assert.equal(await live.locator('.av-research-result').innerText(),'');
  assert.ok(!fs.existsSync(path.join(privateDir,'spool',result.ticket+'.result.json')));report.live_reset_revoked=true;
  save();
  await live.locator('.av-research-consent').check();await live.locator('.av-record').click();await wait(1500);await live.locator('.av-stop').click();await wait(700);
  assert.equal(await live.locator('.av-research-result').innerText(),'');report.live_stop_no_result=true;
  for(const route of ['file=','proxy=','stream/'])assert.equal((await a.request.get(base+'/gradio_api/'+route+privateDir.replaceAll('\\','/')+'/login.json')).status(),403);
  if(requireHead2)assert.equal((await a.request.get(base+'/gradio_api/file='+path.join(root,'_local/temporal_presence/product-models-host/Head2/weights.pt').replaceAll('\\','/'))).status(),403);
  assert.equal((await a.request.get(base+'/config',{headers:{host:'evil.invalid'}})).status(),403);
  report.private_files_and_foreign_host_denied=true;
  assert.equal(await page.evaluate(()=>window.__avDeviceTracks.every(t=>t.readyState==='ended')),true);
  const branding=await page.locator('.av-header,.av-hero,.av-footnote,.tab-nav').allTextContents();
  assert.ok(!/Applied AI Research|Private Model Demonstration|Private Workspace/i.test(branding.join(' ')));
  report.branding_verified=true;
  assert.deepEqual(errors,[]);report.browser_errors=errors;report.browser_version=await browser.version();
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({upload_score:report.upload.score,live_score:report.live.score,checks: Object.keys(report).filter(k=>report[k]===true)},null,2));
  await b.close();await a.close();
 }catch(e){let text='';try{text=await activePage.locator('body').innerText();await activePage.screenshot({path:path.join(out,'failure.png'),fullPage:true});}catch{}fs.writeFileSync(path.join(out,'failure.txt'),String(e)+'\n'+text);throw e;}
 finally{if(browser)await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
