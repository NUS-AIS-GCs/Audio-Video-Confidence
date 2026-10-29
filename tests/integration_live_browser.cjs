// Opt-in: fake devices by default. --physical consumes one pre-authorized capture attempt.

const {chromium}=require(require('node:path').join(process.cwd(),'_local/premium_ui/browser/node_modules/playwright'));
const {spawn,spawnSync}=require('node:child_process');
const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const crypto=require('node:crypto');
const physical=process.argv.includes('--physical');
const root=process.cwd();
// Physical mode always retains the original one-attempt marker directory.
// A new fixture evidence directory cannot bypass that guard.
const out=!physical && process.env.AVAL_RESEARCH_TEST_OUTPUT ?
  path.resolve(process.env.AVAL_RESEARCH_TEST_OUTPUT) : path.join(root,'_local/live_integration');
fs.mkdirSync(out,{recursive:true});
const privateDir=path.join(out,physical?'private-physical':'private'),spool=path.join(privateDir,'spool');
const python=path.join(root,'_local/phase8_stage1/venv/Scripts/python.exe');
const docker=path.join(process.env.LOCALAPPDATA,'Programs/DockerDesktop/resources/bin/docker.exe');
const researchImage=process.env.AVAL_RESEARCH_IMAGE || 'avalidate:live-research-candidate';
const password=crypto.randomBytes(24).toString('hex');
const prefix=physical?'physical-':'';
const serverlog=fs.openSync(path.join(out,prefix+'capture-server.log'),'w');
const worklog=fs.openSync(path.join(out,prefix+'replay-worker.log'),'w');
const base='http://127.0.0.1:7861';
const env={...process.env,AVAL_ACCESS_MODE:'gradio_auth',AVAL_GRADIO_USERNAME:'local-research',AVAL_GRADIO_PASSWORD:password,
 GRADIO_ANALYTICS_ENABLED:'False',AVAL_MODEL_ROOT:path.join(out,'empty-models'),GRADIO_TEMP_DIR:path.join(out,'broker-temp')};
fs.mkdirSync(env.AVAL_MODEL_ROOT,{recursive:true});fs.mkdirSync(env.GRADIO_TEMP_DIR,{recursive:true});
const broker=spawn(python,['-B','-m','deployment.live_research_app','--private-dir',privateDir,'--port','7861'],{cwd:root,env,windowsHide:true,stdio:['ignore',serverlog,serverlog]});
let browser,worker,sealedCapability;const checks={};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 try{
  for(let i=0;i<160;i++){
   try{if((await fetch(base+'/config')).status===401)break;}catch{}
   if(broker.exitCode!==null)throw Error('Broker stopped: '+fs.readFileSync(path.join(out,prefix+'capture-server.log'),'utf8'));
   await sleep(500);
  }
  const args=['run','--rm','--name','aval-live-browser-worker','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
   '--memory','4g','--memory-swap','4g','--cpus','2','--pids-limit','128','--tmpfs','/tmp:rw,nosuid,size=512m','-e','PYTHONPATH=/app',
   '--mount',`type=bind,source=${root}/_local/phase8_stage2c/models,target=/models,readonly`,
   '--mount',`type=bind,source=${spool},target=/spool`,
   '--mount',`type=bind,source=${privateDir}/research.key,target=/research.key,readonly`,
   researchImage,'python','-B','-m','deployment.live.replay','--models','/models','--spool','/spool','--key-file','/research.key','--watch-seconds','240'];
  worker=spawn(docker,args,{cwd:root,windowsHide:true,stdio:['ignore',worklog,worklog]});
  for(let i=0;i<360;i++){
   if(fs.readFileSync(path.join(out,prefix+'replay-worker.log'),'utf8').includes('LIVE_REPLAY_WORKER_READY'))break;
   if(worker.exitCode!==null)throw Error('Worker stopped: '+fs.readFileSync(path.join(out,prefix+'replay-worker.log'),'utf8'));
   await sleep(500);
  }
  if(!fs.readFileSync(path.join(out,prefix+'replay-worker.log'),'utf8').includes('LIVE_REPLAY_WORKER_READY'))throw Error('Worker startup timeout');
  checks.worker_ready=true;
  browser=await chromium.launch({channel:'chrome',headless:true,args:physical?[]:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  const context=await browser.newContext({permissions:['camera','microphone'],viewport:{width:1440,height:1100}});
  await context.route('**/*',route=>{const u=new URL(route.request().url());return ['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)?route.abort():route.continue();});
  const page=await context.newPage();global.activePage=page;const errors=[],diagnosticFailures=[],diagnosticTasks=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{const task=(async()=>{
   if(new URL(r.url()).pathname.includes('component_server')){
    try{const body=await r.json();const request=r.request().postDataJSON();
     fs.appendFileSync(path.join(out,prefix+'component-diagnostics.jsonl'),JSON.stringify({
       at_utc:new Date().toISOString(),fn_name:request.fn_name,status:r.status(),result:body.status,
       error:body.error_code,ready:body.ready,capture_progress:body.capture_progress,
       capture_diagnostics:body.capture_diagnostics})+'\n');
     if(body.status==='RESEARCH_CAPTURE_SEALED'){
      checks.ticket=body.ticket;sealedCapability={ticket:body.ticket,cancel_key:body.cancel_key};
      const src=path.join(spool,body.ticket+'.capture.json');
      if(!physical && fs.existsSync(src))fs.copyFileSync(src,path.join(out,'fixture-capture.json'));
     }
    }catch{diagnosticFailures.push('COMPONENT_DIAGNOSTIC_RECORD_FAILED');}
   }
  })();diagnosticTasks.push(task);});
  await page.goto(base,{waitUntil:'networkidle'});
  await page.getByLabel('username').fill('local-research');await page.getByLabel('password').fill(password);
  await page.getByRole('button',{name:'Login',exact:true}).click();
  await page.getByRole('tab',{name:/Local live research/,exact:false}).click();
  const live=page.locator('.av-live').last();
  await live.locator('.av-record').click();
  await sleep(200);checks.consent_required=(await live.locator('.av-msg').innerText()).includes('Consent');
  if(physical){
   const marker=fs.openSync(path.join(out,'physical-attempt.json'),'wx');
   fs.writeFileSync(marker,JSON.stringify({started:new Date().toISOString(),scope:'one consented local ten-second window; no retry',external_media_transmission:false}));fs.closeSync(marker);
  }
  await live.locator('.av-research-consent').check();await live.locator('.av-record').click();
  let safety;
  if(physical)safety=setTimeout(async()=>{if(!sealedCapability){try{await live.locator('.av-stop').click();}catch{}}},18000);
  await page.waitForFunction(()=>{
    const root=document.querySelector('.av-research-consent')?.closest('.av-live');
    return root?.querySelector('.av-research-result')?.textContent.includes('EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED') ||
      root?.querySelector('.av-msg')?.dataset.kind==='error';
  },{},{timeout:240000});
  if(physical)clearTimeout(safety);
  const display=await live.locator('.av-research-result').innerText();
  if(!display.includes('EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED'))
    throw Error('CAPTURE_REJECTED: '+await live.locator('.av-msg').innerText()+' '+await live.locator('.av-contract').innerText());
  const result=JSON.parse(display.slice(display.indexOf('{')));checks.ticket=result.ticket;checks.result=result;checks.browser_version=await browser.version();
  checks.page_errors=errors;checks.authenticated_browser_replay=true;
  checks.capture_deleted=!fs.existsSync(path.join(spool,checks.ticket+'.capture.json'));
  checks.public_ready_stays_closed=result.live_input.public_live_readiness.inference_enabled===false;

  assert.ok(sealedCapability,'sealed response capability was not observed');
  const cfg=await (await page.request.get(base+'/config')).json();
  const component=cfg.components.find(c=>(c.props.html_template||'').includes('av-research-consent'));
  assert.ok(component);
  const second=await browser.newContext();
  assert.equal((await second.request.post(base+'/gradio_api/component_server',{data:{component_id:component.id,session_hash:'anon',fn_name:'research_poll',data:sealedCapability}})).status(),401);
  await second.request.post(base+'/login',{form:{username:'local-research',password}});
  for(const method of ['research_poll','research_cancel']){
   const r=await second.request.post(base+'/gradio_api/component_server',{data:{component_id:component.id,session_hash:'other-login',fn_name:method,data:sealedCapability}});
   assert.equal((await r.json()).status,'REJECTED');
  }
  checks.cross_login_poll_cancel_denied=true;
  const foreign=await page.request.get(base+'/config',{headers:{host:'evil.invalid'}});assert.equal(foreign.status(),403);checks.foreign_host_denied=true;
  const origin=await page.request.post(base+'/gradio_api/component_server',{headers:{origin:'https://evil.invalid'},data:{component_id:component.id,session_hash:'origin',fn_name:'research_poll',data:sealedCapability}});
  assert.equal(origin.status(),403);checks.foreign_origin_denied=true;
  for(const route of ['file=','proxy=','stream/']){
   const r=await page.request.get(base+'/gradio_api/'+route+spool.replaceAll('\\','/')+'/'+checks.ticket+'.result.json');assert.equal(r.status(),403);
  }
  checks.private_artifact_routes_denied=true;
  await second.close();
  await live.locator('.av-reset').click();await sleep(1000);
  checks.reset_clears_result=(await live.locator('.av-research-result').innerText())==='';
  checks.reset_revokes_result=!fs.existsSync(path.join(spool,checks.ticket+'.result.json'));
  for(const k of ['consent_required','capture_deleted','public_ready_stays_closed','reset_clears_result','reset_revokes_result'])assert.equal(checks[k],true,k);
  await Promise.all(diagnosticTasks);assert.deepEqual(diagnosticFailures,[]);
  checks.sanitized_diagnostics_complete=true;
  fs.writeFileSync(path.join(out,physical?'physical-result.json':'browser-fixture-result.json'),JSON.stringify(checks,null,2));
  console.log(JSON.stringify({authenticated_browser_replay:true,score:result.score,profile:result.profile_sha256,checks:{...checks,result:undefined}},null,2));
 }catch(e){let detail='';try{detail=await global.activePage.locator('.av-live').last().innerText();}catch{}fs.writeFileSync(path.join(out,prefix+'browser-error.txt'),String(e)+'\n'+detail);throw e;}
 finally{
  if(browser)await browser.close();
  if(worker)spawnSync(docker,['stop','--time','10','aval-live-browser-worker'],{windowsHide:true,stdio:'ignore'});
  broker.kill();await sleep(500);fs.closeSync(serverlog);fs.closeSync(worklog);
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
