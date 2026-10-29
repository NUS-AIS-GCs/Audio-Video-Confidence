// Functional acceptance on the separately launched private demo. Fake devices ONLY.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(path.join(process.cwd(),'_local/premium_ui/browser/node_modules/playwright'));
if(process.argv.includes('--physical'))throw Error('Physical devices are not authorized');
const root=process.cwd(),base=process.env.AVAL_PRODUCT_TEST_URL||'http://127.0.0.1:7862';
const privateDir=path.join(root,'_local/product_demo/private'),out=path.join(root,'_local/product_demo/evidence');
fs.mkdirSync(out,{recursive:true});
const login=JSON.parse(fs.readFileSync(path.join(privateDir,'login.json'),'utf8'));
const source=path.join(root,'_local/phase8_stage2c/models/video/parity_samples/c---zaDCTaE.mp4');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const report={fake_devices_only:true};let browser,activePage;
const save=()=>fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));
(async()=>{
 try{
  browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  const a=await browser.newContext({permissions:['camera','microphone'],viewport:{width:1440,height:1080}});
  await a.route('**/*',r=>{const u=new URL(r.request().url());return ['http:','https:'].includes(u.protocol)&&!['127.0.0.1','localhost'].includes(u.hostname)?r.abort():r.continue();});
  const page=await a.newPage(),completed=[],errors=[];let sealed;activePage=page;
  page.on('pageerror',e=>errors.push(e.message));
  page.on('requestfinished',async req=>{
   try{
    const u=new URL(req.url());
    if(u.pathname.includes('/queue/data')){
     const text=await (await req.response()).text();
     for(const line of text.split('\n'))if(line.startsWith('data: ')){
      const r=JSON.parse(line.slice(6));
      if(r.msg==='process_completed'&&r.output?.data?.[1]?.schema==='aval.general-media-research.v1')completed.push(r.output.data[1]);
     }
    }
    if(u.pathname.includes('component_server')){
     const r=await(await req.response()).json();if(r.status==='RESEARCH_CAPTURE_SEALED')sealed={ticket:r.ticket,cancel_key:r.cancel_key};
    }
   }catch{}
  });
  await page.goto(base,{waitUntil:'networkidle'});
  await page.getByLabel('username').fill(login.username);await page.getByLabel('password').fill(login.password);
  await page.getByRole('button',{name:'Login',exact:true}).click();
  await page.getByRole('tab',{name:'Verify uploaded video',exact:true}).click();
  const file=page.locator('input[type=file]').first(),card=page.locator('#av-general-result');
  const selectFile=async value=>{
    const component=page.locator('#av-general-research-source');
    const clear=component.getByRole('button',{name:'Clear',exact:true});
    if(await clear.count())await clear.click();
    await component.locator('input[type=file]').setInputFiles(value);
  };
  const readUpload=async()=>{
    const text=await page.locator('#av-general-response').innerText(),result={};
    // The JSON widget collapses long diagnostic arrays to "Array(10)". Read
    // displayed scalar values; no hidden state, score fixture or model stub.
    for(const key of ['schema','status','request_id','source_sha256','profile_sha256','score','logit','processing_seconds',
      'threshold','model_sha256','checkpoint_sha256','classification','score_semantics','production_predictions_enabled','predictions','error_code']){
      const match=text.match(new RegExp('"'+key+'"\\s*:\\s*("[^"\\n]*"|null|true|false|-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'));
      if(match)result[key]=JSON.parse(match[1]);
    }
    assert.ok(!text.includes('"cancel_key"')&&!text.includes('"owner"')&&!text.includes('"_audit"'));
    return result;
  };
  await selectFile(source);await page.getByRole('button',{name:'Verify',exact:true}).click();
  await page.waitForFunction(()=>{const t=document.querySelector('#av-general-result')?.textContent||'';return t.includes('Actual raw E3 score')||t.includes('Verification unavailable');},{},{timeout:180000});
  assert.ok((await card.innerText()).includes('Actual raw E3 score'),await card.innerText());
  report.upload=await readUpload();assert.equal(report.upload.status,'EXPERIMENTAL_GENERAL_MEDIA_NOT_APPROVED');
  assert.equal(report.upload.threshold,0.009579099009671126);assert.equal(report.upload.production_predictions_enabled,false);
  for(const key of ['_audit','ticket','owner','cancel_key'])assert.ok(!(key in report.upload));
  save();
  await page.screenshot({path:path.join(out,'upload.png'),fullPage:true});
  await selectFile(path.join(root,'_local/product_demo/fixtures/cyclic-negative.mp4'));
  await page.getByRole('button',{name:'Verify',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#av-general-result')?.textContent.includes('Actual raw E3 score'),{},{timeout:180000});
  report.negative_upload=await readUpload();
  assert.equal(report.negative_upload.classification,'Audio-substituted-like pairing');
  assert.ok(report.negative_upload.score<report.negative_upload.threshold);save();
  await page.screenshot({path:path.join(out,'negative-upload.png'),fullPage:true});
  await selectFile({name:'invalid.mp4',mimeType:'video/mp4',buffer:Buffer.from('invalid media')});
  await page.getByRole('button',{name:'Verify',exact:true}).click();
  await card.getByRole('heading',{name:'Verification unavailable'}).waitFor({timeout:60000});
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
  save();

  await page.getByRole('tab',{name:'Verify live capture',exact:true}).click();
  const live=page.locator('.av-live').last();
  await live.locator('.av-record').click();assert.ok((await live.locator('.av-msg').innerText()).includes('Consent'));
  await live.locator('.av-research-consent').check();await live.locator('.av-record').click();
  await page.waitForFunction(()=>document.querySelector('.av-research-result')?.textContent.includes('EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED')||document.querySelector('.av-msg')?.dataset.kind==='error',{}, {timeout:180000});
  const display=await live.locator('.av-research-result').innerText();
  assert.ok(display.includes('EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED'),await live.innerText());
  const result=JSON.parse(display.slice(display.indexOf('{')));
  report.live={score:result.score,logit:result.logit,classification:result.classification,threshold:result.threshold,
    checkpoint_sha256:result.checkpoint_sha256,model_sha256:result.model_sha256,profile_sha256:result.profile_sha256,
    processing_seconds:result.processing_seconds,timing:result.live_input.timing};
  assert.equal(result.production_predictions_enabled,false);assert.ok(sealed);
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
  assert.equal((await a.request.get(base+'/config',{headers:{host:'evil.invalid'}})).status(),403);
  report.private_files_and_foreign_host_denied=true;
  assert.deepEqual(errors,[]);report.browser_errors=errors;report.browser_version=await browser.version();
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({upload_score:report.upload.score,live_score:report.live.score,checks: Object.keys(report).filter(k=>report[k]===true)},null,2));
  await b.close();await a.close();
 }catch(e){let text='';try{text=await activePage.locator('body').innerText();await activePage.screenshot({path:path.join(out,'failure.png'),fullPage:true});}catch{}fs.writeFileSync(path.join(out,'failure.txt'),String(e)+'\n'+text);throw e;}
 finally{if(browser)await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
