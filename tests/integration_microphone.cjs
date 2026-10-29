// Focused microphone acceptance. Every browser uses synthetic camera/audio.
// No physical-device mode, raw audio logging, upload tests or model stubs.
const fs=require('node:fs'), path=require('node:path'), assert=require('node:assert/strict');
const {chromium}=require(path.join(process.cwd(),'_local/premium_ui/browser/node_modules/playwright'));
if(process.argv.includes('--physical'))throw Error('Physical devices are not authorized');
const base=process.env.AVAL_MIC_TEST_URL||'http://127.0.0.1:7864';
const privateDir=path.resolve(process.env.AVAL_MIC_PRIVATE_DIR||'_local/three_head_demo/private');
const out=path.resolve('_local/microphone_checks/browser');fs.mkdirSync(out,{recursive:true});
const login=JSON.parse(fs.readFileSync(path.join(privateDir,'login.json'),'utf8'));

// Explicit test fixtures, never substituted into a physical-device capture.
function wav(name, amplitude, zeroRange){
  const rate=48000,n=rate*20,bytes=Buffer.alloc(44+n*2);
  bytes.write('RIFF');bytes.writeUInt32LE(36+n*2,4);bytes.write('WAVEfmt ',8);
  bytes.writeUInt32LE(16,16);bytes.writeUInt16LE(1,20);bytes.writeUInt16LE(1,22);
  bytes.writeUInt32LE(rate,24);bytes.writeUInt32LE(rate*2,28);bytes.writeUInt16LE(2,32);
  bytes.writeUInt16LE(16,34);bytes.write('data',36);bytes.writeUInt32LE(n*2,40);
  for(let i=0;i<n;i++)bytes.writeInt16LE(zeroRange&&i>=zeroRange[0]*rate&&i<zeroRange[1]*rate?0:
    Math.round(amplitude*Math.sin(2*Math.PI*440*i/rate)),44+i*2);
  const file=path.join(out,name);fs.writeFileSync(file,bytes);return file;
}
const quiet=wav('synthetic-quiet.wav',4),silence=wav('synthetic-silence.wav',0);
const zeroInterval=wav('synthetic-zero-interval.wav',400,[3,7]);
const reports=[];
(async()=>{
  for(const scenario of ['quiet','zero_interval','tiny_tail','zero','disabled','denied']){
    const browser=await chromium.launch({channel:'chrome',headless:true,args:[
      '--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
      '--use-file-for-fake-audio-capture='+(scenario==='zero'?silence:scenario==='zero_interval'?zeroInterval:quiet)]});
    try{
      const context=await browser.newContext({permissions:['camera','microphone']});
      await context.addInitScript(scenario=>{
        window.__micTestTracks=[];
        if(scenario==='tiny_tail'){
          // Synthetic attenuation at the native graph input recreates the
          // reported nonzero-float/all-zero-PCM16 tail. Physical inputs are
          // never used by this harness; production gain is never changed.
          const NativeContext=window.AudioContext;
          window.AudioContext=function(...args){
            const ctx=new NativeContext(...args),createSource=ctx.createMediaStreamSource.bind(ctx);
            ctx.createMediaStreamSource=stream=>{
              const source=createSource(stream),gain=ctx.createGain();source.connect(gain);
              const connect=gain.connect.bind(gain);
              gain.connect=(...targets)=>{
                gain.gain.setValueAtTime(1,ctx.currentTime);
                gain.gain.setValueAtTime(1e-6,ctx.currentTime+7.9);
                return connect(...targets);
              };return gain;
            };return ctx;
          };
          window.AudioContext.prototype=NativeContext.prototype;
        }
        const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia=async constraints=>{
          if(scenario==='denied')throw new DOMException('Synthetic permission-denial test','NotAllowedError');
          const stream=await original(constraints);window.__micTestTracks.push(...stream.getTracks());
          if(scenario==='disabled')stream.getAudioTracks()[0].enabled=false;
          return stream;
        };
      },scenario);
      const page=await context.newPage(),errors=[];let seals=0,lastAccepted=0;
      page.on('pageerror',e=>errors.push(e.message));
      page.on('response',async response=>{
        if(!response.url().includes('component_server'))return;
        try{
          const d=await response.json();
          if(d.status==='RESEARCH_CAPTURE_SEALED')seals++;
          if(d.capture_progress)lastAccepted=d.capture_progress.accepted_audio_samples;
        }catch{}
      });
      await page.goto(base);await page.getByLabel('username').fill(login.username);
      await page.getByLabel('password').fill(login.password);await page.getByRole('button',{name:'Login',exact:true}).click();
      await page.getByRole('tab',{name:'Live Verification',exact:true}).click();
      const live=page.locator('.av-live').last();await live.locator('.av-research-consent').check();
      await live.locator('.av-record').click();
      await page.waitForFunction(()=>{
        const e=document.querySelector('.av-mic-diag');if(!e?.textContent)return false;
        const d=JSON.parse(e.textContent);return ['verified','invalid'].includes(d.phase);
      },{},{timeout:90000});
      const diagnostic=JSON.parse(await live.locator('.av-mic-diag').textContent());
      const message=await live.locator('.av-msg').textContent();
      assert.equal(await page.evaluate(()=>window.__micTestTracks.every(t=>t.readyState==='ended')),true);
      assert.deepEqual(errors,[]);
      if(['quiet','zero_interval','tiny_tail','zero'].includes(scenario)){
        assert.equal(diagnostic.continuity.next_sample_index,160000);
        assert.equal(diagnostic.continuity.discontinuities,0);
        assert.equal(diagnostic.continuity.all_emitted_samples_explicitly_written,true);
        assert.equal(diagnostic.continuity.acquisition_complete,true);
        assert.equal(diagnostic.continuity.finalized_before_acquisition_complete,false);
        assert.equal(diagnostic.server_seconds.length,10);
        assert.ok(diagnostic.seconds.every(s=>s.samples===16000));
        assert.ok(diagnostic.browser_server_comparison.every(s=>s.complete&&s.received_counts_agree&&
          s.nonzero_counts_agree&&s.sample_clock_bounds_agree&&s.rms_difference<1e-12&&s.peak_abs_difference===0));
        assert.equal(diagnostic.sampled_video_frames.length,10);
        assert.ok(diagnostic.sampled_video_frames.every(s=>Math.abs(s.offset_ms)<=100));
      }
      if(scenario==='quiet'){
        assert.equal(diagnostic.phase,'verified');assert.equal(diagnostic.category,'nonzero_pcm_received');
        assert.equal(diagnostic.received_samples,160000);assert.equal(lastAccepted,160000);assert.equal(seals,1);
        assert.ok(diagnostic.web_audio_rms>0&&diagnostic.web_audio_rms<.001);
        assert.ok(diagnostic.pcm16_nonzero_samples>0);assert.equal(diagnostic.capture_context_state,'running');
        assert.equal(diagnostic.current_context_state,'closed');assert.equal(diagnostic.current_track_ready_state,'ended');
        assert.equal(diagnostic.track.enabled,true);assert.equal(diagnostic.track.muted,false);
        assert.equal(await live.locator('[data-head]').count(),3);
      }else{
        assert.equal(diagnostic.phase,'invalid');assert.equal(seals,0);
        assert.equal(await live.locator('[data-head]').count(),0);
        if(scenario==='zero_interval'||scenario==='tiny_tail'){
          const zeros=diagnostic.zero_pcm_seconds_reported_by_server;
          assert.ok(zeros.length>0);assert.deepEqual(zeros,diagnostic.zero_pcm_seconds_at_worklet_input);
          assert.equal(diagnostic.received_samples,160000);assert.equal(lastAccepted,160000);
          for(const second of zeros){
            assert.ok(message.includes('second '+second)||message.includes('second(s) '+second));
            assert.equal(diagnostic.seconds[second].pcm16_nonzero_samples,0);
            assert.equal(diagnostic.server_seconds[second].nonzero_samples,0);
            assert.equal(diagnostic.server_seconds[second].received_samples,16000);
          }
          if(scenario==='tiny_tail'){
            assert.deepEqual(zeros,[8,9]);assert.equal(diagnostic.category,'pcm16_quantized_zero_seconds');
            assert.ok(diagnostic.seconds[8].web_audio_rms>0&&diagnostic.seconds[9].web_audio_rms>0);
            assert.ok(message.includes('PCM16 resolution'));
          }else assert.ok(diagnostic.seconds.some(s=>s.web_audio_nonzero_samples===0&&s.samples===16000));
        }else if(scenario==='zero'){
          assert.equal(diagnostic.category,'all_zero_pcm');assert.equal(diagnostic.received_samples,160000);
          assert.equal(diagnostic.server_accepted_audio_samples,160000);
          assert.equal(diagnostic.web_audio_rms,0);assert.equal(diagnostic.pcm16_nonzero_samples,0);
          assert.deepEqual(diagnostic.zero_pcm_seconds_reported_by_server,[0,1,2,3,4,5,6,7,8,9]);
          assert.ok(message.includes('LIVE_WINDOW_MUTED_AUDIO'));
        }else{
          assert.equal(diagnostic.received_samples,0);
          assert.equal(diagnostic.category,scenario==='denied'?'media_permission_denied':'microphone_disabled');
        }
      }
      assert.ok(diagnostic.events.length<=16);
      const text=JSON.stringify(diagnostic);
      for(const sensitive of [login.password,'deviceId','groupId','cancel_key','tab_key','"pcm":'])assert.ok(!text.includes(sensitive));
      reports.push({scenario,simulated_devices_only:true,seals,diagnostic,browser_errors:errors});
      await live.locator('.av-reset').click();assert.equal(await live.locator('.av-mic-diag').textContent(),'');
      fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(reports,null,2));
      console.log(JSON.stringify({scenario,passed:true,category:diagnostic.category,samples:diagnostic.received_samples,seals,
        zero_seconds:diagnostic.zero_pcm_seconds_reported_by_server}));
    }finally{await browser.close();}
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
