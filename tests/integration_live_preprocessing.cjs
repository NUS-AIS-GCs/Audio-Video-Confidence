// Controlled Chromium preprocessing only; no webcam or microphone access.

const {chromium}=require(require('node:path').join(process.cwd(),'_local/premium_ui/browser/node_modules/playwright'));
const fs=require('node:fs'),path=require('node:path');
const out=path.join(process.cwd(),'_local/live_integration');
(async()=>{const browser=await chromium.launch({channel:'chrome',headless:true});try{
const page=await browser.newPage();await page.goto('about:blank');
await page.addScriptTag({content:fs.readFileSync(path.join(process.cwd(),'deployment/web/live_timing.js'),'utf8')});
const audio=await page.evaluate(async b64=>{
 const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));const input=new Float32Array(bytes.buffer);
 const context=new OfflineAudioContext(1,160000,16000);const buffer=context.createBuffer(2,480000,48000);
 for(let ch=0;ch<2;ch++){const a=buffer.getChannelData(ch);for(let i=0;i<a.length;i++)a[i]=input[2*i+ch];}
 const source=context.createBufferSource();source.buffer=buffer;
 const mono=context.createGain();mono.channelCount=1;mono.channelCountMode='explicit';mono.channelInterpretation='speakers';
 source.connect(mono);mono.connect(context.destination);source.start();const rendered=await context.startRendering();
 const samples=rendered.getChannelData(0);const pcm=AVTiming.floatToPcm16(samples);
 const vals=new Float32Array([1.2,-1.2,1,-1,0.5/32768,-0.5/32768,1.5/32768,-1.5/32768]);
 return {float32:AVTiming.bytesToBase64(new Uint8Array(samples.buffer)),pcm16:AVTiming.bytesToBase64(pcm),
 clip_round_probe_input:Array.from(vals),clip_round_probe_pcm:AVTiming.bytesToBase64(AVTiming.floatToPcm16(vals)),
 context_rate:context.sampleRate,input_rate:buffer.sampleRate,input_channels:buffer.numberOfChannels};
},fs.readFileSync(path.join(out,'stereo-48000.f32')).toString('base64'));
fs.writeFileSync(path.join(out,'browser-audio-f32.bin'),Buffer.from(audio.float32,'base64'));
fs.writeFileSync(path.join(out,'browser-audio-s16.bin'),Buffer.from(audio.pcm16,'base64'));
delete audio.float32;delete audio.pcm16;
const frames=[];
for(const i of [0,2,...Array.from({length:10},(_,k)=>"live-"+k)]){
 const frame=await page.evaluate(async data=>{
  const image=new Image();image.src='data:image/png;base64,'+data;await image.decode();
  const canvas=document.createElement('canvas');const scale=Math.min(1,640/Math.max(image.width,image.height));
  canvas.width=Math.round(image.width*scale);canvas.height=Math.round(image.height*scale);
  const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0,canvas.width,canvas.height);
  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const small=document.createElement('canvas');small.width=small.height=224;
  small.getContext('2d').drawImage(canvas,0,0,224,224);
  return {width:canvas.width,height:canvas.height,jpeg:canvas.toDataURL('image/jpeg',.9).split(',')[1],
   rgba:AVTiming.bytesToBase64(new Uint8Array(pixels.buffer)),
   browser_resize_rgba:AVTiming.bytesToBase64(new Uint8Array(small.getContext('2d').getImageData(0,0,224,224).data.buffer))};
 },fs.readFileSync(path.join(out,`source-${i}.png`)).toString('base64'));
 fs.writeFileSync(path.join(out,`browser-${i}.jpg`),Buffer.from(frame.jpeg,'base64'));
 fs.writeFileSync(path.join(out,`browser-${i}.rgba`),Buffer.from(frame.rgba,'base64'));
 fs.writeFileSync(path.join(out,`browser-${i}-resize.rgba`),Buffer.from(frame.browser_resize_rgba,'base64'));
 delete frame.jpeg;delete frame.rgba;delete frame.browser_resize_rgba;frames.push({source:i,...frame});
}
fs.writeFileSync(path.join(out,'browser-comparison-profile.json'),JSON.stringify({browser:await browser.version(),audio,frames},null,2));
console.log('Actual Chromium audio resampling/mixing, PCM and canvas/JPEG fixtures measured.');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
