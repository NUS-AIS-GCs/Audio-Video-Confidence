// Synthetic presentation fixtures; genuine model evidence comes from browser acceptance.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const engine = require('../../deployment/web/live_research_result.js');
const {spawnSync} = require('node:child_process');
const localPython = path.join(process.cwd(), '_local/phase8_stage1/venv/Scripts/python.exe');
const python = process.env.AVAL_TEST_PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
const renderer = {render(element, result) {
  let result_view;
  if(Number.isFinite(result.score)) {
    const run = spawnSync(python, ['-B','-c', 'import sys,json; from deployment.ui.result_sections import ThreeHeadResults; print(json.dumps(ThreeHeadResults.view(json.load(sys.stdin, parse_int=float))))'],
      {input:JSON.stringify(result),encoding:'utf8',windowsHide:true,env:{...process.env,PYTHONIOENCODING:'utf-8'}});
    assert.equal(run.status,0,run.stderr); result_view=JSON.parse(run.stdout);
  }
  engine.render(element, {...result,result_view});
}};

function document() {
  const doc = {createElement(tag) {return new Node(tag);}, createTextNode(text) {const n=new Node('#text');n.textContent=text;return n;}};
  class Node {
    constructor(tag) {this.tagName=tag;this.children=[];this.attributes={};this.ownerDocument=doc;this.text='';}
    appendChild(node) {this.children.push(node);return node;}
    setAttribute(name,value) {this.attributes[name]=value;if(name==='class')this.className=value;}
    replaceChildren(...nodes) {this.text='';this.children=nodes;}
    set textContent(value) {this.text=String(value);this.children=[];}
    get textContent() {return this.text+this.children.map(c=>c.textContent).join('');}
    set innerHTML(_) {throw Error('Untrusted HTML must never be inserted');}
  }
  return doc;
}
function rows(node) {return [node,...node.children.flatMap(rows)];}
function fixture() {
  const names=JSON.parse(fs.readFileSync(path.join(__dirname,'../../deployment/event_class_names.json')));
  const scores=Array.from({length:10},()=>Array.from({length:29},(_,i)=>Math.fround(i===28?.8:i<2?.1:0)));
  return {schema:'aval.live-input-research.v1',status:'EXPERIMENTAL_LIVE_INPUT_NOT_APPROVED',
    classification:'Original-like pairing',score:.5,threshold:0.009579099009671126,
    model_version:'E3 / frozen concatenation + MLP',processing_seconds:1.25,temporal_predictions:null,
    event_predictions:{class_names:names,audio:{scores},video:{scores}}};
}

test('live result has three objectives and no fabricated temporal head',()=>{
  const element=document().createElement('div');renderer.render(element,fixture());
  assert.deepEqual(rows(element).filter(n=>n.attributes['data-head']).map(n=>n.attributes['data-head']),['events','temporal','correspondence']);
  assert.ok(element.textContent.includes('verified artifact not loaded'));
  assert.equal(rows(element).filter(n=>n.tagName==='tbody')[0].children.length,10);
  assert.equal(rows(element).filter(n=>n.tagName==='meter').length,10);
  const raw=rows(element).find(n=>'data-av-raw-result' in n.attributes);
  assert.equal(JSON.parse(raw.textContent).temporal_predictions,null);
});

test('ranked scores retain original scale and background remains visible',()=>{
  const element=document().createElement('div');renderer.render(element,fixture());
  const list=rows(element).find(n=>n.className==='av-event-list');
  assert.equal(list.children[0].children[0].textContent,'Accordion');
  assert.equal(list.children[1].children[0].textContent,'Acoustic guitar');
  assert.equal(list.children[0].children[1].textContent,'0.100000');
  assert.ok(element.textContent.includes('Mean background score: 0.800000'));
  assert.ok(element.textContent.includes('not a fused temporal-presence prediction'));
});

test('event labels and diagnostics are inserted as text without an HTML sink',()=>{
  const element=document().createElement('div'),result=fixture();
  result.event_predictions.class_names[0]='<img src=x onerror=alert(1)>';
  renderer.render(element,result);
  assert.ok(element.textContent.includes('<img src=x onerror=alert(1)>'));
  assert.ok(!rows(element).some(n=>n.tagName==='img'||n.tagName==='script'));
});

test('fresh results replace all three sections and invalid scores clear prior evidence',()=>{
  const element=document().createElement('div');renderer.render(element,fixture());renderer.render(element,fixture());
  assert.equal(rows(element).filter(n=>n.attributes['data-head']).length,3);
  renderer.render(element,{...fixture(),score:NaN});
  assert.equal(element.children.length,0);
  assert.ok(!element.textContent.includes('Accordion'));
  assert.match(element.textContent,/UNABLE TO VERIFY/);
});

test('missing encoder tensors are explicitly unavailable while E3 is retained',()=>{
  const element=document().createElement('div');renderer.render(element,{...fixture(),event_predictions:null});
  assert.ok(element.textContent.includes('Encoder event evidence is not available'));
  assert.ok(element.textContent.includes('Actual raw E3 score'));
});

test('a trained temporal result renders ten predictions with its separate threshold',()=>{
  const element=document().createElement('div'),result=fixture();
  result.temporal_predictions={model_version:'unit trained-head schema',threshold:.25,
    scores:Array.from({length:10},(_,i)=>i%2?.1:.9),presence:Array.from({length:10},(_,i)=>i%2===0)};
  renderer.render(element,result);
  const cells=rows(element).filter(n=>n.className==='av-presence-cell');
  assert.equal(cells.length,10);
  assert.deepEqual(cells.map(n=>n.attributes['data-state']),['present','background','present','background','present','background','present','background','present','background']);
  assert.ok(element.textContent.includes('Head 2 threshold: 0.25'));
  assert.ok(!element.textContent.includes('verified artifact not loaded'));
  assert.equal(result.threshold,0.009579099009671126);
});

test('shared live view rejects executable tags and attributes and clears old evidence',()=>{
  const element=document().createElement('div');
  for(const result_view of [
    {tag:'script',attrs:{},children:['alert(1)']},
    {tag:'div',attrs:{onclick:'alert(1)'},children:[]}
  ]){
    element.textContent='old evidence';engine.render(element,{score:.5,threshold:.01,result_view});
    assert.match(element.textContent,/UNABLE TO VERIFY/);assert.ok(!element.textContent.includes('old evidence'));
  }
});
