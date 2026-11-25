// Run with node scripts/validate_evaluation.cjs; no dependencies required.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').join(__dirname,'../combatsim/web/evaluation.js'),'utf8');
const fields=['mode','blue-model','red-model','blue-count','red-count','aircraft-per-team','single-match','seed-start','seed-count','swap','pace','seeds','action-mode','team-a','team-b','replays'];
class Node {
  constructor(tag='input'){this.tagName=tag.toUpperCase();this.type='text';this.value='';this.checked=false;this.children=[];this.classList={toggle:(name,on)=>this[name]=on};}
  get options(){return this.children;}
  add(node){this.children.push(node);}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(){}
  querySelectorAll(selector){const tag=selector.split(':')[0].toUpperCase();return this.children.flatMap(n=>[...(n.tagName===tag&&(!selector.includes(':checked')||n.checked)?[n]:[]),...n.querySelectorAll(selector)]);}
}
const state={defaults:{seed_start:2001,seed_count:5,blue_count:2,red_count:2,aircraft_per_team:2,action_mode:'sample',pace:10},catalog:[{id:'one',kind:'learned',name:'One'},{id:'two',kind:'learned',name:'Two'},{id:'rule:fix',kind:'rule',name:'Fix'}],catalog_errors:[]};
let stored=null,blocked=false;
function page(){
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);};
  for(const [id,values] of Object.entries({mode:['duel','mixed','round_robin'],'blue-model':[],'red-model':[],pace:['1','10','30','0'],'action-mode':['sample','greedy']})){
    const node=get(id);node.tagName='SELECT';for(const value of values){const o=new Node('option');o.value=value;node.add(o);}node.value=values[0]||'';
  }
  for(const id of ['single-match','swap','replays']){get(id).type='checkbox';get(id).checked=id!=='single-match';}
  get('team-a').value='チームA';get('team-b').value='チームB';
  const context=vm.createContext({bootstrap:{state},document:{getElementById:get,createElement:tag=>new Node(tag),createTextNode:text=>{const n=new Node('text');n.textContent=text;return n;}},Option:function(text,value){this.value=value;},localStorage:{getItem(){if(blocked)throw Error('blocked');return stored;},setItem(key,value){if(blocked)throw Error('blocked');stored=value;}},preview(){}});
  vm.runInContext(source.slice(0,source.indexOf('function preview()')),context);context.catalog();return {context,get};
}
let {context,get}=page();
assert.equal(get('blue-model').value,'one');assert.equal(get('red-model').value,'two');
get('mode').value='mixed';get('blue-count').value='3';context.lineups();
get('blue-lineup').querySelectorAll('select')[1].value='two';
get('blue-model').value='rule:fix';get('red-model').value='one';get('single-match').checked=true;
get('seed-start').value='';get('seeds').value='2001、 2003\n2009';get('swap').checked=false;
get('pace').value='30';get('action-mode').value='greedy';get('team-a').value='青 & <A>';get('replays').checked=false;
for(const n of get('round-models').querySelectorAll('input'))n.checked=false;
const expected=JSON.stringify(context.request());context.saveSetup();
({context,get}=page());
assert.equal(JSON.stringify(context.request()),expected,'all draft settings and the API request survive navigation');
assert.equal(get('seed-start').value,'','even an unfinished empty numeric field survives');
assert.equal(get('mixed-fields').hidden,false);assert.equal(get('duel-fields').hidden,true);
assert.equal(get('blue-lineup').querySelectorAll('select')[1].value,'two');
assert.equal(get('round-models').querySelectorAll('input:checked').length,0,'explicitly selecting no models is preserved');
state.catalog.push({id:'three',kind:'learned',name:'Three'});context.catalog();
assert.equal(get('round-models').querySelectorAll('input:checked').length,0,'catalog refresh does not silently select models');
const saved=JSON.parse(stored);saved.fields['blue-model']='deleted';saved.lineups.blue[1]='deleted';saved.fields.mode='invalid';stored=JSON.stringify(saved);
({context,get}=page());assert.equal(get('blue-model').value,'one');assert.equal(get('mode').value,'duel');assert.equal(get('blue-lineup').querySelectorAll('select')[1].value,'one');
for(const malformed of ['{','null','[]','42','{"fields":{"single-match":"false","pace":{}}}']){stored=malformed;assert.doesNotThrow(page);}
blocked=true;assert.doesNotThrow(page);
assert(fields.every(id=>source.includes(`'${id}'`)));
console.log('PASS: setup draft, model lineups, empty selections, removed models and unavailable storage');
