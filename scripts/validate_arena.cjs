// Run with node scripts/validate_arena.cjs; no dependencies required.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const scene=fs.readFileSync(path.join(__dirname,'../combatsim/web/arena-scene.js'),'utf8');
const app=fs.readFileSync(path.join(__dirname,'../combatsim/web/arena.js'),'utf8').replace(/^import .*;\r?\n/gm,'');
const context=vm.createContext({});vm.runInContext(scene+'\nglobalThis.scene=ArenaScene;',context);
const S=context.scene;
const aircraft=(x,heading=0)=>({id:'blue_0',label:'Model <one>',team:'blue',x,y:120,z:20,heading,pitch:.15,speed:1,ammo:3,alive:true});
const missile=x=>({id:'m',owner:'blue_0',team:'blue',x,y:180,z:25,heading:0,pitch:0});
const frames=[{tick:0,aircraft:[aircraft(80,3.1)],missiles:[],events:[]},
 {tick:2,aircraft:[aircraft(100,-3.1)],missiles:[missile(110)],events:[{type:'launch',aircraft:'blue_0',target:'red_0'}]},
 {tick:5,aircraft:[{...aircraft(130),alive:false}],missiles:[],events:[{type:'hit',target:'blue_0'}]}];
const before=JSON.stringify(frames),config={arena_x:800,arena_y:400,ceiling:50,aero_length_scale:250,aero_tick_seconds:1};
// Compare against the original viewer for fractional ticks and out-of-order seeks.
const motion=Array.from({length:24},(_,tick)=>({tick,aircraft:[{
  ...aircraft(80),control:{heading_change:tick<8?.12:tick<16?-.12:0,pitch_before:.15,pitch_after:.15}
}]}));
const original=fs.readFileSync(path.join(__dirname,'../combatsim/web/viewer.js'),'utf8');
const old=vm.createContext({frames:motion,config});
vm.runInContext(original.slice(original.indexOf('function color('),original.indexOf('function tintTowardWhite('))+
  original.slice(original.indexOf('function targetAircraftBank('),original.indexOf('function aircraftBasis(')),old);
const bank=S.createBankSampler(motion,config);
for(const tick of [14.5,0,7.5,8,9,10,23,3.5]){
  const index=Math.floor(tick),a=motion[index].aircraft[0];
  assert.equal(bank(a,index,tick),old.aircraftBank(a,index,tick));
}
for(const team of ['blue','red'])for(let model_color=0;model_color<8;model_color++){
  assert.equal(S.aircraftColor({team,model_color}),old.color({team,model_color}));
}
assert.equal(S.aircraftLabel(aircraft(0)),'Model <one>');
assert.equal(S.aircraftLabel({id:'blue_0'}),'blue_0');
console.log('PASS: model colors, labels and seekable roll agree with the original viewer');
assert.equal(S.sample(frames,1).frame.aircraft[0].x,90);
assert(Math.abs(S.sample(frames,1).frame.aircraft[0].heading-Math.PI)<1e-12);
assert.equal(S.sample(frames,1).frame.missiles.length,0,'missiles must not appear before launch');
assert.equal(S.sample(frames,3).frame.aircraft[0].alive,true,'destruction applies at the recorded tick');
const sensorFrames=[{...frames[0],aircraft:[{...aircraft(80),hud_contacts:[]}]},
 {...frames[1],aircraft:[{...aircraft(100),hud_contacts:[{id:'red_0',source:'radar'}]}]}];
assert.equal(S.sample(sensorFrames,1).frame.aircraft[0].hud_contacts.length,0,'interpolation never reveals a future sensor contact');
assert.equal(S.sample(sensorFrames,2).frame.aircraft[0].hud_contacts[0].source,'radar','sensor contacts change at their recorded tick');
assert.equal(S.sample(frames,5).frame.aircraft[0].alive,false);
assert.equal(S.sample(frames,5).frame.missiles.length,0);
assert.equal(S.sample(frames,500).frame.tick,5);
assert.equal(S.clock(125),'02:05');
assert.equal(JSON.stringify(frames),before);
assert.equal(JSON.stringify(S.position({x:400,y:200,z:10},config,3)),JSON.stringify([0,0,37.5]));
const telemetry=S.flightTelemetry({...aircraft(0),heading:0},{...config,aero_tick_seconds:.5});
assert.equal(telemetry.altitude,5000);assert.equal(telemetry.speed,1800);assert.equal(telemetry.heading,90);
assert.equal(S.flightTelemetry({...aircraft(0),heading:Math.PI/2},config).heading,0);
assert.equal(S.flightTelemetry({...aircraft(0),heading:Math.PI*2},config).heading,90,'HUD heading wraps at north');
const trendFrames=Array.from({length:121},(_,tick)=>({tick,aircraft:[{...aircraft(80),z:10+tick/10,speed:2-tick*.005}],missiles:[],events:[]}));
const trendBefore=JSON.stringify(trendFrames),history=S.flightHistory(trendFrames,'blue_0',90,config);
assert.equal(history.from,30);assert.equal(history.to,90);assert.equal(history.points[0].altitude,3250);assert.equal(history.points.at(-1).altitude,4750);
assert.equal(history.points.at(-1).speed,1395);assert(history.points.every(p=>p.tick<=90),'history never contains future telemetry');
const partial=S.flightHistory(trendFrames,'blue_0',90.5,config);assert.equal(partial.points[0].tick,30.5);assert.equal(partial.points.at(-1).tick,90.5,'the line reaches the displayed fractional tick');
const slowTick=S.flightHistory(trendFrames,'blue_0',100,{...config,aero_tick_seconds:2});assert.equal(slowTick.from,70,'history spans 60 simulation seconds rather than 60 ticks');
const fastTick=S.flightHistory(trendFrames,'blue_0',90,{...config,aero_tick_seconds:.5});assert.equal(fastTick.from,0);assert.equal(fastTick.points.at(-1).speed,2790,'speed history uses physical km/h');
assert.equal(S.flightHistory(trendFrames,'absent',90,config).points.length,0);
assert.equal(JSON.stringify(trendFrames),trendBefore,'history sampling never changes recorded telemetry');
for(const axis of S.aircraftBasis(aircraft(1)))assert(Math.abs(Math.hypot(...axis)-1)<1e-12);
const death={tick:5,position:frames[2].aircraft[0],type:'hit'},cues=new Map();
S.updateLossCues(cues,frames[0],[death],null,0);assert.equal(cues.size,0);
S.updateLossCues(cues,frames[2],[death],frames[0],2);
const cue=cues.get('blue_0'),impact=S.wreckState(cue,2),smoking=S.wreckState(cue,6),aftermath=S.wreckState(cue,13);
assert.equal(impact.age,0);assert.equal(impact.flash,1);assert.equal(impact.fade,0);assert(impact.visible,'a newly destroyed aircraft remains visible');
assert.equal(impact.pose.x,death.position.x);assert.equal(impact.pose.z,death.position.z,'the death sequence begins at the recorded impact position');
for(const time of [2.5,3,6,13])assert.equal(S.wreckState(cue,time).pose,death.position,'destroyed aircraft stay at the recorded impact pose without falling or spinning');
assert.equal(smoking.visible,false,'the aircraft disappears into the explosion instead of falling');assert.equal(smoking.fade,1);
assert.equal(aftermath.visible,false);assert.equal(aftermath.fade,1);
S.updateLossCues(cues,{...frames[2],tick:500},[death],frames[2],2.1);
assert(S.wreckState(cues.get('blue_0'),2.1).age<.11,'accelerated simulation cannot skip the real-time death sequence');
S.updateLossCues(cues,frames[0],[death],frames[2],14);assert.equal(cues.size,0,'rewinding clears future impact effects');
S.updateLossCues(cues,frames[2],[death],null,15);assert(S.wreckState(cues.get('blue_0'),15).age>=10,'opening an old result does not replay every explosion');
assert.equal(JSON.stringify(frames),before,'presentation motion never changes recorded flight data');
console.log('PASS: interpolation, stationary impact, fade, fast playback and rewind');

// Validate the bundled F-16 mesh and original author metadata.
const binary=fs.readFileSync(path.join(__dirname,'../combatsim/web/arena-assets/f16.glb'));
assert.equal(binary.toString('ascii',0,4),'glTF');assert.equal(binary.readUInt32LE(4),2);
const model=JSON.parse(binary.toString('utf8',20,20+binary.readUInt32LE(12)));
assert.match(model.asset.extras.license,/CC-BY-4.0/);assert.match(model.asset.extras.author,/iedalton/);
const triangles=model.meshes.reduce((sum,m)=>sum+m.primitives.reduce((sum,p)=>sum+model.accessors[p.indices].count/3,0),0);
assert(triangles>4000,'retain the complete F-16 mesh');assert.match(model.asset.extras.title,/F-16/);
assert(model.meshes.every(m=>m.primitives.every(p=>p.attributes.NORMAL!==undefined)),'retain authored surface normals');
console.log('PASS: F-16 mesh, authored normals and original license');

class Node {
 constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.dataset={};this.style={};this.attributes={};this.classes=new Set();this._text='';this.hidden=false;
 this.classList={add:(...v)=>v.forEach(n=>this.classes.add(n)),remove:(...v)=>v.forEach(n=>this.classes.delete(n)),contains:v=>this.classes.has(v),toggle:(v,on)=>on?this.classes.add(v):this.classes.delete(v)};}
 set textContent(v){this._text=String(v);this.children=[];}
 get textContent(){return this._text+this.children.map(n=>n.textContent).join('');}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this._text='';this.children=[...nodes];}
 setAttribute(k,v){this.attributes[k]=v;if(k==='class')this.className=v;}
 addEventListener(k,callback){(this.listeners??={})[k]=callback;}
 setPointerCapture(){}
 focus(){}
 getBoundingClientRect(){return {left:0,top:0,right:1000,bottom:500,width:1000,height:500};}
 showModal(){this.open=true;}
 close(){this.open=false;}
 get lastElementChild(){return this.children.at(-1);}
 querySelector(selector){const name=selector.slice(1);for(const n of this.children){if(n.className?.split(' ').includes(name))return n;const found=n.querySelector(selector);if(found)return found;}return null;}
}
const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);};get('connection').append(new Node());get('go-live').append(new Node());
const document={getElementById:get,createElement:tag=>new Node(tag),createElementNS:(_,tag)=>new Node(tag),createTextNode:text=>{const n=new Node();n.textContent=text;return n;},querySelectorAll:()=>[]};
document.body=new Node('body');document.documentElement={};
const audioCalls={enabled:[],active:[],effects:[],volume:[],disposed:0};
const audioStub={enable:async value=>audioCalls.enabled.push(value),setActive:value=>audioCalls.active.push(value),play:(...args)=>audioCalls.effects.push(args),volume:(...args)=>audioCalls.volume.push(args),dispose:()=>audioCalls.disposed++};
const ui=vm.createContext({document,ArenaScene:S,createArenaAudio:()=>audioStub,matchMedia:()=>({matches:false}),performance:{now:()=>100},setTimeout(){},clearTimeout(){}});
vm.runInContext(app.slice(0,app.indexOf('// UI bindings')),ui);
const consume=state=>ui.consume({...state,config,seed:12,models:{blue:{policy:'<img src=x>'}}},true);
consume({match_id:'one',new_frames:frames,frame:frames[2],status:'completed',done:true,winner:'red'});
assert.equal(get('blue-name').textContent,'<img src=x>');assert.equal(get('blue-name').children.length,0,'model labels remain literal text');
assert.equal(get('result-title').textContent,'RED WIN');assert(get('result').classes.has('show'));
assert.equal(get('blue-feeds').children.length,1);assert.equal(get('red-feeds').children.length,0);
assert(!get('blue-feeds').children[0].classes.has('lost'),'a hit does not instantly black out its window');
assert.match(get('blue-feeds').children[0].querySelector('.feed-data').attributes['aria-label'],/^LAST TELEMETRY/);
assert.equal(get('blue-feeds').children[0].querySelector('.feed-name').textContent,'Model <one>','aircraft labels remain literal text');
let blueFeed=get('blue-feeds').children[0];
assert.equal(blueFeed.dataset.view,'chase');assert.equal(blueFeed.querySelector('.feed-hud').hidden,true);
blueFeed.querySelector('.feed-modes').children[1].onclick();
assert.equal(blueFeed.dataset.view,'hud');assert.equal(blueFeed.querySelector('.feed-hud').hidden,false);
assert.equal(blueFeed.querySelector('.feed-modes').children[1].textContent,'HSD','existing HUD preferences select the renamed HSD panel');
assert.equal(blueFeed.querySelector('.feed-modes').children[1].attributes['aria-pressed'],'true');
assert.match(blueFeed.attributes['aria-label'],/BVR戦術HSD/);
blueFeed.querySelector('.feed-hud').setAttribute('aria-label','BVR戦術HSD · RMAX 55.0 KM');ui.hud(frames[0]);
assert.match(blueFeed.querySelector('.feed-hud').attributes['aria-label'],/RMAX 55.0/,'telemetry refresh never overwrites the tactical renderer description');
consume({match_id:'one',new_frames:[],frame:frames[2],status:'completed',done:true,winner:'red'});
assert.equal(vm.runInContext('effects.length',ui),1,'repeated polling never duplicates a hit');
ui.hud(frames[0]);assert.match(get('blue-feeds').children[0].querySelector('.feed-data').textContent,/KM\/H/,'rewinding restores live telemetry');assert(!get('result').classes.has('show'),'no result is shown before the end');
consume({match_id:'two',new_frames:[frames[0]],frame:frames[0],status:'running',done:false,winner:null});
assert.equal(get('blue-feeds').children[0].dataset.view,'hud','the next match retains the mode of the same aircraft slot');
assert.equal(vm.runInContext('effects.length',ui),0,'next match clears previous effects');assert(!get('result').classes.has('show'));
consume({match_id:'three',new_frames:frames,frame:frames[2],status:'stopped',done:true,winner:null});
assert(!get('result').classes.has('show'),'interruption is never announced as a draw');
assert.equal(get('phase').textContent,'対戦中断');
consume({match_id:'four',new_frames:frames,frame:frames[2],status:'running',done:true,winner:'blue'});
assert.equal(get('result-title').textContent,'BLUE WIN','batch matches show their result while the job continues');
const red={...aircraft(200),id:'red_0',team:'red'};
const fresh={tick:20,aircraft:[aircraft(100),red],missiles:[],events:[{type:'launch',aircraft:'blue_0',target:'red_0'}]};
ui.consume({match_id:'four',new_frames:[fresh,{...fresh,tick:35,events:[]}],frame:{...fresh,tick:35},config,status:'running',done:false});
ui.hud(fresh);
assert.equal(get('blue-feeds').children.length,1);assert.equal(get('red-feeds').children.length,1);
assert.equal(get('blue-feeds').children[0].dataset.id,'blue_0');assert.equal(get('red-feeds').children[0].dataset.id,'red_0','each team has its own row');
blueFeed=get('blue-feeds').children[0];const redFeed=get('red-feeds').children[0];
assert.equal(blueFeed.dataset.view,'hud','roster changes preserve existing panel modes');assert.equal(redFeed.dataset.view,'chase');
redFeed.querySelector('.feed-modes').children[1].onclick();blueFeed.querySelector('.feed-modes').children[0].onclick();
assert.equal(blueFeed.dataset.view,'chase');assert.equal(blueFeed.querySelector('.feed-hud').hidden,true);
assert.equal(redFeed.dataset.view,'hud','changing one panel never changes another panel');
const threatened={...fresh,aircraft:[{...fresh.aircraft[0],mws_active:true},fresh.aircraft[1]]};
ui.hud(threatened);
blueFeed.querySelector('.feed-modes').children[1].onclick();
assert.equal(blueFeed.dataset.view,'hud');
assert.equal(blueFeed.querySelector('.feed-warning'),null,'HUD has no MWS warning even when older recordings contain MWS data');
consume({match_id:'five',new_frames:[frames[0]],frame:frames[0],status:'running',done:false});
assert.equal(get('red-feeds').children.length,0,'the next match removes old camera windows');
const armed={tick:0,aircraft:[{...aircraft(80),ammo:6},{...red,ammo:2},{...red,id:'unarmed',ammo:0}],missiles:[],events:[]};
const fired={...armed,tick:1,aircraft:armed.aircraft.map(a=>({...a,ammo:a.id==='blue_0'?4:a.ammo}))};
consume({match_id:'ammo',new_frames:[armed,fired],frame:fired,status:'running',done:false});
const ammoPanel=get('blue-feeds').children[0],ammoUI=ammoPanel.querySelector('.feed-ammo'),rounds=ammoUI.querySelector('.ammo-rounds');
assert.equal(rounds.children.length,6,'the initial loadout determines the number of missile icons');
assert.equal(rounds.children.filter(icon=>icon.classes.has('spent')).length,2,'only expended missiles turn gray');
assert.equal(ammoUI.querySelector('.ammo-count').textContent,'4/6');assert.equal(ammoUI.attributes['aria-label'],'ミサイル残弾 4 / 6、飛翔中 0');
assert.equal(get('red-feeds').children[0].querySelector('.ammo-rounds').children.length,2,'each aircraft retains its own loadout');
assert.equal(get('red-feeds').children[1].querySelector('.ammo-rounds').children.length,0,'an unarmed aircraft has no missile icons');
assert.equal(get('red-feeds').children[1].querySelector('.ammo-count').textContent,'0/0');
ammoPanel.querySelector('.feed-modes').children[1].onclick();assert.equal(ammoPanel.dataset.view,'hud');assert.equal(ammoUI.hidden,false,'missile status remains visible in HUD mode');
ui.hud(armed);assert.equal(rounds.children.filter(icon=>icon.classes.has('spent')).length,0,'rewind restores missile availability');
ui.hud({...fired,aircraft:fired.aircraft.map(a=>({...a,ammo:0}))});assert(rounds.children.every(icon=>icon.classes.has('spent')),'zero ammunition leaves all slots visible in gray');
ui.hud({...fired,aircraft:fired.aircraft.map(a=>({...a,alive:false}))});assert.equal(ammoUI.querySelector('.ammo-count').textContent,'4/6','destroyed aircraft retain their last ammunition state');
const shot=(tick,ammo,missiles,events=[])=>({...armed,tick,aircraft:[{...armed.aircraft[0],ammo},{...armed.aircraft[1],ammo:1},armed.aircraft[2]],missiles,events});
const firstRocket={...missile(100),id:'first'},secondRocket={...missile(110),id:'second'},enemyRocket={...missile(120),id:'enemy',owner:'red_0',team:'red'};
const firstShot=shot(1,5,[firstRocket,enemyRocket],[{type:'launch',aircraft:'blue_0',missile:'first'},{type:'launch',aircraft:'red_0',missile:'enemy'}]);
const secondShot=shot(2,4,[firstRocket,secondRocket,enemyRocket],[{type:'launch',aircraft:'blue_0',missile:'second'}]);
const firstEnded=shot(3,4,[secondRocket]),allEnded=shot(4,4,[]);
const flightFrames=[armed,firstShot,secondShot,firstEnded,allEnded];
consume({match_id:'ammo-flight',new_frames:flightFrames,frame:allEnded,status:'running',done:false});
const flightRounds=get('blue-feeds').children[0].querySelector('.ammo-rounds').children;
ui.hud(firstShot);assert(flightRounds[5].classes.has('in-flight'));assert(!flightRounds[4].classes.has('in-flight'),'future launches do not blink while rewinding');
assert.equal(flightRounds.filter(icon=>icon.classes.has('in-flight')).length,1,'an enemy missile never blinks in this aircraft loadout');
ui.hud(secondShot);assert(flightRounds[4].classes.has('in-flight')&&flightRounds[5].classes.has('in-flight'),'simultaneous own missiles each blink');
assert.equal(get('red-feeds').children[0].querySelector('.ammo-rounds').children.filter(icon=>icon.classes.has('in-flight')).length,1,'each team tracks its own launches');
ui.hud(firstEnded);assert(flightRounds[5].classes.has('spent'));assert(flightRounds[4].classes.has('in-flight'),'a surviving missile stays in its original slot when another expires');
assert(!flightRounds[4].classes.has('spent'),'in-flight and spent states are distinct');
ui.hud({...firstEnded,missiles:[{...secondRocket,alive:false}]});assert(!flightRounds[4].classes.has('in-flight'),'inactive legacy missile records cannot blink');
ui.hud(allEnded);assert.equal(flightRounds.filter(icon=>icon.classes.has('in-flight')).length,0,'all icons stop blinking after impact or expiry');
assert.equal(flightRounds.filter(icon=>icon.classes.has('spent')).length,2);
ui.hud(secondShot);assert.equal(flightRounds.filter(icon=>icon.classes.has('in-flight')).length,2,'rewinding restores airborne states');
vm.runInContext('paused=true',ui);ui.hud(secondShot);assert.equal(flightRounds[4].style.animationPlayState,'paused','video pause freezes the icon animation');
vm.runInContext('paused=false',ui);ui.hud(secondShot);assert.equal(flightRounds[4].style.animationPlayState,'running');
consume({match_id:'ammo-flight',new_frames:flightFrames,frame:allEnded,status:'running',done:false});ui.hud(secondShot);
assert(flightRounds[4].classes.has('in-flight')&&flightRounds[5].classes.has('in-flight'),'repeated polling never duplicates launch-slot assignments');
consume({match_id:'ammo-next',new_frames:[{...armed,aircraft:[{...aircraft(80),ammo:1}]}],frame:{...armed,aircraft:[{...aircraft(80),ammo:1}]},status:'running',done:false});
assert.equal(get('blue-feeds').children[0].querySelector('.ammo-rounds').children.length,1,'the next match rebuilds slots for the new loadout');
assert.equal(vm.runInContext('launches.size',ui),0,'a new match clears all old missile assignments');
const loneLine=get('blue-feeds').children[0].querySelector('.telemetry-trend').children[0];
assert(!/NaN|Infinity/.test(loneLine.attributes.points),'a single sample and constant values produce finite graph coordinates');
consume({match_id:'trends',new_frames:trendFrames,frame:trendFrames.at(-1),status:'running',done:false});
const trendPanel=get('blue-feeds').children[0],trendData=trendPanel.querySelector('.feed-data');
const altitudeLine=trendData.children[0].querySelector('.telemetry-trend').children[0],speedLine=trendData.children[1].querySelector('.telemetry-trend').children[0];
const xy=line=>line.attributes.points.split(' ').map(point=>point.split(',').map(Number));
assert(xy(altitudeLine)[0][1]>xy(altitudeLine).at(-1)[1],'climbing altitude slopes upward');
assert(xy(speedLine)[0][1]<xy(speedLine).at(-1)[1],'falling speed slopes downward independently');
assert.match(trendData.children[0].textContent,/ALT.*5,500 M/);assert.match(trendData.children[1].textContent,/SPD.*1,260 KM\/H/);
assert.match(trendData.children[0].title,/60秒/);assert.match(trendData.attributes['aria-label'],/高度・速度/);
trendPanel.querySelector('.feed-modes').children[1].onclick();assert.equal(trendData.hidden,false,'graphs remain visible in HUD mode');
ui.hud(trendFrames[30]);assert.match(trendData.children[0].textContent,/3,250 M/,'rewind restores that tick’s altitude');assert.match(trendData.children[0].title,/30秒/);
const rewindLine=altitudeLine.attributes.points;vm.runInContext('paused=true',ui);ui.hud(trendFrames[30]);assert.equal(altitudeLine.attributes.points,rewindLine,'paused graphs remain fixed');vm.runInContext('paused=false',ui);
const deathTrend={...trendFrames.at(-1),tick:121,aircraft:[{...trendFrames.at(-1).aircraft[0],alive:false}],events:[{type:'hit',target:'blue_0'}]},laterTrend={...deathTrend,tick:200,events:[]};
consume({match_id:'trends',new_frames:[deathTrend,laterTrend],frame:laterTrend,status:'running',done:false});
const lastTrend=altitudeLine.attributes.points;ui.hud(deathTrend);assert.equal(altitudeLine.attributes.points,lastTrend,'destroyed aircraft freeze their graph at the loss tick');
assert.match(trendData.attributes['aria-label'],/^LAST TELEMETRY/);
const dense=Array.from({length:1001},(_,tick)=>({...trendFrames[0],tick}));
ui.consume({match_id:'dense-trends',new_frames:dense,frame:dense.at(-1),config:{...config,aero_tick_seconds:.1},status:'running',done:false},true);
const denseLine=get('blue-feeds').children[0].querySelector('.telemetry-trend').children[0];
assert(xy(denseLine).length<=121,'dense recordings keep the SVG paths bounded');
assert(xy(denseLine).every(([x,y])=>Number.isFinite(x)&&Number.isFinite(y)&&x>=0&&x<=100&&y>=0&&y<=24),'constant histories stay within the graph bounds');
const html=fs.readFileSync(path.join(__dirname,'../combatsim/web/arena.html'),'utf8');
assert(!/MATCH COMPLETE|result-detail/.test(html+app),'match results display only the win/draw title');
assert(!/MISSILE AWAY|SPLASH|id="toast"|id="feed"/.test(html+app),'combat text notifications are removed');
assert(!app.includes('機体喪失'),'loss captions use English');
assert(!/\bautoplay\b|<audio\b/.test(html),'audio starts only after an explicit user action');
assert(html.indexOf('id="red-feeds"')<html.indexOf('id="stage"')&&html.indexOf('id="stage"')<html.indexOf('id="blue-feeds"'),'red cameras sit above the main view and blue cameras below');
console.log('PASS: live updates, escaped labels, team windows, rewind, loss, reset and completed/interrupted results');

// Load the complete entry point, then exercise user controls and connection recovery.
void (async()=>{
 const audioSource=fs.readFileSync(path.join(__dirname,'../combatsim/web/arena-audio.js'),'utf8').replaceAll('export function','function');
 const audioLogic=vm.createContext({});vm.runInContext(audioSource,audioLogic);
 assert.equal(audioLogic.crossedAudioEvents(frames,null,5).length,0,'enabling sound never replays old events');
 assert.equal(audioLogic.crossedAudioEvents(frames,0,1).length,0,'sound never precedes its visible event');
 assert.equal(audioLogic.crossedAudioEvents(frames,1,2)[0].event.type,'launch');
 assert.equal(audioLogic.crossedAudioEvents(frames,2,5)[0].event.type,'hit');
 assert.equal(audioLogic.crossedAudioEvents(frames,5,5).length,0,'repeated renders never duplicate sounds');
 assert.equal(audioLogic.crossedAudioEvents(frames,5,0).length,0,'seeking backward never plays future sounds');
 const audioNodes=[],requests=[];
 class AudioNode {
  constructor(){this.gain={value:1,setTargetAtTime:v=>this.gain.value=v,cancelScheduledValues(){}};this.pan={value:0};for(const n of ['threshold','knee','ratio','attack','release'])this[n]={value:0};audioNodes.push(this);}
  connect(){return this;}disconnect(){this.disconnected=true;}start(){this.started=true;}stop(){this.stopped=true;this.onended?.();}
 }
 let audioContext,failAudio=false;
 class AudioContextMock {
  constructor(){audioContext=this;this.currentTime=0;this.state='suspended';this.destination={};}
  createGain(){return new AudioNode();}createDynamicsCompressor(){return new AudioNode();}createBufferSource(){return new AudioNode();}createStereoPanner(){return new AudioNode();}
  async resume(){this.state='running';}async close(){this.state='closed';}async decodeAudioData(){return {duration:2};}
 }
 const audioRuntime=vm.createContext({AudioContext:AudioContextMock,fetch:async url=>{requests.push(url);return {ok:!failAudio,status:503,arrayBuffer:async()=>new ArrayBuffer(1)};}});
 vm.runInContext(audioSource,audioRuntime);const mixer=audioRuntime.createArenaAudio();
 mixer.setActive(true);mixer.volume(.45,.7);mixer.play('launch');
 assert.equal(audioNodes.length,0,'default-off audio creates no audio device or sound source');assert.equal(requests.length,0,'muted startup downloads no audio assets');
 failAudio=true;await assert.rejects(mixer.enable(true),/Audio HTTP 503/);assert.equal(audioNodes[0].gain.value,0,'loading failure stays silent');
 failAudio=false;await mixer.enable(true);
 assert.equal(requests.length,8,'failed loading can be retried');assert.equal(audioNodes.filter(n=>n.loop).length,2,'music and engine use two loops');
 for(let i=0;i<12;i++)mixer.play('launch',2);
 assert.equal(audioNodes.filter(n=>n.started&&!n.loop&&!n.stopped).length,8,'burst firing has bounded sound voices');
 assert(audioNodes.filter(n=>n.pan.value).every(n=>n.pan.value===.75),'stereo panning stays bounded');
 mixer.setActive(false);assert.equal(audioNodes.filter(n=>n.started&&!n.loop&&!n.stopped).length,0,'pause stops effect voices');assert.equal(audioNodes[0].gain.value,0,'pause silences loops');
 mixer.setActive(true);mixer.volume(.2,.3);mixer.play('hit');assert(audioNodes.some(n=>n.gain.value===.3),'effect volume is independent of music ducking');
 await mixer.enable(false);assert.equal(audioNodes[0].gain.value,0,'off silences audio');
 await mixer.enable(true);assert.equal(requests.length,8,'unmuting reuses audio');assert.equal(audioNodes.filter(n=>n.loop).length,2,'unmuting never duplicates loops');
 mixer.dispose();assert.equal(audioContext.state,'closed');assert(audioNodes.filter(n=>n.loop).every(n=>n.stopped));
 console.log('PASS: default-off audio, lazy loading, retry, event timing, bounded mixing, mute, pause and cleanup');
 const THREE=await import('../combatsim/web/arena-assets/three.module.js');
 const webgl=fs.readFileSync(path.join(__dirname,'../combatsim/web/arena-webgl.js'),'utf8').replace(/^import .*;\r?\n/gm,'').replaceAll('export function','function');
 const rendering=vm.createContext({THREE,ArenaScene:S});vm.runInContext(webgl,rendering);
 const roster=[{...aircraft(0),label:'Model Alpha #1'},red,{...aircraft(0),id:'blue_wing1',label:'Model Alpha #2'}, {...red,id:'custom-wing',label:'Model Beta'}];
 assert.equal(rendering.overviewLabel(roster[0],config,'full',roster),'Model Alpha #1 · 5.0 KM','detailed labels retain model, number and altitude');
 assert.equal(rendering.overviewLabel(roster[0],config,'simple',roster),'#1');
 assert.equal(rendering.overviewLabel(roster[2],config,'simple',roster),'#2','simple labels keep the existing model suffix');
 assert.equal(rendering.overviewLabel(roster[3],config,'simple',roster),'#2','unnumbered labels use their team slot, including custom IDs');
 assert.equal(rendering.overviewLabel({...roster[2],alive:false},config,'simple',roster),'#2','shootdowns never renumber aircraft');
 const quantized=new THREE.BufferGeometry();quantized.setAttribute('position',new THREE.BufferAttribute(new Uint16Array([0,0,0,65535,0,0,0,65535,65535]),3,true));
 const source=new THREE.Group(),mesh=new THREE.Mesh(quantized,new THREE.MeshStandardMaterial());mesh.scale.set(12,3,8);mesh.position.set(2,1,-4);source.add(mesh);
 const bytes=Buffer.from(quantized.attributes.position.array.buffer).toString('hex'),normalized=rendering.normalizeAircraft(source),bounds=new THREE.Box3().setFromObject(normalized);
 assert.equal(Buffer.from(quantized.attributes.position.array.buffer).toString('hex'),bytes,'normalization preserves quantized vertex buffers');
 assert(Math.abs(bounds.getSize(new THREE.Vector3()).x-1)<1e-12,'normalization preserves authored decoding transforms');
 assert(bounds.getCenter(new THREE.Vector3()).length()<1e-12,'normalized aircraft is centered');
 for(const a of [aircraft(0),{...aircraft(0,1),pitch:.4,control:{heading_change:.03}}]){
  const forward=new THREE.Vector3(1,0,0).applyQuaternion(rendering.aircraftQuaternion(a));
  const expected=new THREE.Vector3(Math.cos(a.heading)*Math.cos(a.pitch),Math.sin(a.pitch),-Math.sin(a.heading)*Math.cos(a.pitch));
  assert(forward.distanceTo(expected)<1e-12,'glTF attitude follows the simulator heading and pitch');
 }
 const normals=rendering.waterNormals();assert.equal(normals.image.data.length,256*256*4);normals.dispose();
 const clouds=rendering.cloudMaterial();assert.match(clouds.vertexShader,/#include <common>/);clouds.dispose();
 const island=rendering.islandGeometry(340,210,24,1);assert.equal(island.attributes.color.count,island.attributes.position.count);
 assert(island.attributes.position.array.every(Number.isFinite));assert(island.attributes.normal.array.every(Number.isFinite));
 assert(Math.max(...island.attributes.position.array.filter((_,i)=>i%3===1))>10,'the coast has actual shaded mountain relief');island.dispose();
 const plume=rendering.plumeTexture(),layer=rendering.particleLayer(36,plume,true);
 rendering.updateExplosion(new THREE.Group().add(layer,rendering.particleLayer(32,plume,false)),smoking,death.position,config,1,.095);
 assert(layer.geometry.attributes.position.array.every(Number.isFinite));assert(layer.geometry.attributes.particleSize.array.every(n=>Number.isFinite(n)&&n>0));
 layer.geometry.dispose();layer.material.dispose();plume.dispose();
 const chase=rendering.chaseView(aircraft(600),config,1);
 assert.deepEqual(Array.from(chase.target),Array.from(S.position(aircraft(600),config,1)),'individual camera follows the exact aircraft position');
 assert(Math.abs(chase.distance-15*1000/(800*250)*2.8)<1e-12,'individual camera uses actual aircraft scale');
 const testCamera=new THREE.PerspectiveCamera(42,2,.00008,100000);
 const hudAircraft={...aircraft(80,0),pitch:0},drawCommands=[];
 const hudContext=new Proxy({measureText:text=>({width:String(text).length*7})},{get:(target,key)=>key in target?target[key]:(...args)=>{
  assert(args.flat().every(value=>typeof value!=='number'||Number.isFinite(value)),'HUD drawing coordinates stay finite');drawCommands.push([key,...args]);
  if(key==='fillText')drawCommands.at(-1).push(target.fillStyle);
 }});
 const hudCanvas={width:0,height:0,attributes:{},setAttribute(key,value){this.attributes[key]=value;},getBoundingClientRect:()=>({width:380,height:160}),getContext:()=>hudContext};
 const hudBefore=JSON.stringify(hudAircraft);rendering.drawAircraftHUD(hudCanvas,hudAircraft,config,2);
 assert.equal(hudCanvas.width,760);assert.equal(hudCanvas.height,320);
 assert(drawCommands.some(c=>c[0]==='fillText'&&c[1]==='BVR · HDG 090 · RNG 10 KM'),'BVR shows own heading and a physical range scale');
 assert(drawCommands.some(c=>c[0]==='fillText'&&c[1]==='NO TRACK DATA'),'old recordings have no invented contacts');
 assert.equal(drawCommands.filter(c=>c[0]==='arc').length,4,'range rings replace the forward-view pitch ladder');
 assert.equal(JSON.stringify(hudAircraft),hudBefore,'HUD presentation never alters the simulation');
 const standardHUD=JSON.stringify(drawCommands);drawCommands.length=0;
 rendering.drawAircraftHUD(hudCanvas,{...hudAircraft,mws_active:true,mws_direction:[-1,.4,0]},config,2);
 assert.equal(JSON.stringify(drawCommands),standardHUD,'MWS data never changes the BVR HUD');
 const targets=[{...hudAircraft,id:'friend',x:100,y:118,label:'Friendly #2'},
   {...red,id:'radar',x:100,y:120,z:20,heading:Math.PI/2},{...red,id:'link',x:100,y:122,z:24},
   {...red,id:'unseen',x:100,y:120,z:20},{...hudAircraft,id:'behind',x:60}];
 const sensed={...hudAircraft,hud_contacts:[{id:'friend',source:'ally'},{id:'radar',source:'radar'},
   {id:'link',source:'datalink'},{id:'behind',source:'ally'}]},sensorBefore=JSON.stringify(sensed);
 const bvrAircraft=[sensed,...targets],contacts=S.bvrContacts(sensed,bvrAircraft,config);
 assert.deepEqual(Array.from(contacts,p=>p.target.id),['friend','radar','link','behind'],'only recorded contacts enter the BVR picture');
 assert.equal(contacts[1].forward,5);assert.equal(contacts[1].right,0);assert.equal(contacts[1].range,5);
 assert.equal(contacts[2].altitude,1000,'relative altitude is in physical meters');
 assert.equal(contacts[3].forward,-5,'rear contacts retain a negative forward distance');
 assert.equal(contacts[1].heading,-Math.PI/2,'course stems rotate into the own-heading reference frame');
 const around=[{...red,id:'front',x:100,y:120},{...red,id:'right',x:80,y:100},
   {...red,id:'aft',x:60,y:120},{...red,id:'left',x:80,y:140}];
 const own={...hudAircraft,hud_contacts:around.map(t=>({id:t.id,source:'radar'}))};
 for(const heading of [0,Math.PI/2,Math.PI,-Math.PI/2,2*Math.PI]){
  const picture=S.bvrContacts({...own,heading},around,config);
  for(let i=0;i<4;i++){
   const bearing=i*Math.PI/2+heading;
   assert(Math.abs(picture[i].right-5*Math.sin(bearing))<1e-10,'left/right positions follow own heading through a complete turn');
   assert(Math.abs(picture[i].forward-5*Math.cos(bearing))<1e-10,'forward/rear positions follow own heading through a complete turn');
   assert.equal(picture[i].range,5,'turning never changes physical contact range');
  }
 }
 const vertical={...red,id:'vertical',x:80,y:120,z:24};
 const above=S.bvrContacts({...own,hud_contacts:[{id:'vertical',source:'radar'}]},[vertical],config)[0];
 assert.equal(above.horizontal,0);assert.equal(above.range,1);assert.equal(above.altitude,1000,'vertical separation stays separate from the plan-view position');
 assert.equal(S.bvrContacts({...sensed,alive:false},bvrAircraft,config).length,0);
 assert.equal(S.bvrContacts(hudAircraft,bvrAircraft,config).length,0);
 assert.equal(S.bvrContacts({...own,hud_contacts:[{id:'front',source:'unknown'},{id:'missing',source:'radar'},{id:own.id,source:'ally'}]},[own,...around],config).length,0,'unknown, missing and ownship contacts are excluded');
 assert.equal(S.bvrContacts(own,around.map(t=>({...t,alive:false})),config).length,0,'dead contacts disappear');
 drawCommands.length=0;rendering.drawAircraftHUD(hudCanvas,sensed,config,2,bvrAircraft);
 for(const label of ['F #2 · 5.0 KM','R #1 · 5.0 KM','DL #2 · 5.1 KM'])assert(drawCommands.some(c=>c[0]==='fillText'&&c[1]===label),'each source has an aircraft number and slant range');
 for(const label of ['R #1 · 5.0 KM','DL #2 · 5.1 KM'])assert.equal(drawCommands.find(c=>c[0]==='fillText'&&c[1]===label).at(-1),'#ff5969','both enemy contact sources use red');
 assert.equal(drawCommands.find(c=>c[0]==='fillText'&&c[1]==='F #2 · 5.0 KM').at(-1),'#77e7ff','friendly information remains cyan');
 assert(drawCommands.some(c=>c[0]==='fillText'&&c[1]==='AFT · Δ +0 M'),'a rear contact explicitly reads AFT');
 assert(drawCommands.some(c=>c[0]==='fillText'&&c[1].includes('Δ +1,000 M')),'relative altitude is labeled independently of range');
 const plotted=drawCommands.filter(c=>c[0]==='translate');
 assert(plotted[1][2]<95&&plotted[3][2]>95,'forward and rear contacts lie above and below ownship');
 assert.equal(drawCommands.filter(c=>c[0]==='rect').length,1,'only the recorded radar target is boxed');
 assert.equal(drawCommands.filter(c=>c[0]==='rotate').length,4,'each contact has its own course stem');
 assert.match(hudCanvas.attributes['aria-label'],/相対方位AFT/,'all contacts have an accessible relative bearing');
 assert.equal(JSON.stringify(sensed),sensorBefore,'sensor rendering leaves recorded contacts unchanged');
 const unbanked=JSON.stringify(drawCommands);drawCommands.length=0;
 rendering.drawAircraftHUD(hudCanvas,{...sensed,pitch:1.2},config,2,bvrAircraft);
 assert.equal(JSON.stringify(drawCommands),unbanked,'plan-view contact geometry does not tilt with aircraft pitch');
 drawCommands.length=0;rendering.drawAircraftHUD(hudCanvas,{...sensed,alive:false},config,2,bvrAircraft);
 assert(!drawCommands.some(c=>c[0]==='fillText'&&/^(F #|R #|DL #)/.test(c[1])),'destroyed observers show no active sensor contacts');
 assert(drawCommands.some(c=>c[0]==='fillText'&&c[1]==='TRACK ENDED'));
 const distant={...targets[0],x:1080};
 drawCommands.length=0;rendering.drawAircraftHUD(hudCanvas,sensed,config,2,[sensed,distant]);
 assert.match(hudCanvas.attributes['aria-label'],/表示半径500km/,'the range expands to include a distant known teammate');
 const distantPosition=drawCommands.find(c=>c[0]==='translate');assert(distantPosition[1]>190&&distantPosition[1]<=241,'distant contacts stay inside the outer physical range ring');
 for(const [width,height] of [[1000,250],[380,160],[180,80],[180,76]]){
  hudCanvas.getBoundingClientRect=()=>({width,height});drawCommands.length=0;
  rendering.drawAircraftHUD(hudCanvas,sensed,config,1,bvrAircraft);
  for(const c of drawCommands.filter(c=>c[0]==='arc'&&c[1]===width/2))assert(c[1]-c[3]>=0&&c[1]+c[3]<=width&&c[2]-c[3]>=0&&c[2]+c[3]<=height,'the plan view fits compact and maximized panels');
  assert(drawCommands.some(c=>c[0]==='fillText'&&String(c[1]).includes('090')),'small panels retain own heading');
 }
 console.log('PASS: heading-up BVR geometry, all-around bearings, physical range/altitude, course, recorded-only contacts and responsive panels');
 const guided={...sensed,hud_contacts:sensed.hud_contacts.map(t=>t.id==='radar'?{...t,rmax_km:55,rmax_tick:10}:t)};
 const guideContacts=S.bvrContacts(guided,[guided,...targets],config);
 assert.equal(S.bvrContacts({...sensed,hud_contacts:[{id:'friend',source:'ally',rmax_km:55}]},bvrAircraft,config)[0].rmax,null,'friendly contacts never get weapon-range markers');
 assert.equal(guideContacts[1].rmax,55);assert.equal(guideContacts[1].rmaxHorizontal,55);
 assert.equal(guideContacts[2].rmax,null,'older records never invent range predictions');
 const elevated=S.bvrContacts({...sensed,hud_contacts:[{id:'link',source:'datalink',rmax_km:5}]},bvrAircraft,config)[0];
 assert(Math.abs(elevated.rmaxHorizontal-Math.sqrt(24))<1e-12,'RMAX slant range projects onto horizontal rings using real altitude separation');
 const badRange={...sensed,hud_contacts:[{id:'radar',source:'radar',rmax_km:-1}]};
 assert.equal(S.bvrContacts(badRange,[badRange,...targets],config)[0].rmax,null,'invalid range predictions are not drawn');
 const rockets=[{...missile(110),id:'own',alive:true}, {...missile(120),id:'ally',owner:'friend'},
  {...missile(60),id:'enemy',team:'red',owner:'red_0'}, {...missile(100),id:'shared',team:'red',owner:'red_0'},
  {...missile(140),id:'unseen',team:'red',owner:'red_0'}, {...missile(140),id:'dead',alive:false}];
 const armed={...guided,hud_missiles:[{id:'own',source:'own'},{id:'ally',source:'ally'},
  {id:'enemy',source:'radar'},{id:'shared',source:'datalink'},{id:'dead',source:'own'}]};
 assert.deepEqual(Array.from(S.bvrMissiles(armed,rockets,config),m=>m.target.id),['own','ally','enemy','shared'],'BVR includes friendly telemetry and only observed enemy missiles');
 assert(S.bvrMissiles(armed,rockets,config)[2].forward<0,'enemy missiles behind ownship remain on the rear half');
 assert.equal(S.bvrMissiles({...armed,alive:false},rockets,config).length,0);
 assert.equal(S.bvrMissiles({...armed,hud_missiles:[]},rockets,config).length,0,'recorded loss of missile contact is authoritative');
 assert.deepEqual(Array.from(S.bvrMissiles(sensed,rockets,config),m=>m.target.id),['own','ally'],'legacy recordings only recover known friendly missile telemetry');
 assert.equal(S.bvrMissiles({...armed,hud_missiles:[{id:'unseen',source:'own'}]},rockets,config).length,0,'a forged friendly source cannot reveal an enemy missile');
 for(const width of [380,900]){
  hudCanvas.getBoundingClientRect=()=>({width,height:160});drawCommands.length=0;
  rendering.drawAircraftHUD(hudCanvas,armed,config,1,[armed,...targets],rockets);
  assert(drawCommands.some(c=>c[0]==='fillText'&&String(c[1]).includes('RMAX')),'both narrow and wide panels display the recorded range estimate');
  const marks=drawCommands.filter(c=>c[0]==='fillText'&&c[1]==='M');
  assert.equal(marks.length,4);assert.deepEqual(marks.map(c=>c.at(-1)),['#e8c68c','#77e7ff','#ff5969','#ff5969'],'missile markers distinguish own, allied and enemy rounds');
  assert.match(hudCanvas.attributes['aria-label'],/推定RMAX 55.0km/);
  assert.match(hudCanvas.attributes['aria-label'],/レーダー捕捉の敵弾 enemy/);
  assert(!hudCanvas.attributes['aria-label'].includes('unseen'),'unobserved missile identity never reaches the HUD');
 }
 console.log('PASS: recorded RMAX, legacy behavior, friendly/observed missile markers and source colors');
 const gridView={target:[0,0,30],distance:600,yaw:.2,elevation:.52},grid=rendering.distanceGrid(config,gridView);
 assert.equal(grid.step,10,'grid spacing uses readable physical kilometers');
 const xLines=grid.lines.filter(line=>!line.axis&&line.from[0]===line.to[0]);
 assert.equal((xLines[1].from[0]-xLines[0].from[0])*grid.units*config.aero_length_scale,10000,'neighboring grid lines are exactly 10 km apart');
 assert.equal(grid.lines.find(line=>line.axis==='z').to[2],12.5,'Z axis reaches the physical altitude ceiling');
 assert(grid.labels.some(label=>label.text==='X 100 KM')&&grid.labels.some(label=>label.text==='Y 50 KM'),'axis coordinates use the scenario origin');
 assert.equal(rendering.distanceGrid({...config,aero_length_scale:500},gridView).step,20,'coordinate scale follows scenario length units');
 assert(rendering.distanceGrid(config,{...gridView,distance:60000}).lines.length<50,'wide zoom keeps the number of grid lines bounded');
 for(const [w,h,yaw,elevation,height] of [[800,400,.2,.52,1],[300,600,2.3,.12,1],[800,400,-2,1.4,5]]){
  const view={...gridView,yaw,elevation};rendering.setCameraView(testCamera,view,w/h);drawCommands.length=0;
  rendering.drawDistanceGrid(hudContext,testCamera,config,view,height,w,h);
  assert(drawCommands.some(c=>c[0]==='fillText'&&c[1]==='GRID 10 KM'),'grid legend states the real horizontal distance at every altitude scale');
  assert(drawCommands.some(c=>c[0]==='stroke'),'distance grid is drawn at each camera angle');
  for(const command of drawCommands.filter(c=>c[0]==='moveTo'||c[0]==='lineTo')){
   assert(command[1]>=-1e-5&&command[1]<=w+1e-5&&command[2]>=-1e-5&&command[2]<=h+1e-5,'grid segments clip to the camera frustum without crossing the horizon');
  }
 }
 const mainLength=rendering.overviewLength(.075,600,180);
 assert(mainLength>.075,'the wide shot enlarges actual 3D geometry');
 assert(Math.abs(mainLength/(2*Math.tan(21*Math.PI/180)*600)*180-18)<1e-10,'standard size uses the former compact 18-pixel scale');
 assert(Math.abs(rendering.overviewLength(.075,600,180,28)/mainLength-28/18)<1e-10,'large size uses the former standard 28-pixel scale');
 assert.equal(rendering.overviewLength(.075,600,180,0),.075,'actual size stays at physical dimensions even in the wide shot');
 assert.equal(rendering.overviewLength(.075,.1,1000,56),.075,'close views never shrink below actual scale');
 const rocket=rendering.createMissile(),rocketBounds=new THREE.Box3().setFromObject(rocket),rocketSize=rocketBounds.getSize(new THREE.Vector3());
 assert(Math.abs(rocketSize.x-1)<1e-6,'missile geometry has a normalized body length');
 assert(rocket.children[0].children.length>=7,'missiles have a body, pointed nose, four fins and exhaust');
 rocket.traverse(n=>{if(n.isMesh)assert(n.geometry.attributes.position.array.every(Number.isFinite));});
 const root={left:10,top:20,right:1010,bottom:620,width:1000,height:600};
 const slot={left:50,top:500,right:350,bottom:600,width:300,height:100};
 const clip={left:100,top:500,right:700,bottom:600};
 const viewport=rendering.viewportBounds(slot,root,clip);
 assert.deepEqual(Array.from(viewport.scissor),[90,20,250,100],'scrolling camera rows clip without distorting the camera viewport');
 assert.equal(viewport.x,40);assert.equal(viewport.width,300);
 const listeners={},windowListeners={};document.addEventListener=(name,handler)=>listeners[name]=handler;
 const timers=[],drawCalls=[],liveState={match_id:'live',new_frames:frames,frame:frames[2],status:'running',done:false,winner:null,config,seed:12};
 const browser=vm.createContext({document,ArenaScene:S,createArenaAudio:()=>audioStub,crossedAudioEvents:audioLogic.crossedAudioEvents,bootstrap:{...liveState,token:"test-token"},matchMedia:()=>({matches:false}),performance:{now:()=>100},
   window:{addEventListener:(name,handler)=>windowListeners[name]=handler},createArenaRenderer:()=>({ready:Promise.resolve(),draw(...args){drawCalls.push(args);},reset(){},dispose(){}}),requestAnimationFrame(){},setTimeout:(callback,delay)=>{timers.push(delay);return timers.length;},clearTimeout(){},AbortSignal,
   fetch:async()=>{throw Error('offline');}});
 vm.runInContext(app,browser);
 assert.equal(get('show-labels').value,'simple','aircraft numbers are the default labels');
 assert.equal(get('aircraft-size').value,'18','standard aircraft size is selected initially');
 assert.equal(get('show-feeds').checked,false,'individual views are disabled by default');
 assert.equal(get('show-grid').checked,false,'distance grid is disabled by default');
 assert.equal(get('sound').checked,false,'sound is off when no preference has been saved');assert.equal(audioCalls.enabled.length,0,'startup never enables audio');
 assert(get('viewports').classes.has('feeds-hidden'),'the overview fills the screen on startup');
 browser.updateCamera(frames[0],16);
 const distance=vm.runInContext('camera.distance',browser);
 assert(distance>100&&distance<1000,'the main camera frames the engagement rather than an empty 200 km ocean');
 const stage=get('stage');
 stage.onpointerdown({button:0,clientX:0,clientY:0,pointerId:1});stage.onpointermove({clientX:30,clientY:10});
 browser.updateCamera(frames[0],16);
 assert.equal(vm.runInContext('camera.yaw',browser),vm.runInContext('manualYaw',browser),'drag rotation follows the pointer without easing');
 assert.equal(vm.runInContext('camera.elevation',browser),vm.runInContext('manualElevation',browser),'drag elevation follows the pointer without easing');
 const rotatedDistance=vm.runInContext('camera.distance',browser),rotatedView=S.battleView(frames[0],[],config,1,2,.02,.57);
 assert(Math.abs(rotatedDistance-rotatedView.distance)<1e-10,'fitting the new viewing angle does not add delayed zoom');
 stage.onpointerup();stage.onpointermove({clientX:100,clientY:100});
 for(let i=0;i<120;i++)browser.updateCamera(frames[0],16);
 assert(Math.abs(vm.runInContext('camera.distance',browser)-rotatedDistance)<1e-10,'a stationary scene does not keep zooming after a drag');
 assert.equal(vm.runInContext('camera.yaw',browser),vm.runInContext('manualYaw',browser),'rotation stops when the pointer is released');
 for(const [deltaY,deltaMode] of [[-80,0],[-5,1],[-80/stage.getBoundingClientRect().height,2]]){
  get('reset-camera').onclick();browser.updateCamera(frames[0],16);
  stage.listeners.wheel({preventDefault(){},deltaY,deltaMode});browser.updateCamera(frames[0],16);
  assert(Math.abs(vm.runInContext('zoom',browser)-1.12)<1e-12,'pixel, line and page wheel units have consistent sensitivity');
  assert(Math.abs(vm.runInContext('camera.distance',browser)-distance/1.12)<1e-10,'wheel zoom takes effect on the next render without easing');
  stage.listeners.wheel({preventDefault(){},deltaY,deltaMode});
  assert(Math.abs(vm.runInContext('zoom',browser)-1.24)<1e-12,'equal wheel deltas change zoom by the same amount');
  stage.listeners.wheel({preventDefault(){},deltaY:-deltaY*2,deltaMode});browser.updateCamera(frames[0],16);
  assert(Math.abs(vm.runInContext('camera.distance',browser)-distance)<1e-10,'opposite wheel input restores the view');
 }
 get('reset-camera').onclick();browser.updateCamera(frames[0],16);
 const moved={...frames[0],tick:300,aircraft:[aircraft(600)]};browser.updateCamera(moved,16);
 const shift=Math.abs(vm.runInContext('camera.target[0]',browser)-S.position(aircraft(80),config,1)[0]);
 assert(shift>0&&shift<5,'the wide camera pans gently instead of snapping to a new formation position');
 const spread={...frames[0],aircraft:[aircraft(80),{...aircraft(720),id:'red_0',team:'red',y:350}]};
 browser.updateCamera(spread,16);
 const expandingDistance=vm.runInContext('camera.distance',browser);
 assert(expandingDistance>distance&&expandingDistance<S.battleView(spread,[],config,1,2,.2,.52).distance,'automatic framing still eases out as the engagement spreads');
 browser.updateCamera(frames[0],16);
 assert(vm.runInContext('camera.distance',browser)>distance&&vm.runInContext('camera.distance',browser)<expandingDistance,'automatic framing still eases in after aircraft disappear');
 for(const aspect of [.55,1,3])for(const yaw of [0,.5,2]){
  const elevation=.52,scene={tick:0,aircraft:[aircraft(80),{...red,y:380,z:35}],missiles:[]},view=S.battleView(scene,[],config,1,aspect,yaw,elevation),d=view.distance,target=new THREE.Vector3(view.target[0],view.target[2],-view.target[1]),cam=new THREE.PerspectiveCamera(42,aspect,.00008,100000);
  cam.position.copy(target).add(new THREE.Vector3(Math.sin(yaw)*Math.cos(elevation)*d,Math.sin(elevation)*d,Math.cos(yaw)*Math.cos(elevation)*d));cam.lookAt(target);cam.updateMatrixWorld();
  for(const a of scene.aircraft){
   const point=S.position(a,config,1),p=new THREE.Vector3(point[0],point[2],-point[1]).project(cam);assert(Math.abs(p.x)<=1&&Math.abs(p.y)<=1,'both teams fit in the wide view at portrait and wide aspect ratios');
  }
 }
 get('play').onclick();assert.equal(get('phase').textContent,'映像を一時停止中');
 get('timeline').oninput({target:{value:'1'}});assert(!get('blue-feeds').children[0].classes.has('lost'));
 get('go-live').onclick();assert.match(get('blue-feeds').children[0].querySelector('.feed-data').attributes['aria-label'],/^LAST TELEMETRY/);assert.equal(get('phase').textContent,'交戦中');
 get('settings').onclick();assert.equal(get('display-settings').open,true);get('close-settings').onclick();assert.equal(get('display-settings').open,false);
 for(const mode of ['simple','off','full']){
  get('show-labels').onchange({target:{value:mode}});
  browser.animate(100);assert.equal(drawCalls.at(-1)[5].labels,mode,'all three label settings reach the renderer');
 }
 assert.equal(drawCalls.at(-1)[7].length,0,'the default layout does not render individual cameras');
 for(const checked of [true,false]){
  get('show-grid').onchange({target:{checked}});browser.animate(100);
  assert.equal(drawCalls.at(-1)[5].grid,checked,'display settings toggle the overview distance grid');
 }
 for(const size of [0,28,18]){
  get('aircraft-size').onchange({target:{value:String(size)}});browser.animate(100);
  assert.equal(drawCalls.at(-1)[5].aircraftSize,size,'actual, large and standard sizes reach the renderer unchanged');
  assert.equal(get('scale-note').textContent,size===0?'ACTUAL SCALE':'SCALE ENHANCED');
 }
 get('show-feeds').checked=true;get('show-feeds').onchange({target:get('show-feeds')});
 const layout=get('viewports'),redResize=get('red-resize'),blueResize=get('blue-resize');
 const tracks=()=>Array.from((layout.style.gridTemplateRows||'').matchAll(/,([\d.]+)fr/g),match=>Number(match[1]));
 for(const [index,id] of ['red-views','stage','blue-views'].entries()){
  const row=get(id),rect=row.getBoundingClientRect();row.getBoundingClientRect=()=>({...rect,height:layout.classes.has('feeds-hidden')?(index===1?416:0):(tracks().length?tracks():[100,200,100])[index]});
 }
 const pointer=y=>({button:0,pointerId:7,clientY:y,preventDefault(){}});
 for(const [handle,move,expected] of [[redResize,140,[140,160,100]],[blueResize,60,[100,160,140]]]){
  handle.ondblclick();handle.onpointerdown(pointer(100));handle.onpointermove(pointer(move));
  assert.deepEqual(tracks(),expected,'each boundary changes its own team row and the main view only');
  assert(layout.classes.has('resizing'));assert.equal(handle.attributes['aria-valuenow'],'140');
  const current=layout.style.gridTemplateRows;handle.onpointermove({...pointer(500),pointerId:8});assert.equal(layout.style.gridTemplateRows,current);
  handle.onpointerup(pointer(move));handle.onpointermove(pointer(300));assert.equal(layout.style.gridTemplateRows,current);assert(!layout.classes.has('resizing'));
 }
 redResize.ondblclick();redResize.onpointerdown(pointer(100));redResize.onpointermove(pointer(10000));
 assert.deepEqual(tracks(),[170,130,100],'enlarging a feed preserves the main view minimum');
 redResize.onpointermove(pointer(-10000));assert.deepEqual(tracks(),[76,224,100],'feeds retain a usable minimum height');
 redResize.onpointercancel(pointer(100));const cancelled=layout.style.gridTemplateRows;redResize.onpointermove(pointer(150));assert.equal(layout.style.gridTemplateRows,cancelled);assert(!layout.classes.has('resizing'));
 for(const [handle,key,expected] of [[redResize,'ArrowDown',[110,190,100]],[blueResize,'ArrowUp',[100,190,110]]]){
  handle.ondblclick();handle.onkeydown({key,preventDefault(){},stopPropagation(){}});assert.deepEqual(tracks(),expected,'keyboard resizing follows the direction of each boundary');
 }
 const resizedTracks=layout.style.gridTemplateRows;windowListeners.resize();assert.equal(layout.style.gridTemplateRows,resizedTracks,'window resize keeps proportional tracks');
 const showFeeds=get('show-feeds'),savedFeed=get('blue-feeds').children[0];
 savedFeed.querySelector('.feed-modes').children[1].onclick();
 assert.equal(showFeeds.checked,true,'individual views can be enabled from settings');
 browser.animate(100);assert.equal(drawCalls.at(-1)[7].length,1);
 showFeeds.checked=false;showFeeds.onchange({target:showFeeds});
 assert(layout.classes.has('feeds-hidden'));
 browser.animate(116);assert.equal(drawCalls.at(-1)[7].length,0,'hidden feeds are not rendered');
 assert.equal(drawCalls.at(-1)[0].aircraft.length,1,'the overview still receives all aircraft');
 windowListeners.resize();browser.setMaximized(true);browser.setMaximized(false);
 assert.equal(layout.style.gridTemplateRows,resizedTracks,'hiding feeds preserves resized tracks across viewport changes');
 assert.equal(blueResize.attributes['aria-valuenow'],'110','hidden rows do not overwrite saved accessibility values with zero');
 showFeeds.checked=true;showFeeds.onchange({target:showFeeds});
 assert(!layout.classes.has('feeds-hidden'));
 browser.animate(132);assert.equal(drawCalls.at(-1)[7][0].node,savedFeed,'showing feeds reuses the same panels');
 assert.equal(savedFeed.dataset.view,'hud','showing feeds retains HUD mode');
 assert.equal(blueResize.attributes['aria-valuenow'],'110','showing feeds restores resize accessibility values');
 get('stage').listeners.wheel({preventDefault(){},deltaY:-100000,deltaMode:0});assert.equal(vm.runInContext('zoom',browser),2.4);
 get('stage').listeners.wheel({preventDefault(){},deltaY:100000,deltaMode:0});assert.equal(vm.runInContext('zoom',browser),.5);
 get('reset-camera').onclick();assert.equal(vm.runInContext('zoom',browser),1);
 document.documentElement.requestFullscreen=async()=>{throw Error('fullscreen unavailable');};
 await get('fullscreen').onclick();assert(document.body.classes.has('maximized'),'embedded browsers still maximize the layout');assert.equal(get('fullscreen-label').textContent,'元に戻す');
 assert.equal(layout.style.gridTemplateRows,resizedTracks,'maximizing retains the chosen height proportions');
 listeners.keydown({code:'Escape',target:{matches:()=>true}});assert(!document.body.classes.has('maximized'),'Escape exits layout maximization even when the timeline is focused');
 document.documentElement.requestFullscreen=async()=>{document.fullscreenElement={};listeners.fullscreenchange();};
 document.exitFullscreen=async()=>{document.fullscreenElement=null;listeners.fullscreenchange();};
 await get('fullscreen').onclick();assert(document.body.classes.has('maximized'));assert.equal(get('fullscreen').attributes['aria-pressed'],'true');
 await get('fullscreen').onclick();assert(!document.body.classes.has('maximized'));assert.equal(get('fullscreen-label').textContent,'最大化');
 blueResize.ondblclick();assert.equal(layout.style.gridTemplateRows,'','double-click restores the CSS default layout');
 await browser.poll();assert.equal(get('connection').lastElementChild.textContent,'RECONNECTING');assert.equal(timers.at(-1),1500);
 browser.fetch=async()=>({ok:true,json:async()=>({...liveState,new_frames:[]})});await browser.poll();
 assert.equal(get('connection').lastElementChild.textContent,'LIVE');assert.equal(timers.at(-1),300);assert.equal(get('blue-feeds').children.length,1);
 let paceRequest;
 browser.fetch=async(url,options)=>{if(url.startsWith('/api/watch'))return {ok:true,json:async()=>({...liveState,new_frames:[],pace:120})};paceRequest={url,options};return {ok:true,json:async()=>({job:{pace:120}})};};
 get('arena-pace').value='120';
 await get('arena-pace').onchange();
 assert.equal(paceRequest.url,'/api/pace');
 assert.equal(paceRequest.options.headers['X-CombatSim-Token'],'test-token');
 assert.deepEqual(JSON.parse(paceRequest.options.body),{pace:120});
 assert.equal(get('arena-pace').value,'120');
 assert.equal(get('arena-pace').disabled,false);
 await browser.poll();
 // Timeline buttons use recorded frames, clamp at either end and never call a match API.
 get('go-live').onclick();get('back').onclick();assert.equal(vm.runInContext('shownTick',browser),2);
 assert.equal(vm.runInContext('paused',browser),true);assert.equal(vm.runInContext('follow',browser),false);
 get('back-ten').onclick();assert.equal(vm.runInContext('shownTick',browser),0);assert.equal(get('back').disabled,true);
 get('step').onclick();assert.equal(vm.runInContext('shownTick',browser),2);
 get('step').onclick();assert.equal(vm.runInContext('shownTick',browser),5);assert.equal(get('step').disabled,true);
 get('go-live').onclick();assert.equal(vm.runInContext('follow',browser),true);
 // Match actions share token protection, serialize mutations and retain API errors through polling.
 let serverStatus='running',finishControl,controlRequests=[];
 let matchDraft={fields:{pace:'10'},request:{mode:'duel',single_match:true,blue_model:'saved-blue',red_model:'saved-red',seed_start:2099,pace:10}};
 browser.localStorage={getItem:()=>JSON.stringify(matchDraft),setItem:(key,value)=>{matchDraft=JSON.parse(value);}};
 browser.fetch=async(url,options)=>{
  if(url.startsWith('/api/watch'))return {ok:true,json:async()=>({...liveState,new_frames:[],status:serverStatus,pace:120})};
  controlRequests.push({url,options});
  if(url==='/api/pause')return new Promise(resolve=>{finishControl=()=>{serverStatus=JSON.parse(options.body).paused?'paused':'running';resolve({ok:true,json:async()=>({job:{status:serverStatus,pace:120}})});};});
  serverStatus=url==='/api/stop'?'stopping':'running';return {ok:true,json:async()=>({job:{status:serverStatus,pace:120}})};
 };
 const pendingPause=get('match-pause').onclick();
 assert.equal(get('match-pause').disabled,true);assert.equal(get('match-stop').disabled,true);assert.equal(get('arena-pace').disabled,true);
 await get('match-stop').onclick();assert.equal(controlRequests.length,1,'duplicate operations cannot race');
 finishControl();await pendingPause;await browser.poll();
 assert.equal(controlRequests[0].url,'/api/pause');assert.deepEqual(JSON.parse(controlRequests[0].options.body),{paused:true});
 assert.equal(controlRequests[0].options.headers['X-CombatSim-Token'],'test-token');assert.equal(get('match-pause').textContent,'対戦を再開');
 const pendingResume=get('match-pause').onclick();finishControl();await pendingResume;await browser.poll();
 assert.deepEqual(JSON.parse(controlRequests[1].options.body),{paused:false});
 await get('match-stop').onclick();await browser.poll();assert.equal(controlRequests[2].url,'/api/stop');
 assert.equal(get('match-start').disabled,true,'stopping must finish before another match can start');
 serverStatus='stopped';await browser.poll();assert.equal(get('match-start').disabled,false);
 await get('match-start').onclick();await browser.poll();assert.equal(controlRequests[3].url,'/api/start');
 assert.deepEqual(JSON.parse(controlRequests[3].options.body),{mode:'duel',single_match:true,blue_model:'saved-blue',red_model:'saved-red',seed_start:2099,pace:10});
 get('arena-pace').value='120';await get('arena-pace').onchange();assert.equal(matchDraft.request.pace,120);assert.equal(matchDraft.fields.pace,'120','ARENA pace changes update both the next match and the setup form');
 browser.localStorage={getItem(){throw Error('blocked');}};
 browser.fetch=async()=>({ok:true,json:async()=>({catalog:[{id:'rule:fix',kind:'rule'}]})});
 assert.deepEqual(JSON.parse(JSON.stringify(await browser.matchSetup())),{blue_model:'rule:fix',red_model:'rule:fix'},'direct ARENA visits also have a default setup');
 browser.fetch=async(url)=>url.startsWith('/api/watch')?{ok:true,json:async()=>({...liveState,new_frames:[]})}:{ok:false,json:async()=>({error:'invalid setup'})};
 await get('match-stop').onclick();await browser.poll();assert.match(get('message').textContent,/invalid setup/,'polling does not erase control errors');
 browser.fetch=async()=>({ok:true,json:async()=>({...liveState,new_frames:[],pace:120})});
 vm.runInContext("controlError=''",browser);await browser.poll();
 const soundToggle=get('sound');soundToggle.checked=true;await soundToggle.onchange({target:soundToggle});
 assert.equal(audioCalls.enabled.at(-1),true);assert.equal(audioCalls.active.at(-1),true);
 get('play').onclick();assert.equal(audioCalls.active.at(-1),false,'pausing video immediately mutes audio');
 get('play').onclick();assert.equal(audioCalls.active.at(-1),true);
 get('timeline').oninput({target:{value:'1'}});assert.equal(audioCalls.active.at(-1),false,'seeking mutes audio');assert.equal(vm.runInContext('soundTick',browser),null);
 get('go-live').onclick();assert.equal(audioCalls.active.at(-1),true);assert.equal(vm.runInContext('soundTick',browser),null,'LIVE skips the sound backlog');
 get('music-volume').value='20';get('effects-volume').value='30';get('music-volume').oninput();assert.deepEqual(audioCalls.volume.at(-1),[.2,.3]);
 document.hidden=true;listeners.visibilitychange();assert.equal(audioCalls.active.at(-1),false,'hidden tabs are silent');document.hidden=false;
 soundToggle.checked=false;await soundToggle.onchange({target:soundToggle});assert.equal(audioCalls.enabled.at(-1),false);
 audioStub.enable=async()=>{throw Error('audio unavailable');};soundToggle.checked=true;await soundToggle.onchange({target:soundToggle});
 assert.equal(soundToggle.checked,false,'a failed audio start restores the off setting');assert.equal(soundToggle.disabled,false);assert.match(get('message').textContent,/audio unavailable/);
 windowListeners.pagehide({persisted:false});assert.equal(audioCalls.disposed,1,'leaving the page releases audio resources');
 const preferenceKey='combatsim.arena.preferences',savedValues=new Map();
 const storage={getItem:key=>savedValues.get(key)??null,setItem:(key,value)=>savedValues.set(key,value)};
 function reloadArena(localStorage){
  const events={},windowEvents={},audio={enabled:[],volume:[],active:[]};delete get('sound').parentElement;
  const page={...document,body:new Node('body'),documentElement:{},fullscreenElement:null,hidden:false,addEventListener:(name,handler)=>events[name]=handler};
  const context=vm.createContext({document:page,localStorage,ArenaScene:S,bootstrap:{...liveState,token:'test-token'},matchMedia:()=>({matches:false}),performance:{now:()=>100},
   window:{addEventListener:(name,handler)=>windowEvents[name]=handler},createArenaRenderer:()=>({ready:Promise.resolve(),reset(){},dispose(){}}),
   createArenaAudio:()=>({enable:async value=>audio.enabled.push(value),setActive:value=>audio.active.push(value),volume:(...values)=>audio.volume.push(values),play(){},dispose(){}}),
   crossedAudioEvents:audioLogic.crossedAudioEvents,requestAnimationFrame(){},setTimeout(){},clearTimeout(){},AbortSignal,fetch:async()=>({ok:true,json:async()=>({...liveState,new_frames:[]})})});
  vm.runInContext(app,context);return {context,events,windowEvents,audio,page};
 }
 async function changePreference(id,value){
  const control=get(id);if(typeof value==='boolean')control.checked=value;else control.value=String(value);
  await control.onchange({target:control});get('display-settings').listeners.change();
 }
 const initial=reloadArena(storage);
 for(const [id,value] of [['show-feeds',true],['show-grid',true],['show-trails',false],['show-effects',false],['show-labels','full'],['height-scale',3],['aircraft-size',0],['quality','standard']])await changePreference(id,value);
 get('music-volume').value='20';get('effects-volume').value='30';get('effects-volume').oninput();
 get('blue-feeds').children[0].querySelector('.feed-modes').children[1].onclick();
 get('red-resize').onkeydown({key:'ArrowDown',preventDefault(){},stopPropagation(){}});
 const savedRows=layout.style.gridTemplateRows;
 get('stage').onpointerdown({button:0,clientX:0,clientY:0,pointerId:1});get('stage').onpointermove({clientX:30,clientY:10});get('stage').onpointerup();
 get('stage').listeners.wheel({preventDefault(){},deltaY:-40,deltaMode:0});
 const savedCamera=JSON.parse(vm.runInContext('JSON.stringify({yaw:manualYaw,elevation:manualElevation,zoom})',initial.context));
 initial.context.setMaximized(true);await changePreference('sound',true);initial.windowEvents.pagehide({persisted:false});
 assert(!savedValues.get(preferenceKey).includes('test-token'),'visual preferences never store the API token');
 const restored=reloadArena(storage);
 assert.equal(get('show-feeds').checked,true);assert(!layout.classes.has('feeds-hidden'));assert.equal(get('show-grid').checked,true);
 assert.equal(get('show-trails').checked,false);assert.equal(get('show-effects').checked,false);assert.equal(get('show-labels').value,'full');
 assert.equal(get('height-scale').value,'3');assert.equal(get('height-note').textContent,'3');assert.equal(get('quality').value,'standard');
 assert.equal(get('aircraft-size').value,'0');assert.equal(get('scale-note').textContent,'ACTUAL SCALE');
 assert.equal(get('music-volume').value,'20');assert.equal(get('effects-volume').value,'30');assert.deepEqual(restored.audio.volume.at(-1),[.2,.3]);
 assert.equal(layout.style.gridTemplateRows,savedRows,'reload restores row height proportions');
 assert.equal(get('blue-feeds').children[0].dataset.view,'hud','reload restores each aircraft view mode');
 const restoredCamera=JSON.parse(vm.runInContext('JSON.stringify({yaw:manualYaw,elevation:manualElevation,zoom})',restored.context));
 for(const key of ['yaw','elevation','zoom'])assert(Math.abs(restoredCamera[key]-savedCamera[key])<1e-12,'camera controls survive reload');
 assert(restored.page.body.classes.has('maximized'),'reload restores layout maximization without requesting native fullscreen');
 assert.equal(get('sound').checked,true);assert.equal(restored.audio.enabled.length,0,'saved audio preference waits for a user gesture');
 assert.match(get('message').textContent,/クリックで音声を再開/);await restored.events.click({target:get('settings')});
 assert.deepEqual(restored.audio.enabled,[true]);assert.equal(restored.audio.active.at(-1),true,'the first click resumes saved audio');
 const pending=reloadArena(storage),soundLabel=new Node('label');soundLabel.contains=target=>target===soundLabel||target===get('sound');get('sound').parentElement=soundLabel;
 await pending.events.click({target:soundLabel});assert.equal(pending.audio.enabled.length,0,'clicking the sound label cannot start audio before switching it off');
 await changePreference('sound',false);await pending.events.click({target:get('settings')});assert(!pending.audio.enabled.includes(true));
 assert.equal(JSON.parse(savedValues.get(preferenceKey)).sound,false,'the off preference is saved too');
 get('blue-resize').ondblclick();get('reset-camera').onclick();
 const reset=reloadArena(storage);assert.equal(layout.style.gridTemplateRows,'','reset heights remain reset after reload');assert.equal(vm.runInContext('zoom',reset.context),1);
 savedValues.set(preferenceKey,'{broken');reloadArena(storage);assert.equal(get('show-feeds').checked,false);assert.equal(get('sound').checked,false,'invalid JSON falls back to first-use defaults');
 savedValues.set(preferenceKey,JSON.stringify({options:{feeds:'yes',grid:true,trails:false,height:1000,aircraftSize:40,quality:'ultra',labels:'bad'},rows:[1,'bad',3],modes:{blue_0:'invalid'},musicVolume:5,effectsVolume:-1,sound:'true',camera:{elevation:-3,zoom:0}}));
 const invalid=reloadArena(storage);assert.equal(get('show-grid').checked,true);assert.equal(get('show-trails').checked,false,'valid fields survive alongside invalid ones');
 assert.equal(get('height-scale').value,'1');assert.equal(get('aircraft-size').value,'18');assert.equal(get('show-labels').value,'simple');assert.equal(get('quality').value,'high');
 assert.equal(get('music-volume').value,'45');assert.equal(get('effects-volume').value,'70');assert.equal(get('sound').checked,false);assert.equal(invalid.audio.enabled.length,0);
 assert.equal(layout.style.gridTemplateRows,'');assert.equal(get('blue-feeds').children[0].dataset.view,'chase');assert.equal(vm.runInContext('zoom',invalid.context),1);
 const blocked={getItem(){throw Error('storage blocked');},setItem(){throw Error('quota exceeded');}};
 reloadArena(blocked);await changePreference('show-feeds',true);assert(!layout.classes.has('feeds-hidden'),'storage failures never disable the controls');
 console.log('PASS: match start/pause/resume/stop, saved setup, serialized requests, persistent errors and recorded frame stepping');
 console.log('PASS: preference reload, volumes, per-aircraft HSD, row sizes, camera, maximization, gesture-based audio and unavailable storage');
 console.log('PASS: WebGL attitude, simultaneous cameras, arena framing, playback, feed visibility/resizing, maximize/restore and connection recovery');
})().catch(error=>{console.error(error);process.exitCode=1;});
