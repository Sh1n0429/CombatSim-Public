import {createArenaRenderer} from './arena-webgl.js';
import {createArenaAudio,crossedAudioEvents} from './arena-audio.js';
const $ = id => document.getElementById(id);
const {clamp,mix,teamColor,aircraftColor,aircraftLabel,flightTelemetry,clock,sample} = ArenaScene;
const frames = [], effects = [], feeds = [];
const launches=new Map();
let config, matchId, matchState, follow=true, paused=false;
let shownTick=0, liveFrom=0, liveTo=0, liveAt=0, lastPaint=0, lastHUD=0, feedKey='';
let manualYaw=.2, manualElevation=.52, zoom=1, drag=null, busy=false, sound=false, soundPending=false, soundTick=null, errors=0, pollTimer, view, cameraInitialized=false;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const camera={target:[0,0,60],yaw:.2,elevation:.52,distance:600};
const framingHalf=[0,0,0];
const options={height:1,feeds:false,grid:false,trails:true,labels:'simple',effects:!reducedMotion,quality:'high',aircraftSize:18};
const preferenceKey='combatsim.arena.preferences',preferences=readPreferences(),panelModes=new Map(preferences.modes||[]);
let feedRows=preferences.rows||null;
const audio=createArenaAudio();
let controlPending=false, controlRevision=0, controlError='';
function readPreferences(){
  try{
    const saved=JSON.parse(localStorage.getItem(preferenceKey)||'null');
    if(!saved||typeof saved!=='object'||Array.isArray(saved))return {};
    for(const key of ['feeds','grid','trails','effects'])if(typeof saved.options?.[key]==='boolean')options[key]=saved.options[key];
    for(const [key,values] of Object.entries({height:[1,3,5],labels:['simple','full','off'],quality:['high','standard'],aircraftSize:[0,18,28]})){
      if(values.includes(saved.options?.[key]))options[key]=saved.options[key];
    }
    const camera=saved.camera||{};
    if(Number.isFinite(camera.yaw))manualYaw=Math.atan2(Math.sin(camera.yaw),Math.cos(camera.yaw));
    if(Number.isFinite(camera.elevation)&&camera.elevation>=.12&&camera.elevation<=1.45)manualElevation=camera.elevation;
    if(Number.isFinite(camera.zoom)&&camera.zoom>=.5&&camera.zoom<=2.4)zoom=camera.zoom;
    return {
      sound:saved.sound===true,maximized:saved.maximized===true,
      musicVolume:Number.isFinite(saved.musicVolume)&&saved.musicVolume>=0&&saved.musicVolume<=1?saved.musicVolume:.45,
      effectsVolume:Number.isFinite(saved.effectsVolume)&&saved.effectsVolume>=0&&saved.effectsVolume<=1?saved.effectsVolume:.7,
      rows:Array.isArray(saved.rows)&&saved.rows.length===3&&saved.rows.every(n=>Number.isFinite(n)&&n>0&&n<=1e6)?saved.rows:null,
      modes:saved.modes&&typeof saved.modes==='object'&&!Array.isArray(saved.modes)?Object.entries(saved.modes).filter(([,mode])=>['chase','hud'].includes(mode)):[]
    };
  }catch{return {};}
}
function savePreferences(){
  try{localStorage.setItem(preferenceKey,JSON.stringify({options,sound:$('sound').checked,
    musicVolume:Number($('music-volume').value)/100,effectsVolume:Number($('effects-volume').value)/100,
    rows:feedRows,modes:Object.fromEntries(panelModes),camera:{yaw:manualYaw,elevation:manualElevation,zoom},
    maximized:document.body.classList.contains('maximized')}));}catch{/* Storage may be unavailable in private or embedded browsers. */}
}
function syncControls(){
  const running=['running','paused'].includes(matchState.status),active=running||matchState.status==='stopping';
  $('match-start').disabled=controlPending||active;
  $('match-pause').disabled=$('match-stop').disabled=controlPending||!running;
  $('match-pause').textContent=matchState.status==='paused'?'対戦を再開':'対戦を一時停止';
  const select=$('arena-pace');
  select.value=String(matchState.pace??10);
  select.disabled=controlPending||!['running','paused'].includes(matchState.status);
}
function consume(state,initial=false) {
  const reset=matchId!==state.match_id;
  if(reset){
    frames.length=effects.length=0;launches.clear();feedKey='';follow=true;paused=false;cameraInitialized=false;soundTick=null;view?.reset();
    if(!initial&&!reducedMotion)$('viewports').animate?.([{opacity:0},{opacity:1}],{duration:1200,easing:'ease-out'});
  }
  config=state.config;matchId=state.match_id;matchState=state;
  if(!controlPending)syncControls();
  const incoming=state.new_frames||[state.frame];
  for(const frame of incoming){
    if(frames.length&&frame.tick<=frames.at(-1).tick)continue;
    frames.push(frame);
    for(const event of frame.events||[]){
      if(event.type==='launch'&&event.missile){
        const ids=launches.get(event.aircraft)||[];ids.push(event.missile);launches.set(event.aircraft,ids);
      }
      if(event.type==='hit'||event.type==='crash'){
        const entity=frame.aircraft.find(a=>a.id===(event.target||event.aircraft));
        if(entity)effects.push({tick:frame.tick,position:entity,type:event.type});
      }
    }
  }
  // shortcut: keep the current match in memory, use saved replays for previous matches.
  if(effects.length>100)effects.splice(0,effects.length-100);
  if(!frames.length)frames.push(state.frame);
  const now=performance.now();
  if(reset||initial){shownTick=liveFrom=liveTo=state.frame.tick;liveAt=now;}
  else if(state.frame.tick!==liveTo){liveFrom=shownTick;liveTo=state.frame.tick;liveAt=now;}
  const names=state.models||{};
  for(const side of ['blue','red']){
    const node=$(side+'-name');node.textContent=names[side]?.policy||`${side.toUpperCase()} TEAM`;node.title=node.textContent;
  }
  $('seed').textContent=`SEED ${state.seed} / ${matchId==='preview'?'STANDBY':'MATCH '+matchId.split(':').at(-1)}`;
  const unit=config.aero_length_scale||250;
  $('airspace-size').textContent=`${(config.arena_x*unit/1000).toFixed(0)} × ${(config.arena_y*unit/1000).toFixed(0)} KM / COMBAT ZONE`;
  $('waiting').hidden=state.status!=='idle';
  errors=0;connection(state.status);
  hud(sample(frames,shownTick).frame);
}
function connection(status) {
  const names={idle:'STANDBY',running:'LIVE',paused:'PAUSED',completed:'COMPLETE',stopping:'STOPPING',stopped:'STOPPED',failed:'ERROR',error:'RECONNECTING'};
  $('connection').dataset.status=status;$('connection').lastElementChild.textContent=names[status]||status.toUpperCase();
  if(status!=='error'&&controlError){$('message').textContent=controlError;return;}
  if(status!=='error')$('message').textContent=status==='failed'?'対戦でエラーが発生しました。「対戦設定」で詳細を確認してください。':status==='stopped'?'対戦は中断されました。「対戦開始」で保存した設定の対戦を開始できます。':status==='idle'?'「対戦開始」で保存した設定の対戦を開始できます。モデルや条件は「対戦設定」で変更できます。':'AI AIR COMBAT · 実際のシミュレーションを配信中';
  if(status!=='error'&&soundPending)$('message').textContent+=' · クリックで音声を再開';
}
function renderFeeds(frame) {
  const key=frame.aircraft.map(a=>`${a.team}:${a.id}:${aircraftLabel(a)+':'+aircraftColor(a)}`).join('|');
  if(key!==feedKey){
    const modes=new Map(feeds.map(feed=>[feed.id,feed.mode]));
    feedKey=key;feeds.length=0;
    for(const team of ['blue','red']){
      const row=$(team+'-feeds'),aircraft=frame.aircraft.filter(a=>a.team===team);
      row.replaceChildren();row.style.gridTemplateColumns=`repeat(${Math.max(1,aircraft.length)},minmax(180px,1fr))`;
      for(const a of aircraft){
        const node=document.createElement('article');node.className='aircraft-feed';node.dataset.id=a.id;node.style.borderColor=aircraftColor(a);node.setAttribute('aria-label',`${aircraftLabel(a)} の追尾映像`);
        const name=document.createElement('h3');name.className='feed-name';name.textContent=aircraftLabel(a);name.style.color=aircraftColor(a);
        const footer=document.createElement('div');footer.className='feed-footer';
        const data=document.createElement('div');data.className='feed-data';data.setAttribute('role','img');
        const charts=[];
        for(const [key,label,unit,color,minSpan] of [['altitude','ALT','M','#78e2ed',100],['speed','SPD','KM/H','#e8c68c',50]]){
          const row=document.createElement('div');row.className='telemetry-row';row.style.color=color;row.setAttribute('aria-hidden','true');
          const value=document.createElement('span');value.className='telemetry-value';
          const chart=document.createElementNS('http://www.w3.org/2000/svg','svg');chart.setAttribute('class','telemetry-trend');chart.setAttribute('viewBox','0 0 100 24');chart.setAttribute('preserveAspectRatio','none');
          const line=document.createElementNS('http://www.w3.org/2000/svg','polyline'),dot=document.createElementNS('http://www.w3.org/2000/svg','circle');dot.setAttribute('r','2');
          chart.append(line,dot);row.append(value,chart);data.append(row);charts.push({key,label,unit,minSpan,row,value,line,dot});
        }
        const ammo=document.createElement('div');ammo.className='feed-ammo';ammo.style.color=aircraftColor(a);ammo.setAttribute('role','img');
        const count=document.createElement('span');count.className='ammo-count';count.setAttribute('aria-hidden','true');
        const rounds=document.createElement('div');rounds.className='ammo-rounds';rounds.setAttribute('aria-hidden','true');
        const capacity=frames[0].aircraft.find(item=>item.id===a.id)?.ammo??a.ammo,icons=[];
        for(let i=0;i<capacity;i++){
          const icon=document.createElementNS('http://www.w3.org/2000/svg','svg'),use=document.createElementNS('http://www.w3.org/2000/svg','use');
          use.setAttribute('href','#i-missile');icon.append(use);rounds.append(icon);icons.push(icon);
        }
        ammo.append(count,rounds);
        const status=document.createElement('div');status.className='feed-lost';
        const hud=document.createElement('canvas');hud.className='feed-hud';hud.setAttribute('role','img');
        const controls=document.createElement('div');controls.className='feed-modes';controls.setAttribute('role','group');controls.setAttribute('aria-label',`${aircraftLabel(a)} の表示切替`);
        const feed={id:a.id,node,clip:row,data,charts,ammo,count,icons,status,hud,mode:modes.get(a.id)||panelModes.get(a.id)||'chase',buttons:[]};
        function selectMode(mode){
          feed.mode=mode;node.dataset.view=mode;hud.hidden=mode!=='hud';
          node.setAttribute('aria-label',`${aircraftLabel(a)} の${mode==='hud'?'BVR戦術HSD':'追尾映像'}`);
          for(const button of feed.buttons)button.setAttribute('aria-pressed',String(button.dataset.mode===mode));
        }
        for(const [mode,label] of [['chase','3D'],['hud','HSD']]){
          const button=document.createElement('button');button.type='button';button.textContent=label;button.dataset.mode=mode;
          button.setAttribute('aria-label',`${aircraftLabel(a)} を${mode==='hud'?'BVR戦術HSD':'3D追尾'}に切り替え`);
          button.onclick=()=>{selectMode(mode);panelModes.set(a.id,mode);savePreferences();};feed.buttons.push(button);controls.append(button);
        }
        selectMode(feed.mode);
        footer.append(data,ammo);node.append(hud,status,name,footer,controls);row.append(node);feeds.push(feed);
      }
    }
  }
  const flying=new Set(frame.missiles.filter(m=>m.alive!==false).map(m=>m.id));
  for(const feed of feeds){
    const a=frame.aircraft.find(a=>a.id===feed.id);
    const values=flightTelemetry(a,config);
    const loss=effects.find(e=>e.position.id===a.id&&e.tick<=frame.tick);
    const history=ArenaScene.flightHistory(frames,a.id,!a.alive&&loss?loss.tick:frame.tick,config);
    if(!history.points.length)history.points.push({tick:history.to,...values});
    const duration=(history.to-history.from)*(config.aero_tick_seconds||1);
    const summary=`${a.alive?'':'LAST TELEMETRY · '}${values.altitude.toLocaleString()} M · ${values.speed} KM/H`;
    feed.data.setAttribute('aria-label',`${summary}、高度・速度の直近${Math.round(duration)}秒の推移`);
    for(const metric of feed.charts){
      let min=Infinity,max=-Infinity;
      for(const point of history.points){min=Math.min(min,point[metric.key]);max=Math.max(max,point[metric.key]);}
      const span=Math.max(metric.minSpan,max-min);
      const low=Math.min(min,Math.max(0,min-(span-(max-min))/2));
      const stride=Math.max(1,Math.ceil(history.points.length/120));
      const points=history.points.filter((_,i)=>i%stride===0||i===history.points.length-1).map(point=>[
        history.to===history.from?100:(point.tick-history.from)/(history.to-history.from)*100,22-(point[metric.key]-low)/span*20]);
      metric.value.textContent=`${metric.label} ${values[metric.key].toLocaleString()} ${metric.unit}`;
      metric.row.title=`直近${Math.round(duration)}秒 · ${Math.round(low)}–${Math.round(low+span)} ${metric.unit}`;
      metric.line.setAttribute('points',points.map(point=>point.map(n=>n.toFixed(2)).join(',')).join(' '));
      const last=points.at(-1);metric.dot.setAttribute('cx',String(last[0]));metric.dot.setAttribute('cy',String(last[1]));
    }
    const ids=launches.get(a.id)||[];let airborne=0;
    for(const [i,icon] of feed.icons.entries()){
      const inFlight=i>=a.ammo&&flying.has(ids[feed.icons.length-1-i]);
      icon.classList.toggle('in-flight',inFlight);icon.classList.toggle('spent',i>=a.ammo&&!inFlight);
      icon.style.animationPlayState=paused?'paused':'running';if(inFlight)airborne++;
    }
    feed.count.textContent=`${a.ammo}/${feed.icons.length}`;feed.ammo.setAttribute('aria-label',`ミサイル残弾 ${a.ammo} / ${feed.icons.length}、飛翔中 ${airborne}`);

  }
  for(const team of ['blue','red']){
    const aircraft=frame.aircraft.filter(a=>a.team===team);
    $(team+'-feed-count').textContent=`${aircraft.filter(a=>a.alive).length} / ${aircraft.length}`;
  }
}
function hud(frame) {
  const unit=config.aero_tick_seconds||1,latest=frames.at(-1).tick;
  for(const team of ['blue','red']){
    const aircraft=frame.aircraft.filter(a=>a.team===team),alive=aircraft.filter(a=>a.alive);
    const count=$(team+'-alive');count.replaceChildren(document.createTextNode(alive.length+' '));const total=document.createElement('small');total.textContent='/ '+aircraft.length;count.append(total);
    const strength=$(team+'-strength');strength.replaceChildren(...aircraft.map(a=>{const n=document.createElement('i');n.classList.toggle('lost',!a.alive);return n;}));
    $(team+'-ammo').textContent=alive.reduce((n,a)=>n+a.ammo,0);
    $(team+'-missiles').textContent=frame.missiles.filter(m=>m.team===team).length;
  }
  $('clock').textContent=clock(frame.tick*unit);
  const atEnd=frame.tick>=latest,completed=['running','paused','completed'].includes(matchState.status)&&matchState.done&&atEnd;
  $('phase').textContent=paused?'映像を一時停止中':!follow?'過去の映像を再生中':completed?'対戦終了':({idle:'対戦を待っています',paused:'対戦を一時停止中',stopped:'対戦中断',failed:'対戦エラー',stopping:'停止処理中'})[matchState.status]||'交戦中';
  $('clock-label').textContent=follow?'SIMULATION TIME':'PLAYBACK TIME';
  $('frame-info').textContent=`TICK ${String(Math.floor(frame.tick)).padStart(4,'0')} · ${options.quality==='high'?'HD':'SD'} RENDER`;
  $('timeline').max=latest;$('timeline').min=frames[0].tick;$('timeline').value=frame.tick;
  $('timeline-label').textContent=`${clock(frame.tick*unit)} / ${clock(latest*unit)}`;
  $('go-live').setAttribute('aria-pressed',String(follow));$('go-live').lastElementChild.textContent='最新へ';
  $('play-icon').setAttribute('href',paused?'#i-play':'#i-pause');$('play-label').textContent=paused?'映像再生':'映像停止';$('play').setAttribute('aria-label',paused?'映像を再生':'映像を一時停止');$('play').disabled=matchState.status==='idle';
  const index=sample(frames,frame.tick).index;
  $('back-ten').disabled=$('back').disabled=index===0;
  $('step').disabled=index===frames.length-1;
  $('timeline').disabled=frames.length<2;
  $('result').classList.toggle('show',completed);
  if(completed){const winner=matchState.winner;$('result-title').textContent=winner?`${winner.toUpperCase()} WIN`:'DRAW';$('result-title').style.color=winner?teamColor(winner):'#dfc18b';}
  renderFeeds(frame);
}
function updateCamera(frame,dt) {
  const rect=$('stage').getBoundingClientRect();
  const aspect=rect.width/Math.max(1,rect.height),desired=ArenaScene.battleView(frame,effects,config,options.height,aspect,manualYaw,manualElevation);
  const smoothing=cameraInitialized?1-Math.exp(-dt/2200):1;
  // Smooth the engagement bounds so manual rotation and zoom stay immediate.
  for(let i=0;i<3;i++){
    camera.target[i]=mix(camera.target[i],desired.target[i],smoothing);
    if(desired.half)framingHalf[i]=mix(framingHalf[i],desired.half[i],cameraInitialized?1-Math.exp(-dt/(desired.half[i]>framingHalf[i]?700:3500)):1);
  }
  camera.yaw=manualYaw;camera.elevation=manualElevation;
  camera.distance=(desired.half?ArenaScene.fitDistance(...framingHalf,aspect,manualYaw,manualElevation):desired.distance)/zoom;
  cameraInitialized=true;
}
function animate(now) {
  const dt=Math.min(now-lastPaint||16,80);lastPaint=now;
  if(frames.length){
    if(!paused){
      if(follow){
        const duration=matchState.pace==='realtime'?Math.max(1,(liveTo-liveFrom)*(config.aero_tick_seconds||1)*1000):300;
        shownTick=mix(liveFrom,liveTo,clamp((now-liveAt)/duration,0,1));
      }
      else{shownTick=Math.min(frames.at(-1).tick,shownTick+dt/1000*(matchState.pace==='realtime'?1/(config.aero_tick_seconds||1):30));if(shownTick>=frames.at(-1).tick){follow=true;liveFrom=liveTo=shownTick;}}
    }
    const {frame,index}=sample(frames,shownTick);
    if(syncAudio())for(const {event,frame:recorded} of crossedAudioEvents(frames,soundTick,shownTick)){
      const entity=recorded.aircraft.find(a=>a.id===(event.target||event.aircraft));
      audio.play(event.type,entity?(entity.x/config.arena_x-.5)*1.5:0);
    }
    soundTick=shownTick;
    updateCamera(frame,dt);view?.draw(frame,frames,index,config,camera,options,effects,options.feeds?feeds:[],dt,paused);
    if(now-lastHUD>160){hud(frame);lastHUD=now;}
  }
  requestAnimationFrame(animate);
}
function syncAudio(){
  const active=sound&&!paused&&!document.hidden&&['running','paused','completed'].includes(matchState?.status);
  audio.setActive(active);return active;
}
function togglePause(){if(matchState.status==='idle')return;paused=!paused;syncAudio();if(!paused&&follow){liveFrom=shownTick;liveTo=frames.at(-1).tick;liveAt=performance.now();}hud(sample(frames,shownTick).frame);}
function seekTick(tick){
  shownTick=clamp(tick,frames[0].tick,frames.at(-1).tick);soundTick=null;
  audio.setActive(false);follow=false;paused=true;hud(sample(frames,shownTick).frame);
}
function stepFrames(delta){const index=sample(frames,shownTick).index;seekTick(frames[clamp(index+delta,0,frames.length-1)].tick);}
function goLive(){soundTick=null;follow=true;paused=false;shownTick=liveFrom=liveTo=frames.at(-1).tick;liveAt=performance.now();syncAudio();hud(sample(frames,shownTick).frame);}
function feedHeights(){return ['red-views','stage','blue-views'].map(id=>$(id).getBoundingClientRect().height);}
function applyFeedRows(){
  $('viewports').style.gridTemplateRows=feedRows?feedRows.map((value,i)=>`minmax(${i===1?130:76}px,${value}fr)`).join(' '):'';
}
function syncFeedHeights(){
  if(!options.feeds)return;
  const heights=feedHeights();
  for(const [id,index] of [['red-resize',0],['blue-resize',2]]){
    const handle=$(id);handle.setAttribute('aria-valuemin','76');handle.setAttribute('aria-valuemax',String(Math.round(heights[index]+heights[1]-130)));
    handle.setAttribute('aria-valuenow',String(Math.round(heights[index])));handle.setAttribute('aria-valuetext',`${Math.round(heights[index])}px`);
  }
}
function setFeedHeight(index,height){
  const heights=feedHeights(),available=heights[index]+heights[1];
  heights[index]=clamp(height,76,available-130);heights[1]=available-heights[index];
  // Fractional tracks retain the chosen proportions when the viewport changes.
  feedRows=heights;applyFeedRows();
  syncFeedHeights();
}
function setMaximized(enabled){
  document.body.classList.toggle('maximized',enabled);
  $('fullscreen').setAttribute('aria-pressed',String(enabled));
  const label=enabled?'元に戻す':'最大化';
  $('fullscreen-label').textContent=label;$('fullscreen').setAttribute('aria-label',enabled?'観戦画面の最大化を終了':'観戦画面を最大化');$('fullscreen').title=label+' (F)';
  syncFeedHeights();
  savePreferences();
}
async function fullscreen(){
  const enabled=!document.body.classList.contains('maximized');setMaximized(enabled);
  try{
    if(enabled&&!document.fullscreenElement)await document.documentElement.requestFullscreen?.();
    else if(!enabled&&document.fullscreenElement)await document.exitFullscreen();
  }catch{/* Embedded browsers can still expand the layout when native fullscreen is unavailable. */}
}
async function poll(){
  clearTimeout(pollTimer);
  if(busy||document.hidden)return;busy=true;
  const revision=controlRevision;
  try{const response=await fetch(`/api/watch?match=${encodeURIComponent(matchId)}&after=${frames.at(-1)?.tick??-1}`,{signal:AbortSignal.timeout(5000)});if(!response.ok)throw Error(`HTTP ${response.status}`);const state=await response.json();if(revision===controlRevision)consume(state);}
  catch(error){errors++;connection('error');$('message').textContent=`配信との接続が途切れました。再接続中… (${error.message})`;}
  finally{busy=false;pollTimer=setTimeout(poll,errors?1500:300);}
}
// UI bindings stay below the state/render functions for dependency-free checks.
async function arenaAPI(path,data){
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{
    'Content-Type':'application/json','X-CombatSim-Token':bootstrap.token
  },body:JSON.stringify(data)});
  const result=await response.json();
  if(!response.ok)throw Error(result.error||response.statusText);
  return result;
}
async function matchSetup(){
  try{
    const saved=JSON.parse(localStorage.getItem('combatsim.evaluation.setup'));
    if(saved?.request&&typeof saved.request==='object'&&!Array.isArray(saved.request))return saved.request;
  }catch{/* The default setup also works without browser storage. */}
  const state=await arenaAPI('/api/state'),learned=state.catalog.filter(m=>m.kind!=='rule');
  return {blue_model:learned[0]?.id||state.catalog[0]?.id,red_model:learned[1]?.id||'rule:fix'};
}
async function controlMatch(path,data){
  if(controlPending)return;
  controlPending=true;controlRevision++;controlError='';syncControls();
  try{
    const result=await arenaAPI(path,path==='/api/start'?await matchSetup():data);
    matchState.status=result.job?.status??matchState.status;
    matchState.pace=result.job?.pace??matchState.pace;
    if(path==='/api/pace'){
      try{
        const saved=JSON.parse(localStorage.getItem('combatsim.evaluation.setup'));
        if(saved?.request){saved.request.pace=matchState.pace;if(saved.fields)saved.fields.pace=String(matchState.pace);localStorage.setItem('combatsim.evaluation.setup',JSON.stringify(saved));}
      }catch{/* Storage failures must not interrupt an active match. */}
    }
    connection(matchState.status);
  }catch(error){controlError='対戦操作を実行できませんでした。 '+error.message+' 条件は「対戦設定」で確認できます。';$('message').textContent=controlError;}
  finally{controlPending=false;controlRevision++;syncControls();await poll();}
}
$('arena-pace').onchange=()=>{const value=$('arena-pace').value;return controlMatch('/api/pace',{pace:value==='realtime'?value:Number(value)});};
$('match-start').onclick=()=>controlMatch('/api/start');
$('match-pause').onclick=()=>controlMatch('/api/pause',{paused:matchState.status!=='paused'});
$('match-stop').onclick=()=>controlMatch('/api/stop',{});
$('play').onclick=togglePause;$('go-live').onclick=goLive;$('fullscreen').onclick=fullscreen;
$('timeline').oninput=event=>seekTick(Number(event.target.value));
$('back-ten').onclick=()=>stepFrames(-10);$('back').onclick=()=>stepFrames(-1);$('step').onclick=()=>stepFrames(1);
$('reset-camera').onclick=()=>{manualYaw=.2;manualElevation=.52;zoom=1;savePreferences();};
$('settings').onclick=()=>$('display-settings').showModal();$('close-settings').onclick=()=>$('display-settings').close();
$('display-settings').addEventListener('change',savePreferences);
soundPending=preferences.sound===true;$('sound').checked=soundPending;
$('sound').onchange=async event=>{
  const control=event.target,enabled=control.checked;control.disabled=true;soundPending=false;
  try{await audio.enable(enabled);sound=enabled;soundTick=shownTick;syncAudio();}
  catch(error){control.checked=sound;$('message').textContent='音声を読み込めませんでした。もう一度オンにしてください。 '+error.message;}
  finally{control.disabled=false;savePreferences();}
};
$('music-volume').value=String(Math.round((preferences.musicVolume??.45)*100));$('effects-volume').value=String(Math.round((preferences.effectsVolume??.7)*100));
audio.volume(Number($('music-volume').value)/100,Number($('effects-volume').value)/100);
for(const id of ['music-volume','effects-volume'])$(id).oninput=()=>{audio.volume(Number($('music-volume').value)/100,Number($('effects-volume').value)/100);savePreferences();};
$('show-feeds').checked=options.feeds;
$('viewports').classList.toggle('feeds-hidden',!options.feeds);
$('show-feeds').onchange=event=>{
  options.feeds=event.target.checked;$('viewports').classList.toggle('feeds-hidden',!options.feeds);syncFeedHeights();
};
for(const [id,key] of [['show-grid','grid'],['show-trails','trails'],['show-effects','effects']]){$(id).checked=options[key];$(id).onchange=event=>options[key]=event.target.checked;}
$('show-labels').value=options.labels;
$('show-labels').onchange=event=>options.labels=event.target.value;
$('height-scale').value=String(options.height);$('height-note').textContent=String(options.height);
$('height-scale').onchange=event=>{options.height=Number(event.target.value);$('height-note').textContent=event.target.value;};
$('quality').value=options.quality;
$('quality').onchange=event=>options.quality=event.target.value;
$('aircraft-size').value=String(options.aircraftSize);
$('scale-note').textContent=options.aircraftSize===0?'ACTUAL SCALE':'SCALE ENHANCED';
$('aircraft-size').onchange=event=>{
  options.aircraftSize=Number(event.target.value);$('scale-note').textContent=options.aircraftSize===0?'ACTUAL SCALE':'SCALE ENHANCED';
};
for(const [id,index] of [['red-resize',0],['blue-resize',2]]){
  const handle=$(id),direction=index===0?1:-1;let start=null;
  handle.onpointerdown=event=>{
    if(event.button!==0)return;
    event.preventDefault();handle.focus();handle.setPointerCapture(event.pointerId);
    start={y:event.clientY,height:feedHeights()[index],pointerId:event.pointerId};$('viewports').classList.add('resizing');
  };
  handle.onpointermove=event=>{if(start&&event.pointerId===start.pointerId)setFeedHeight(index,start.height+(event.clientY-start.y)*direction);};
  handle.onpointerup=handle.onpointercancel=handle.onlostpointercapture=event=>{if(start&&event.pointerId===start.pointerId){start=null;$('viewports').classList.remove('resizing');savePreferences();}};
  handle.onkeydown=event=>{if(event.key==='ArrowUp'||event.key==='ArrowDown'){event.preventDefault();event.stopPropagation();setFeedHeight(index,feedHeights()[index]+(event.key==='ArrowDown'?10:-10)*direction);savePreferences();}};
  handle.ondblclick=()=>{feedRows=null;applyFeedRows();syncFeedHeights();savePreferences();};
  handle.onfocus=syncFeedHeights;
}
$('stage').onpointerdown=event=>{if(event.button!==0)return;drag={x:event.clientX,y:event.clientY};$('stage').setPointerCapture(event.pointerId);};
$('stage').onpointermove=event=>{if(!drag)return;manualYaw-=(event.clientX-drag.x)*.006;manualElevation=clamp(manualElevation+(event.clientY-drag.y)*.005,.12,1.45);drag={x:event.clientX,y:event.clientY};};
$('stage').onpointerup=$('stage').onpointercancel=$('stage').onlostpointercapture=()=>{drag=null;savePreferences();};
$('stage').addEventListener('wheel',event=>{
  event.preventDefault();
  const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?$('stage').getBoundingClientRect().height:1);
  zoom=clamp(zoom-delta*.0015,.5,2.4);savePreferences();
},{passive:false});
document.addEventListener('click',event=>{
  if(soundPending&&$('sound').checked&&event.target!==$('sound')&&!$('sound').parentElement?.contains(event.target))return $('sound').onchange({target:$('sound')});
});
document.addEventListener('keydown',event=>{
  if($('display-settings').open)return;
  if(event.code==='Escape'){if(!document.fullscreenElement)setMaximized(false);return;}
  if(event.target.matches('input,select,textarea,[contenteditable=true]'))return;
  if(event.code==='Space'&&!event.target.matches('button,a')){event.preventDefault();togglePause();}else if(event.code==='KeyF')fullscreen();
});
document.addEventListener('visibilitychange',()=>{soundTick=null;syncAudio();if(!document.hidden&&!busy)poll();});
document.addEventListener('fullscreenchange',()=>setMaximized(Boolean(document.fullscreenElement)));
window.addEventListener('resize',syncFeedHeights);
applyFeedRows();setMaximized(Boolean(document.fullscreenElement)||preferences.maximized===true);
consume(bootstrap,true);
try{view=createArenaRenderer($('scene'),$('scene-overlay'),$('stage'));view.ready.catch(error=>{$('render-error').hidden=false;$('render-error').textContent='精細モデルを読み込めませんでした。画面を再読み込みしてください。 '+error.message;});}
catch(error){$('render-error').hidden=false;$('render-error').textContent='3D描画を開始できませんでした。WebGL 2とブラウザのハードウェアアクセラレーションを確認してください。 '+error.message;}
requestAnimationFrame(animate);pollTimer=setTimeout(poll,300);
window.addEventListener('pagehide',event=>{savePreferences();soundTick=null;audio.setActive(false);if(!event.persisted){view?.dispose();audio.dispose();}});
