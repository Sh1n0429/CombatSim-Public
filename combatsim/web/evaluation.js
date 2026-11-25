'use strict';
const $=id=>document.getElementById(id);
let state=bootstrap.state, catalogKey='', previewTimer, previewVersion=0, polling=false, lastFinished='', retrying=false;
for(const [key,id] of Object.entries({seed_start:'seed-start',seed_count:'seed-count',blue_count:'blue-count',red_count:'red-count',aircraft_per_team:'aircraft-per-team',action_mode:'action-mode',pace:'pace'})) {
  const node=$(id), value=state.defaults[key];
  if(node.tagName==='SELECT' && ![...node.options].some(o=>o.value===String(value))) node.add(new Option(String(value),String(value)));
  node.value=String(value);
}
const setupKey='combatsim.evaluation.setup';
const setupFields=['mode','blue-model','red-model','blue-count','red-count','aircraft-per-team','single-match','seed-start','seed-count','swap','pace','seeds','action-mode','team-a','team-b','replays'];
function saveSetup(){
  const fields=Object.fromEntries(setupFields.map(id=>{const node=$(id);return [id,node.type==='checkbox'?node.checked:node.value];}));
  const models=[...$('round-models').querySelectorAll('input:checked')].map(n=>n.value);
  const lineups=Object.fromEntries(['blue','red'].map(side=>[side,[...$(side+'-lineup').querySelectorAll('select')].map(n=>n.value)]));
  try{localStorage.setItem(setupKey,JSON.stringify({fields,models,lineups,request:request()}));}catch{/* Browser storage may be unavailable. */}
}
function restoreSetup(){
  let saved;
  try{saved=JSON.parse(localStorage.getItem(setupKey));}catch{return;}
  if(!saved||typeof saved!=='object')return;
  for(const id of setupFields){
    const node=$(id),value=saved.fields?.[id];
    if(node.type==='checkbox'){if(typeof value==='boolean')node.checked=value;}
    else if(typeof value==='string'&&(node.tagName!=='SELECT'||[...node.options].some(o=>o.value===value)))node.value=value;
  }
  if(Array.isArray(saved.models))for(const node of $('round-models').querySelectorAll('input'))node.checked=saved.models.includes(node.value);
  lineups();
  for(const side of ['blue','red'])[...$(side+'-lineup').querySelectorAll('select')].forEach((node,i)=>{
    const value=saved.lineups?.[side]?.[i];
    if([...node.options].some(o=>o.value===value))node.value=value;
  });
}
function showMode(){const mode=$('mode').value;$('duel-fields').classList.toggle('hidden',mode!=='duel');$('mixed-fields').classList.toggle('hidden',mode!=='mixed');$('round-fields').classList.toggle('hidden',mode!=='round_robin');$('team-fields').classList.toggle('hidden',mode==='round_robin');}
const active=()=>['running','paused','stopping'].includes(state.job?.status);
async function api(path,data){const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-CombatSim-Token':bootstrap.token},body:JSON.stringify(data)});const result=await response.json();if(!response.ok)throw Error(result.error||response.statusText);return result;}
function message(text){$('message').textContent=text;}
function option(model){const node=document.createElement('option');node.value=model.id;node.textContent=model.name;return node;}
function fillSelect(node,models,fallback){const previous=node.value;node.replaceChildren(...models.map(option));node.value=models.some(m=>m.id===previous)?previous:(fallback||models[0]?.id||'');}
function lineups(){for(const side of ['blue','red']){const parent=$(side+'-lineup'), count=Math.max(1,Math.min(31,Number($(side+'-count').value)||1));const values=[...parent.querySelectorAll('select')].map(n=>n.value);parent.replaceChildren();for(let i=0;i<count;i++){const label=document.createElement('label');label.textContent=`${side} ${i+1}`;const select=document.createElement('select');select.setAttribute('aria-label',`${side} ${i+1} 担当モデル`);fillSelect(select,state.catalog.filter(m=>m.kind!=='rule'),values[i]);parent.append(label,select);}}}
function catalog(){const key=JSON.stringify(state.catalog);if(key===catalogKey)return;const first=!catalogKey;catalogKey=key;const learned=state.catalog.filter(m=>m.kind!=='rule');fillSelect($('blue-model'),state.catalog,learned[0]?.id);fillSelect($('red-model'),state.catalog,learned[1]?.id||'rule:fix');const checked=new Set([...$('round-models').querySelectorAll('input:checked')].map(n=>n.value));$('round-models').replaceChildren();state.catalog.forEach(m=>{const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=m.id;input.checked=!first?checked.has(m.id):[learned[0]?.id,learned[1]?.id||'rule:fix'].includes(m.id);label.append(input,document.createTextNode(m.name));$('round-models').append(label);});lineups();if(first){restoreSetup();showMode();}saveSetup();$('model-count').textContent=`${learned.length} models`;$('catalog-errors').textContent=state.catalog_errors.join('\n');if(!learned.length){$('add-details').open=true;message('ルール同士で評価するか、学習モデルを追加してください。');}preview();}
function readPace(){const value=$('pace').value;return value==='realtime'?value:Number(value);}
function request(){const result={single_match:document.getElementById("single-match").checked,mode:$('mode').value,scenario:'simulator',blue_model:$('blue-model').value,red_model:$('red-model').value,blue_count:Number($('blue-count').value),red_count:Number($('red-count').value),aircraft_per_team:Number($('aircraft-per-team').value),models:[...$('round-models').querySelectorAll('input:checked')].map(n=>n.value),blue_lineup:[...$('blue-lineup').querySelectorAll('select')].map(n=>n.value),red_lineup:[...$('red-lineup').querySelectorAll('select')].map(n=>n.value),seed_start:Number($('seed-start').value),seed_count:Number($('seed-count').value),swap_sides:$('swap').checked,save_replays:$('replays').checked,pace:readPace(),action_mode:$('action-mode').value,team_names:[$('team-a').value,$('team-b').value]};if($('seeds').value.trim())result.seeds=$('seeds').value.trim().split(/[\s,、]+/).map(Number);return result;}
function preview(){clearTimeout(previewTimer);const version=++previewVersion;previewTimer=setTimeout(async()=>{try{const p=await api('/api/preview',request());if(version!==previewVersion)return;$('preview').textContent=`${p.pairs}カード × ${p.seeds.length} seeds${$('swap').checked?' × 青赤交替':''} = ${p.total}試合`;$('start').disabled=active();}catch(error){if(version!==previewVersion)return;$('preview').textContent=error.message;$('start').disabled=true;}},180);}
function table(id,headers,rows){const root=$(id);root.classList.remove('empty');if(!rows.length){root.textContent='まだ結果はありません。';root.classList.add('empty');return;}const t=document.createElement('table'),head=document.createElement('tr');headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.append(th);});const thead=document.createElement('thead');thead.append(head);t.append(thead);const body=document.createElement('tbody');rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(value=>{const td=document.createElement('td');if(value instanceof Node)td.append(value);else td.textContent=value;tr.append(td);});body.append(tr);});t.append(body);root.replaceChildren(t);}
function link(label,url){const a=document.createElement('a');a.textContent=label;a.href=url;a.target='_blank';a.rel='noopener';return a;}
function matchRow(r){
 const side=r.reference_side||'blue',opponent=side==='blue'?'red':'blue';
 const actions=document.createElement('div');actions.className='match-actions';
 if(r.replay_url)actions.append(link('リプレイ ↗',r.replay_url));
 else actions.append(document.createTextNode('リプレイ未保存'));
 const button=document.createElement('button');button.type='button';button.textContent='再対戦';
 button.disabled=active()||retrying||!r.retry_request;
 button.title=active()?'実行中の評価が完了するか停止した後に再対戦できます':'同じモデル・seed・陣営・機数・行動選択で1試合を開始';
 button.onclick=async()=>{
  if(active()||retrying)return;
  retrying=true;render();message('同じ条件で1試合を準備しています…');
  try{state=await api('/api/start',{...r.retry_request,pace:readPace()});message('再対戦を開始しました。前の結果は「過去の評価」に保存されています。');}
  catch(error){message(error.message);}
  finally{retrying=false;render();}
 };
 actions.append(button);
 return [r.match,r.model_names?.[side]||r.team_names[side],r.model_names?.[opponent]||r.team_names[opponent],r.seed,side.toUpperCase(),!r.winner?'引き分け':r.winner===side?'勝利':'敗北',r.ticks,actions];
}
const statusNames={running:'対戦中',paused:'一時停止',stopping:'停止処理中',stopped:'中断',completed:'完了',failed:'エラー'};
function render(){catalog();const job=state.job,current=state.current, busy=active();if(busy&&!$('pace').disabled)$('pace').value=String(job.pace);$('settings').disabled=busy;$('add').disabled=busy;$('start').disabled=busy||$('start').disabled;$('pause').disabled=!busy||job?.status==='stopping';$('stop').disabled=!busy||job?.status==='stopping';$('pause').textContent=job?.status==='paused'?'対戦を再開':'対戦を一時停止';$('job-status').textContent=statusNames[job?.status]||'待機中';$('progress').textContent=`${job?.completed||0} / ${job?.total||0}`;$('progress-bar').style.width=`${job?100*job.completed/job.total:0}%`;$('current-seed').textContent=current?.seed??'—';$('current-pair').textContent=current?`${current.team_names.blue} vs ${current.team_names.red}`:'モデルを選択して開始';if(job?.error)message(job.error);const downloads=$('downloads');downloads.replaceChildren();if(job){const base='/files/'+encodeURIComponent(job.id)+'/';downloads.append(link('実行設定 JSON',base+'manifest.json'));if(job.completed){for(const [label,file] of [['試合ログ JSONL','matches.jsonl'],['チーム CSV','team-standings.csv'],['モデル CSV','model-standings.csv']])downloads.append(link(label,base+file));}if(job.completed||!busy)downloads.append(link('結果 JSON',base+'summary.json'));}
 table('teams',['順位','チーム','試合','勝 / 分 / 敗','勝ち点','撃墜 / 損失','平均報酬'],state.teams.map(r=>[r.rank,r.team,r.played,`${r.wins} / ${r.draws} / ${r.losses}`,r.points,`${r.kills} / ${r.deaths}`,r.mean_return.toFixed(3)]));table('models',['チーム','モデル','参加試合','勝 / 分 / 敗','撃墜 / 損失','平均報酬'],state.models.map(r=>[r.team,r.model_name,r.played,`${r.wins} / ${r.draws} / ${r.losses}`,`${r.kills} / ${r.deaths}`,r.mean_return.toFixed(3)]));table('matches',['#','自機モデル','相手','seed','自機陣営','結果','tick','操作'],[...state.matches].reverse().map(matchRow));if(job&&!busy&&lastFinished!==job.id){lastFinished=job.id;history();preview();}}
async function history(){try{const rows=await api('/api/history');$('archive').replaceChildren();if(!rows.length){$('archive').textContent='過去の評価はありません。';return;}for(const row of rows){const p=document.createElement('p'),j=row.job;p.append(document.createTextNode(`${j.id} · ${statusNames[j.status]||j.status} · ${j.completed}/${j.total}試合　`),link('結果 JSON',`/files/${j.id}/summary.json`));if(j.completed)p.append(document.createTextNode('　'),link('試合ログ',`/files/${j.id}/matches.jsonl`));$('archive').append(p);}}catch(error){$('archive').textContent=error.message;}}
$('setup').addEventListener('input',event=>{if(['blue-count','red-count'].includes(event.target.id))lineups();if(event.target.id==='mode')showMode();saveSetup();preview();});
$('setup').addEventListener('change',saveSetup);
window.addEventListener('pagehide',saveSetup);
$('setup').onsubmit=async event=>{event.preventDefault();saveSetup();$('start').disabled=true;message('モデルを読み込み、対戦を準備しています…');try{state=await api('/api/start',request());message('');render();}catch(error){message(error.message);preview();}};
$('pause').onclick=async()=>{try{state=await api('/api/pause',{paused:state.job?.status!=='paused'});render();}catch(error){message(error.message);}};
$('stop').onclick=async()=>{try{state=await api('/api/stop',{});render();}catch(error){message(error.message);}};
$('model-kind').onchange=()=>{$('policy-id').disabled=$('model-kind').value==='model';};
$('add-model').onsubmit=async event=>{event.preventDefault();$('add').disabled=true;message('モデルを読み込み中…');try{await api('/api/models',{kind:$('model-kind').value,path:$('model-path').value,policy_id:$('policy-id').value,name:$('model-name').value});state=await api('/api/state');message('モデルを追加しました。');render();}catch(error){message(error.message);}finally{$('add').disabled=active();}};
$('history-refresh').onclick=history;
async function poll(){if(polling)return;polling=true;try{state=await api('/api/state');render();}catch(error){message('接続を確認してください：'+error.message);}finally{polling=false;}}
render();history();setInterval(poll,700);

// Pace remains editable while the rest of the match settings are locked.
$('settings').after(document.querySelector('label[for="pace"]'), $('pace'));
$('pace').onchange=async()=>{
  if(!active())return;
  $('pace').disabled=true;
  try{state=await api('/api/pace',{pace:readPace()});render();}
  catch(error){message(error.message);$('pace').value=String(state.job?.pace??state.defaults.pace);}
  finally{$('pace').disabled=false;}
};
