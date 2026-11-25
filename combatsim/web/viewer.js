const frames=bootstrap.frames, config=bootstrap.config, live=bootstrap.live;
if(live)document.body.classList.add("external-live");
const arenaX=config.arena_x, arenaY=config.arena_y, ceiling=config.ceiling;
const byId=id=>document.getElementById(id);
const slider=byId('slider'), play=byId('play'), scope=byId('scope');
const modeNames={search:'探索',follow:'追従',separate:'離脱',boundary:'境界復帰',support:'誘導支援',egress:'支援後離脱',evade:'回避',terminal_evade:'終末回避',withdraw:'弾切れ退避',patrol:'待機巡回',learned:'学習モデル'};
let liveStatus="idle", liveFrom=null, liveTo=null, liveShown=null, liveAt=0, liveFollow=true;
let livePace=null;
let playing=false,yaw=-.65,tilt=.62,zoom=1,panX=0,panY=0,last=0,drag=null,done=!!config.done;
byId('kind').textContent=live?'LIVE':'REPLAY';
slider.disabled=false;
function aircraftLabel(a){return a.label||a.id;}
function escapeLabel(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function refreshScope(frame){const selected=scope.value;scope.innerHTML='<option value="all">全機</option>';for(const a of frame.aircraft){const option=document.createElement('option');option.value=a.id;option.textContent=aircraftLabel(a);scope.appendChild(option);}scope.value=[...scope.options].some(o=>o.value===selected)?selected:'all';}
refreshScope(frames[0]);
slider.max=frames.length-1;slider.value=live?frames.length-1:0;

function color(a){
  const colors=a.team==='blue'?['#20ddff','#648cff','#55f4c9','#a9d4ff','#449ac1','#8587fa']:
    ['#ff4567','#ff9b53','#ff75bd','#ffb3b9','#e36640','#e27987'];
  const index=Number.isInteger(a.model_color)&&a.model_color>=0?a.model_color:0;
  return colors[index%colors.length];
}
function tintTowardWhite(hex,fraction){
  return '#'+[1,3,5].map(i=>{
    const value=parseInt(hex.slice(i,i+2),16);
    return Math.round(value+(255-value)*fraction).toString(16).padStart(2,'0');
  }).join('');
}
function direction(a){return [Math.cos(a.heading)*Math.cos(a.pitch),Math.sin(a.heading)*Math.cos(a.pitch),Math.sin(a.pitch)];}
function project(x,y,z,view){
  const w=view.width,h=view.height;x-=arenaX/2;y-=arenaY/2;z-=ceiling/2;
  if(view.id==='top'){const s=Math.min((w-70)/arenaX,(h-45)/arenaY);return [w/2+x*s,h/2-y*s,z];}
  // Side view deliberately uses independent axis scales: the arena is much
  // longer than it is high, so a physically equal scale makes altitude unreadable.
  if(view.id==='side'){const sx=(w-70)/arenaX,sz=(h-45)/ceiling;return [w/2+x*sx,h/2-z*sz,y];}
  // Exaggerate altitude only in the 3D display; all geometry shares this projection.
  const heightScale=Number(byId('height-scale').value);
  z*=heightScale;
  const width=Math.abs(Math.cos(yaw))*arenaX+Math.abs(Math.sin(yaw))*arenaY;
  const depth=Math.abs(Math.sin(yaw))*arenaX+Math.abs(Math.cos(yaw))*arenaY;
  const height=depth*Math.abs(Math.sin(tilt))+ceiling*heightScale*Math.abs(Math.cos(tilt));
  const s=Math.min((w-100)/width,(h-80)/height);
  const u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);
  return [w/2+panX+u*s*zoom,h/2+panY+(v*Math.sin(tilt)-z*Math.cos(tilt))*s*zoom,v*Math.cos(tilt)+z*Math.sin(tilt)];
}

// Display attitude only: the simulator stores flight direction, not body roll.
function targetAircraftBank(a,index){
  const c=a.control;
  let change=c?.heading_change, pitchChange=c?c.pitch_after-c.pitch_before:0;
  if(!Number.isFinite(change)){
    const previous=frames[Math.max(0,index-1)], current=frames[index];
    const before=previous?.aircraft.find(item=>item.id===a.id), after=current?.aircraft.find(item=>item.id===a.id);
    const ticks=current?.tick-previous?.tick;
    change=before&&after&&ticks>0?Math.atan2(Math.sin(after.heading-before.heading),Math.cos(after.heading-before.heading))/ticks:0;
    pitchChange=before&&after&&ticks>0?(after.pitch-before.pitch)/ticks:0;
  }
  const seconds=config.aero_tick_seconds||1;
  const speed=(a.speed??config.aircraft_speed??1)*(config.aero_length_scale||250)/seconds;
  const normal=Math.max(.1,(config.aero_gravity||9.80665)*Math.cos(a.pitch)+speed*pitchChange/seconds);
  // Keep estimated roll subtle; this is a visual cue, not measured attitude.
  const limit=Math.min(config.aero_max_bank||1.22,Math.PI*35/180);
  const target=Math.max(-limit,Math.min(limit,.5*Math.atan2(speed*Math.cos(a.pitch)*change/seconds,normal)));
  const deadzone=3*Math.PI/180;
  return Math.sign(target)*Math.max(0,Math.abs(target)-deadzone);
}
// Cache by recorded frame, never by paint time: seek, pause and playback speed
// must not change the displayed attitude. Live replacements invalidate the tail.
const bankFrames=[];
function recordedBanks(index){
  for(let i=bankFrames.length;i<=index;i++){
    const frame=frames[i],previous=bankFrames[i-1],banks=new Map();
    const dt=i?Math.max(0,frame.tick-frames[i-1].tick)*(config.aero_tick_seconds||1):0;
    for(const a of frame.aircraft){
      const before=previous?.get(a.id)??0,target=targetAircraftBank(a,i);
      const requested=(target-before)*(1-Math.exp(-dt/3.5));
      const step=6*Math.PI/180*dt;
      banks.set(a.id,before+Math.max(-step,Math.min(step,requested)));
    }
    bankFrames.push(banks);
  }
  return bankFrames[index];
}
function aircraftBank(a,index,tick=frames[index]?.tick){
  const current=recordedBanks(index).get(a.id)??0,next=frames[index+1];
  if(!next||tick<=frames[index].tick)return current;
  const fraction=Math.max(0,Math.min(1,(tick-frames[index].tick)/(next.tick-frames[index].tick)));
  return current+((recordedBanks(index+1).get(a.id)??current)-current)*fraction;
}
function aircraftBasis(a,bank){
  const forward=direction(a),left=[-Math.sin(a.heading),Math.cos(a.heading),0];
  const up=[-Math.sin(a.pitch)*Math.cos(a.heading),-Math.sin(a.pitch)*Math.sin(a.heading),Math.cos(a.pitch)];
  return [forward,left.map((v,i)=>v*Math.cos(bank)-up[i]*Math.sin(bank)),up.map((v,i)=>v*Math.cos(bank)+left[i]*Math.sin(bank))];
}
// Preserve the original aircraft symbol's exact outline, extruded into a solid.
// Local axes: nose forward, left wing positive Y, upper surface positive Z.
const aircraftOutline=[
 [15,0],[5.5,-2],[-2.5,-9.5],[-5.5,-9.5],[-3.8,-2.8],
 [-9.5,-4.7],[-8,-1.6],[-11,0],[-8,1.6],[-9.5,4.7],
 [-3.8,2.8],[-5.5,9.5],[-2.5,9.5],[5.5,2]
].map(([x,y])=>[x/15,-y/15]);
const aircraftVertices=[
 ...aircraftOutline.map(([x,y])=>[x,y,.045]),
 ...aircraftOutline.map(([x,y])=>[x,y,-.045]),
 // Raised faceted fuselage and canopy, within the original silhouette.
 [.86,0,.045],[.02,-.14,.045],[-.66,0,.045],[.02,.14,.045],[.02,0,.27],
 [.57,0,.12],[.34,-.075,.24],[.13,0,.30],[.34,.075,.24],
 [.86,0,-.045],[.02,-.14,-.045],[-.66,0,-.045],[.02,.14,-.045],[.02,0,-.17]
];
const outlineCount=aircraftOutline.length;
const aircraftFaces=[
 [Array.from({length:outlineCount},(_,i)=>i),'top'],
 [Array.from({length:outlineCount},(_,i)=>i+outlineCount),'bottom'],
 ...aircraftOutline.map((_,i)=>[[i,(i+1)%outlineCount,(i+1)%outlineCount+outlineCount,i+outlineCount],'bottom']),
 [[28,29,32],'top'],[[29,30,32],'wing'],[[30,31,32],'wing'],[[31,28,32],'top'],
 [[33,34,35],'glass'],[[33,35,36],'glass'],
 [[37,41,38],'bottom'],[[38,41,39],'bottom'],[[39,41,40],'bottom'],[[40,41,37],'bottom']
];
function drawAircraft3D(ctx,a,center,index,viewId='view',tick=frames[index]?.tick){
  const basis=aircraftBasis(a,aircraftBank(a,index,tick));
  const size=Number(byId('aircraft-size').value)*(viewId==='view'?zoom:.72);
  const world=aircraftVertices.map(v=>basis[0].map((_,i)=>v.reduce((sum,n,j)=>sum+n*basis[j][i],0)));
  // Uniform local scale: altitude exaggeration applies to positions, never the mesh.
  const projected=world.map(([x,y,z])=>{
    if(viewId==='top')return [center[0]+x*size,center[1]-y*size,z];
    if(viewId==='side')return [center[0]+x*size,center[1]-z*size,y];
    const u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);
    return [center[0]+u*size,center[1]+(v*Math.sin(tilt)-z*Math.cos(tilt))*size,v*Math.cos(tilt)+z*Math.sin(tilt)];
  });
  const tint=color(a),palette={top:tintTowardWhite(tint,.35),bottom:tintTowardWhite(tint,.24),wing:tint,glass:tintTowardWhite(tint,.72)};
  const faces=aircraftFaces.map(([ids,material])=>({ids,material,depth:ids.reduce((sum,i)=>sum+projected[i][2],0)/ids.length})).sort((a,b)=>a.depth-b.depth);
  ctx.lineJoin='round';ctx.lineWidth=.45;
  for(const {ids,material} of faces){
    const p=world[ids[0]],q=world[ids[1]],r=world[ids[2]],u=q.map((v,i)=>v-p[i]),v=r.map((n,i)=>n-p[i]);
    const normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
    const light=.80+.20*Math.abs((normal[0]*-.35+normal[1]*-.25+normal[2]*.9)/(Math.hypot(...normal)||1));
    const hex=palette[material];ctx.fillStyle='#'+[1,3,5].map(i=>Math.round(parseInt(hex.slice(i,i+2),16)*light).toString(16).padStart(2,'0')).join('');
    ctx.strokeStyle=tintTowardWhite(tint,.55)+'70';ctx.beginPath();ids.forEach((i,j)=>{const p=projected[i];if(j)ctx.lineTo(p[0],p[1]);else ctx.moveTo(p[0],p[1]);});ctx.closePath();ctx.fill();ctx.stroke();
  }
}

// Boundary of the actual spherical sector used by the simulation (range + cone).
function sectorPoint(a,radius,theta,phi){
  const f=direction(a),u=[-Math.sin(a.heading),Math.cos(a.heading),0];
  const v=[-Math.sin(a.pitch)*Math.cos(a.heading),-Math.sin(a.pitch)*Math.sin(a.heading),Math.cos(a.pitch)];
  return [a.x,a.y,a.z].map((value,i)=>value+radius*(Math.cos(theta)*f[i]+Math.sin(theta)*(Math.cos(phi)*u[i]+Math.sin(phi)*v[i])));
}

function drawView(view,f,index){
  while(index>0&&frames[index].tick>f.tick)index--;
  const ctx=view.getContext('2d'),point=a=>project(...a,view),w=view.width,h=view.height,is3D=view.id==='view';
  ctx.clearRect(0,0,w,h);
  const bg=ctx.createRadialGradient(w*.52,h*.48,0,w*.5,h*.5,w*.72);
  bg.addColorStop(0,is3D?'#0c1c2a':view.id==='top'?'#09262e':'#09212d');bg.addColorStop(.56,is3D?'#091522':'#06141e');bg.addColorStop(1,is3D?'#06101b':'#02070d');
  ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
  function line(a,b,c,width=1,dash=[]){const p=point(a),q=point(b);ctx.strokeStyle=c;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([]);}
  function polygon(points,c){ctx.fillStyle=c;ctx.beginPath();for(let i=0;i<points.length;i++){const p=point(points[i]);if(i)ctx.lineTo(p[0],p[1]);else ctx.moveTo(p[0],p[1]);}ctx.closePath();ctx.fill();}
  // The plane and every grid line follow the simulation's actual arena bounds.
  if(view.id!=='side'){
    polygon([[0,0,0],[arenaX,0,0],[arenaX,arenaY,0],[0,arenaY,0]],is3D?'#122d3b35':'#0d39423b');
    const divisions=is3D?10:20;
    for(let i=0;i<=divisions;i++){
      const major=i%5===0,tint=is3D?(major?'#577d9655':'#3d5b7138'):(major?'#42b7c244':'#32757f24'),x=arenaX*i/divisions,y=arenaY*i/divisions;
      line([x,0,0],[x,arenaY,0],tint,major?1.1:.65);line([0,y,0],[arenaX,y,0],tint,major?1.1:.65);
    }
    for(const [a,b] of [[[0,0,0],[arenaX,0,0]],[[arenaX,0,0],[arenaX,arenaY,0]],[[arenaX,arenaY,0],[0,arenaY,0]],[[0,arenaY,0],[0,0,0]]])line(a,b,'#5ac8cf70',1.1);
  }
  if(view.id==='side'){
    polygon([[0,0,0],[arenaX,0,0],[arenaX,0,ceiling],[0,0,ceiling]],'#10303a2b');
    for(let i=0;i<=4;i++)line([0,0,ceiling*i/4],[arenaX,0,ceiling*i/4],'#42a7b32b');
    for(let i=0;i<=10;i++)line([arenaX*i/10,0,0],[arenaX*i/10,0,ceiling],'#32757f24',.7);
    line([0,0,0],[arenaX,0,0],'#5ac8cf70',1.2);
  }
  if(view.id==='view'){
    for(const x of [0,arenaX])for(const y of [0,arenaY])line([x,y,0],[x,y,ceiling],'#7193ae99',1,[3,6]);
    for(const [a,b] of [[[0,0,ceiling],[arenaX,0,ceiling]],[[arenaX,0,ceiling],[arenaX,arenaY,ceiling]],[[arenaX,arenaY,ceiling],[0,arenaY,ceiling]],[[0,arenaY,ceiling],[0,0,ceiling]]])line(a,b,'#5c809b99',1);
  }

  function sector(a,radius,fov,tint){
    const count=32,rings=5,origin=[a.x,a.y,a.z];
    for(let j=1;j<=rings;j++)for(let k=0;k<count;k++){
      const theta=fov*.5*j/rings,previous=fov*.5*(j-1)/rings,phi=2*Math.PI*k/count,next=2*Math.PI*(k+1)/count;
      const p=sectorPoint(a,radius,theta,phi),q=sectorPoint(a,radius,theta,next);
      polygon([sectorPoint(a,radius,previous,phi),p,q,sectorPoint(a,radius,previous,next)],tint+(is3D?'02':'06'));
      if(j===rings){line(p,q,tint+'65',.8);polygon([origin,p,q],tint+(is3D?'01':'04'));}
    }
    const spokes=is3D?4:8;
    for(let k=0;k<spokes;k++)line(origin,sectorPoint(a,radius,fov/2,k*2*Math.PI/spokes),tint+(is3D?'20':'30'),.8,[3,6]);
    line(origin,sectorPoint(a,radius,0,0),tint+'55',.8,[2,6]);
  }
  if(byId('radar').checked)for(const a of f.aircraft)if(a.alive&&(scope.value==='all'||a.id===scope.value))sector(a,a.radar_range,a.radar_fov,color(a));
  if(byId('seeker').checked)for(const m of f.missiles)if(m.phase==='active')sector(m,m.seeker_range,m.seeker_fov,'#71ef9a');
  if(byId('trails').checked){
    const trailStart=Math.max(0,index-65),tracks=new Map();
    for(let j=trailStart;j<=index;j++)for(const a of (j===index?f:frames[j]).aircraft){
      if(!a.alive)continue;
      if(!tracks.has(a.id))tracks.set(a.id,{tint:color(a),points:[]});
      tracks.get(a.id).points.push({p:point([a.x,a.y,a.z]),frame:j});
    }
    ctx.lineCap='round';
    for(const track of tracks.values())for(let band=0;band<5;band++){
      ctx.beginPath();let segments=0;
      for(let j=1;j<track.points.length;j++){
        const a=track.points[j-1],b=track.points[j],age=index-b.frame;
        if(b.frame-a.frame!==1||Math.min(4,Math.floor(age/13))!==band)continue;
        ctx.moveTo(a.p[0],a.p[1]);ctx.lineTo(b.p[0],b.p[1]);segments++;
      }
      if(!segments)continue;
      ctx.strokeStyle=track.tint;ctx.globalAlpha=(1-band/5)*.12;ctx.lineWidth=5-band*.55;ctx.stroke();
      ctx.globalAlpha=(1-band/5)*.68;ctx.lineWidth=1.5-band*.16;ctx.stroke();
    }
    ctx.globalAlpha=1;ctx.lineCap='butt';
    // Missile streaks are made from recorded positions, including turns.
    const missileTracks=new Map(f.missiles.map(m=>[m.id,{tint:m.phase==='active'?'#71ef9a':'#ffc66d',points:[]}]));
    for(let j=Math.max(0,index-10);j<=index;j++)for(const m of (j===index?f:frames[j]).missiles){
      const track=missileTracks.get(m.id);if(track)track.points.push(point([m.x,m.y,m.z]));
    }
    for(const track of missileTracks.values()){
      if(track.points.length<2)continue;
      ctx.beginPath();ctx.moveTo(track.points[0][0],track.points[0][1]);
      for(let j=1;j<track.points.length-1;j++){
        const p=track.points[j],next=track.points[j+1];
        ctx.quadraticCurveTo(p[0],p[1],(p[0]+next[0])/2,(p[1]+next[1])/2);
      }
      const end=track.points.at(-1);ctx.lineTo(end[0],end[1]);
      ctx.strokeStyle=track.tint;ctx.globalAlpha=.11;ctx.lineWidth=4;ctx.stroke();ctx.globalAlpha=.48;ctx.lineWidth=.8;ctx.stroke();
    }
    ctx.globalAlpha=1;
  }
  const objects=[...f.aircraft.map(a=>({...a,kind:'aircraft'})),...f.missiles.map(m=>({...m,kind:'missile'}))].sort((a,b)=>point([a.x,a.y,a.z])[2]-point([b.x,b.y,b.z])[2]);
  for(const a of objects){
    const p=point([a.x,a.y,a.z]),d=direction(a),nose=point([a.x+d[0]*3,a.y+d[1]*3,a.z+d[2]*3]),angle=Math.atan2(nose[1]-p[1],nose[0]-p[0]);
    if(a.kind==='missile'){
      const active=a.phase==='active',coast=a.phase==='coast',tint=active?'#71ef9a':coast?'#8595a7':'#ffc66d';
      ctx.save();ctx.translate(p[0],p[1]);ctx.rotate(angle);
      if(!coast){
        const flame=ctx.createLinearGradient(-19,0,-5,0);flame.addColorStop(0,tint+'00');flame.addColorStop(.6,tint+'80');flame.addColorStop(1,'#ffe6a7');
        ctx.fillStyle=flame;ctx.beginPath();ctx.moveTo(-4,-1.7);ctx.lineTo(-20,0);ctx.lineTo(-4,1.7);ctx.closePath();ctx.fill();
      }
      ctx.shadowColor=tint;ctx.shadowBlur=active?11:6;ctx.fillStyle=tint;ctx.strokeStyle='#031018';ctx.lineWidth=.9;
      ctx.beginPath();ctx.moveTo(7,0);ctx.lineTo(1,-1.7);ctx.lineTo(-5,-1.2);ctx.lineTo(-7,-3.3);ctx.lineTo(-5.8,0);ctx.lineTo(-7,3.3);ctx.lineTo(-5,1.2);ctx.lineTo(1,1.7);ctx.closePath();ctx.fill();ctx.stroke();
      ctx.shadowBlur=0;ctx.strokeStyle=active?'#b2ffd1':'#ffdf9b';ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(-3,0);ctx.lineTo(5,0);ctx.stroke();ctx.restore();continue;
    }
    const tint=color(a);ctx.globalAlpha=a.alive?1:.22;
    if(view.id==='view'&&a.alive){
      const ground=point([a.x,a.y,0]);ctx.fillStyle=tint+'24';ctx.beginPath();ctx.ellipse(ground[0],ground[1],5,2.4,0,0,Math.PI*2);ctx.fill();
      line([a.x,a.y,0],[a.x,a.y,a.z],tint+'a0',1.15,[3,4]);
    }
    drawAircraft3D(ctx,a,p,index,view.id,f.tick);
    const label=aircraftLabel(a);ctx.font=is3D?'600 12px monospace':'600 10px monospace';
    const labelWidth=ctx.measureText(label).width,lx=Math.min(w-labelWidth-10,Math.max(10,p[0]+17)),ly=Math.max(38,p[1]-15),labelHeight=is3D?17:14;
    ctx.fillStyle='#06101bed';ctx.fillRect(lx-4,ly-labelHeight+3,labelWidth+8,labelHeight+3);ctx.fillStyle=tint;ctx.fillText(label,lx,ly);ctx.globalAlpha=1;
  }
  ctx.fillStyle='#678591';ctx.font='9px monospace';ctx.fillText(view.id==='top'?`X 0–${arenaX} / Y 0–${arenaY}`:view.id==='side'?`X 0–${arenaX} / Z 0–${ceiling}（高度強調）`:`X 0–${arenaX} / Y 0–${arenaY} / Z 0–${ceiling}`,14,21);
  ctx.save();ctx.fillStyle='#7eabb5';ctx.font='9px monospace';ctx.fillText(`${view.id==='view'?'3D戦域':view.id==='top'?'上面図':'側面図'}  /  ステップ ${String(Math.floor(f.tick)).padStart(4,'0')}`,14,h-14);ctx.textAlign='right';ctx.fillText('飛行中 '+f.aircraft.filter(a=>a.alive).length+' 機',w-14,h-14);ctx.restore();
}

function status(message){byId('status').textContent=message;}
function lerp(a,b,t){return a+(b-a)*t;}
function lerpAngle(a,b,t){return a+Math.atan2(Math.sin(b-a),Math.cos(b-a))*t;}
function interpolateFrame(from,to,t){
  if(t>=1)return to;
  if(t<=0)return from;
  const blend=(items,next)=>{
    const targets=new Map(next.map(item=>[item.id,item]));
    return items.map(item=>{
      const target=targets.get(item.id);
      if(!target)return item;
      return {...item,x:lerp(item.x,target.x,t),y:lerp(item.y,target.y,t),
        z:lerp(item.z,target.z,t),heading:lerpAngle(item.heading,target.heading,t),
        pitch:lerpAngle(item.pitch,target.pitch,t),
        ...(item.control&&target.control?{control:{...item.control,
          heading_change:lerp(item.control.heading_change,target.control.heading_change,t),
          pitch_before:lerp(item.control.pitch_before,target.control.pitch_before,t),
          pitch_after:lerp(item.control.pitch_after,target.control.pitch_after,t)}}:{})};
    });
  };
  return {...from,tick:lerp(from.tick,to.tick,t),aircraft:blend(from.aircraft,to.aircraft),missiles:blend(from.missiles,to.missiles)};
}
function sampleRecordedFrame(tick){
  let lo=0,hi=frames.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].tick<=tick)lo=mid;else hi=mid-1;}
  const from=frames[lo],to=frames[lo+1];
  return {index:lo,frame:to?interpolateFrame(from,to,(tick-from.tick)/(to.tick-from.tick)):from};
}
function frameDuration(index){
  const seconds=(frames[index+1]?.tick??frames[index].tick+1)-frames[index].tick;
  const speed=byId('speed').value;
  return 1000*Math.max(0.001,seconds)*(speed==='realtime'?(config.aero_tick_seconds||1):1/Number(speed));
}
function drawMotion(now){
  const index=+slider.value,from=frames[index],to=frames[index+1];
  if(!from||!to)return;
  const fraction=Math.max(0,Math.min(1,(now-last)/frameDuration(index)));
  const frame=interpolateFrame(from,to,fraction);
  for(const id of ['view','top','side'])drawView(byId(id),frame,index);
}
function timelineStatus(){
  if(live)status(!liveFollow?'履歴を表示中。対戦は進行します。「最新へ」でライブ表示に戻ります。':liveStatus==='paused'?'対戦を一時停止中':done?'対戦終了：結果表からリプレイを開けます。':'巻き戻しは表示のみ。対戦の一時停止は設定パネルから操作します。');
  else status(playing?'リプレイ再生中':'一時停止');
}
function seek(index){
  playing=false;
  slider.value=Math.max(0,Math.min(index,frames.length-1));
  if(live){liveFollow=+slider.value===frames.length-1;liveShown=liveFollow?frames.at(-1):null;}
  timelineStatus();draw();
}
function rewardRows(rewards){
  const rows=new Map();
  const get=(reason,id)=>{
    const key=JSON.stringify([reason,id]);
    if(!rows.has(key))rows.set(key,{reason,id,value:0,discarded:0});
    return rows.get(key);
  };
  for(const [name,agents] of Object.entries(rewards.components)){
    const reason=name.replace(/^HookReward:/,'');
    for(const [id,value] of Object.entries(agents))if(value)get(reason,id).value+=value;
  }
  for(const [reason,agents] of Object.entries(rewards.clipping)){
    for(const [id,entry] of Object.entries(agents))get(reason,id).discarded+=entry.discarded_abs;
  }
  return [...rows.values()];
}
function drawRewards(frame){
  const rewards=frame.rewards,rows=rewards?rewardRows(rewards):[];
  const name=id=>aircraftLabel(frame.aircraft.find(a=>a.id===id)??{id});
  const amount=value=>value===0?'0':(value>0?'+':'')+Number(value.toPrecision(5));
  byId('reward-status').textContent=rewards?`tick ${frame.tick}：${rows.length?'適用後の報酬（理由別の合計）':'報酬の変動なし'}`:'このリプレイには報酬記録がありません。';
  byId('reward-rows').innerHTML=rows.map(({reason,id,value,discarded})=>
    `<tr><td>${escapeLabel(name(id))}</td><td>${escapeLabel(reason)}</td><td>${amount(value)}</td><td>${discarded?amount(discarded):'—'}</td></tr>`).join('');
  byId('reward-totals').textContent=rewards?'このtickまでの累計：'+
    Object.entries(rewards.episode_totals).map(([id,value])=>name(id)+' '+amount(value)).join(' / '):'';
}
function draw(){
  const index=+slider.value,f=frames[index];if(!f)return;
  for(const id of ['view','top','side'])drawView(byId(id),live&&liveFollow&&liveShown?liveShown:f,index);
  const history=live?!liveFollow:!playing&&index<frames.length-1;
  byId('kind').textContent=live?(history?'HISTORY':'LIVE'):'REPLAY';
  document.body.dataset.playing=String(playing);
  document.body.dataset.livePlaying=String(live&&playing);
  for(const team of ['blue','red']){
    const aircraft=f.aircraft.filter(a=>a.team===team),alive=aircraft.filter(a=>a.alive);
    byId(team+'-alive').textContent=`${alive.length} / ${aircraft.length}`;
    byId(team+'-ammo').textContent=alive.reduce((sum,a)=>sum+a.ammo,0);
    byId(team+'-strength').style.width=`${aircraft.length?alive.length/aircraft.length*100:0}%`;
  }
  byId('missile-count').textContent=f.missiles.length;
  byId('active-count').textContent=f.missiles.filter(m=>m.phase==='active').length;
  byId('mission-tick').textContent=String(Math.floor(f.tick)).padStart(4,'0');
  byId('mission-state').textContent=live?(liveStatus==='paused'?'一時停止':done?'対戦終了':liveStatus==='running'?'交戦中':'待機中'):(playing?'リプレイ再生':'リプレイ');
  byId('time').textContent=live&&playing?`tick ${f.tick}`:`tick ${f.tick} / ${frames.at(-1).tick}`;
  byId('aircraft').innerHTML=f.aircraft.map(a=>`<tr><td class="${a.team}">${escapeLabel(aircraftLabel(a))}</td><td>${a.alive?'飛行中':'撃墜'}</td><td>${a.z.toFixed(1)}</td><td>${(a.pitch*180/Math.PI).toFixed(1)}°</td><td>${a.ammo}</td><td>${modeNames[a.mode]||'—'}</td></tr>`).join('');
  drawRewards(f);
  byId('events').textContent=JSON.stringify(f.events,null,2);
  play.textContent=live?(playing?'表示を一時停止':'表示を再生'):(playing?'一時停止':'再生');
  play.disabled=frames.length<2;byId('step').disabled=index===frames.length-1;
  byId('back').disabled=byId('back-ten').disabled=index===0;
  byId('latest').disabled=!history;
}
function showModels(models){
  let label=byId('model-info');
  if(!label){label=document.createElement('p');label.id='model-info';byId('status').after(label);}
  label.textContent=Object.entries(models).map(([side,m])=>`${side}: ${m.sha256 ? `model ${m.epoch??'?'} (${m.sha256.slice(0,10)})` : m.policy}`).join(' / ');
}
function updateState(state,reset=false,follow=true){
  if(reset){frames.length=0;bankFrames.length=0;}
  for(const frame of state.new_frames||[state.frame]){
    if(frames.at(-1)?.tick===frame.tick){bankFrames.length=Math.min(bankFrames.length,frames.length-1);frames[frames.length-1]=frame;}else frames.push(frame);
  }
  done=state.done;slider.max=frames.length-1;
  if(reset||follow)slider.value=frames.length-1;
  if(reset)refreshScope(state.frame);
  if(state.models)showModels(state.models);
  if(done&&+slider.value===frames.length-1)playing=false;
  timelineStatus();
}
function advance(now=performance.now()){
  if(+slider.value<frames.length-1){
    slider.value=+slider.value+1;
    last=now;
    if(+slider.value===frames.length-1&&(!live||done))playing=false;
    timelineStatus();draw();
  }else if(!live||done){playing=false;draw();}
}
play.onclick=()=>{if((!live||done)&&+slider.value===frames.length-1)slider.value=0;playing=!playing;last=performance.now();timelineStatus();draw();};
byId('back').onclick=()=>seek(+slider.value-1);
byId('back-ten').onclick=()=>seek(+slider.value-10);
byId('latest').onclick=()=>seek(frames.length-1);
byId('step').onclick=()=>{if(live)seek(+slider.value+1);else{playing=false;advance();}};
slider.oninput=()=>seek(+slider.value);for(const id of ['radar','seeker','trails','scope','height-scale','aircraft-size'])byId(id).onchange=draw;
function resetCamera(){yaw=-.65;tilt=.62;zoom=1;panX=0;panY=0;draw();}
byId('camera-reset').onclick=resetCamera;
const canvas=byId('view');
function canvasPosition(e){
  const rect=canvas.getBoundingClientRect();
  // object-fit: contain can letterbox the canvas inside its stage.
  const scale=Math.min(rect.width/canvas.width,rect.height/canvas.height);
  return [(e.clientX-rect.left-(rect.width-canvas.width*scale)/2)/scale,
    (e.clientY-rect.top-(rect.height-canvas.height*scale)/2)/scale];
}
function zoomCamera(factor,x,y){
  const next=Math.max(.5,Math.min(8,zoom*factor)),ratio=next/zoom;
  panX=x-canvas.width/2-(x-canvas.width/2-panX)*ratio;
  panY=y-canvas.height/2-(y-canvas.height/2-panY)*ratio;
  zoom=next;draw();
}
canvas.addEventListener('wheel',e=>{
  e.preventDefault();
  const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?canvas.height:1);
  zoomCamera(Math.exp(Math.max(-1,Math.min(1,-delta*.0015))),...canvasPosition(e));
},{passive:false});
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onpointerdown=e=>{
  if(e.button!==0&&e.button!==2)return;
  e.preventDefault();drag={point:canvasPosition(e),pan:e.button===2,id:e.pointerId};
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointermove=e=>{
  if(!drag||drag.id!==e.pointerId)return;
  const point=canvasPosition(e),dx=point[0]-drag.point[0],dy=point[1]-drag.point[1];
  if(drag.pan){panX+=dx;panY+=dy;}
  else{yaw+=dx*.008;tilt=Math.max(.08,Math.min(1.5,tilt+dy*.008));}
  drag.point=point;draw();
};
canvas.onpointerup=canvas.onpointercancel=canvas.onlostpointercapture=()=>drag=null;
if(config.models)showModels(config.models);
function animate(now){
  if(live&&liveFollow&&liveFrom&&liveTo&&liveStatus==='running'){
    const fraction=Math.max(0,Math.min(1,(now-liveAt)/(livePace==='realtime'?Math.max(1,(liveTo.tick-liveFrom.tick)*(config.aero_tick_seconds||1)*1000):250)));
    const sampled=sampleRecordedFrame(lerp(liveFrom.tick,liveTo.tick,fraction));
    liveShown=sampled.frame;
    for(const id of ['view','top','side'])drawView(byId(id),liveShown,sampled.index);
  }
  if(playing){
    if(now-last>=frameDuration(+slider.value))advance(now);
    else drawMotion(now);
  }
  requestAnimationFrame(animate);
}
timelineStatus();draw();requestAnimationFrame(animate);

if(live){
 let watchId='',watchBusy=false;
 async function watch(){if(watchBusy)return;watchBusy=true;try{
  const response=await fetch('/api/watch?match='+encodeURIComponent(watchId)+'&after='+frames.at(-1).tick);
  if(!response.ok)throw Error(response.statusText);
  const state=await response.json(),reset=state.match_id!==watchId;
  liveStatus=state.status||'idle';
  livePace=state.pace;
  const target=state.frame;
  if(reset){liveFollow=true;}
  if(reset||liveStatus!=='running'){liveFrom=liveTo=liveShown=target;}
  else if(!liveTo||target.tick!==liveTo.tick){liveFrom=liveShown||target;liveTo=target;liveAt=performance.now();}
  watchId=state.match_id;updateState(state,reset,liveFollow);
  playing=false; // Live always displays the latest server tick; no separate playback clock.
  timelineStatus();draw();
 }catch(error){status('観戦の接続エラー：'+error.message);}finally{watchBusy=false;}}
 watch();setInterval(watch,250);
}
