'use strict';
// Shared recorded motion, flight attitude and overview framing.
const ArenaScene = (() => {
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const mix = (a, b, t) => a + (b - a) * t;
  const angleMix = (a, b, t) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;
  const teamColor = team => team === 'blue' ? '#59d9ef' : '#ff796b';
  const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
  function sample(frames, tick) {
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (frames[mid].tick <= tick) lo = mid; else hi = mid - 1; }
    const a = frames[lo], b = frames[lo + 1];
    if (!b || tick <= a.tick) return {frame:a, index:lo};
    const t = clamp((tick - a.tick) / (b.tick - a.tick), 0, 1);
    const blend = (items, next) => {
      const targets = new Map(next.map(item => [item.id, item]));
      return items.map(item => {
        const target = targets.get(item.id);
        if (!target) return item;
        return {...item, x:mix(item.x, target.x, t), y:mix(item.y, target.y, t), z:mix(item.z, target.z, t),
          heading:angleMix(item.heading, target.heading, t), pitch:angleMix(item.pitch, target.pitch, t)};
      });
    };
    return {frame:{...a, tick, aircraft:blend(a.aircraft, b.aircraft), missiles:blend(a.missiles, b.missiles)}, index:lo};
  }
  function position(entity, config, height) {
    const scale = 1000 / config.arena_x;
    return [(entity.x - config.arena_x / 2) * scale, (entity.y - config.arena_y / 2) * scale, entity.z * scale * height];
  }
  function bvrTrack(observer,target,source,config){
    const km=(config.aero_length_scale||250)/1000,cos=Math.cos(observer.heading),sin=Math.sin(observer.heading);
    const dx=target.x-observer.x,dy=target.y-observer.y,dz=target.z-observer.z;
    const forward=(dx*cos+dy*sin)*km,right=(dx*sin-dy*cos)*km,heading=observer.heading-target.heading;
    return {target,source,forward,right,bearing:Math.atan2(right,forward),
      horizontal:Math.hypot(dx,dy)*km,range:Math.hypot(dx,dy,dz)*km,altitude:dz*km*1000,
      heading:Math.atan2(Math.sin(heading),Math.cos(heading))};
  }
  function bvrContacts(observer,aircraft,config){
    if(!observer.alive)return [];
    return (observer.hud_contacts||[]).flatMap(contact=>{
      const target=aircraft.find(a=>a.id===contact.id);
      if(!target?.alive||target.id===observer.id||!['ally','radar','datalink'].includes(contact.source))return [];
      const track=bvrTrack(observer,target,contact.source,config);
      const rmax=contact.source!=='ally'&&Number.isFinite(contact.rmax_km)&&contact.rmax_km>=0?contact.rmax_km:null;
      return [{...track,rmax,rmaxHorizontal:rmax===null?null:Math.sqrt(Math.max(0,rmax*rmax-(track.altitude/1000)**2))}];
    });
  }
  function bvrMissiles(observer,missiles,config){
    if(!observer.alive)return [];
    const known=observer.hud_missiles??missiles.filter(m=>m.team===observer.team).map(m=>({id:m.id,source:m.owner===observer.id?'own':'ally'}));
    return known.flatMap(contact=>{
      const target=missiles.find(m=>m.id===contact.id);
      if(!target||target.alive===false||!['own','ally','radar','datalink'].includes(contact.source))return [];
      if((contact.source==='own'&&(target.team!==observer.team||target.owner!==observer.id))||
         (contact.source==='ally'&&(target.team!==observer.team||target.owner===observer.id))||
         (['radar','datalink'].includes(contact.source)&&target.team===observer.team))return [];
      return [bvrTrack(observer,target,contact.source,config)];
    });
  }
  function fitDistance(x,y,z,aspect,yaw,elevation) {
    const horizontal=Math.abs(Math.cos(yaw))*x+Math.abs(Math.sin(yaw))*y;
    const ground=Math.abs(Math.sin(yaw))*x+Math.abs(Math.cos(yaw))*y;
    const vertical=Math.sin(elevation)*ground+Math.cos(elevation)*z;
    const depth=Math.cos(elevation)*ground+Math.sin(elevation)*z;
    return (Math.max(horizontal/Math.max(.1,aspect),vertical)/Math.tan(21*Math.PI/180)+depth)*1.08;
  }
  function battleView(frame,effects,config,height,aspect,yaw,elevation) {
    const recent=new Set(effects.filter(e=>frame.tick>=e.tick&&frame.tick-e.tick<120).map(e=>e.position.id));
    let aircraft=frame.aircraft.filter(a=>a.alive||recent.has(a.id));
    if(!aircraft.length)aircraft=frame.aircraft;
    if(!aircraft.length)return {target:[0,0,30],distance:900};
    const points=aircraft.map(a=>position(a,config,height)),lo=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))),hi=[0,1,2].map(i=>Math.max(...points.map(p=>p[i])));
    const half=lo.map((v,i)=>Math.max([85,55,12][i],(hi[i]-v)*.65+12));
    return {target:lo.map((v,i)=>(v+hi[i])/2),distance:fitDistance(...half,aspect,yaw,elevation),half};
  }
  function updateLossCues(cues,frame,effects,previous,time) {
    if(!previous||frame.tick<previous.tick)cues.clear();
    for(const a of frame.aircraft){
      if(a.alive){cues.delete(a.id);continue;}
      if(cues.has(a.id))continue;
      const event=effects.find(e=>e.position.id===a.id&&e.tick<=frame.tick)||{tick:frame.tick,position:a,type:'crash'};
      const witnessed=previous?.aircraft.some(p=>p.id===a.id&&p.alive);
      cues.set(a.id,{event,started:time-(witnessed?0:12)});
    }
    return cues;
  }
  function wreckState(cue,time) {
    const age=Math.max(0,time-cue.started);
    return {age,pose:cue.event.position,flash:Math.max(0,1-age/.45),fade:clamp((age-.3)/1.2,0,1),visible:age<1.5};
  }
  const aircraftLabel = a => a.label || a.id;
  function aircraftColor(a){
    const colors=a.team==='blue'?['#20ddff','#648cff','#55f4c9','#a9d4ff','#449ac1','#8587fa']:
      ['#ff4567','#ff9b53','#ff75bd','#ffb3b9','#e36640','#e27987'];
    const index=Number.isInteger(a.model_color)&&a.model_color>=0?a.model_color:0;
    return colors[index%colors.length];
  }
  function targetAircraftBank(a,index,frames,config){
    const c=a.control;
    let change=c?.heading_change, pitchChange=c&&Number.isFinite(c.pitch_after-c.pitch_before)?c.pitch_after-c.pitch_before:0;
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
  function createBankSampler(frames,config){
    const bankFrames=[];
    // Cache by recorded frame, never by paint time: seek, pause and playback speed
    // must not change the displayed attitude. Start a fresh sampler for each match.
    function recordedBanks(index){
      for(let i=bankFrames.length;i<=index;i++){
        const frame=frames[i],previous=bankFrames[i-1],banks=new Map();
        const dt=i?Math.max(0,frame.tick-frames[i-1].tick)*(config.aero_tick_seconds||1):0;
        for(const a of frame.aircraft){
          const before=previous?.get(a.id)??0,target=targetAircraftBank(a,i,frames,config);
          const requested=(target-before)*(1-Math.exp(-dt/3.5));
          const step=6*Math.PI/180*dt;
          banks.set(a.id,before+Math.max(-step,Math.min(step,requested)));
        }
        bankFrames.push(banks);
      }
      return bankFrames[index];
    }
    return function aircraftBank(a,index,tick=frames[index]?.tick){
      const current=recordedBanks(index).get(a.id)??0,next=frames[index+1];
      if(!next||tick<=frames[index].tick)return current;
      const fraction=Math.max(0,Math.min(1,(tick-frames[index].tick)/(next.tick-frames[index].tick)));
      return current+((recordedBanks(index+1).get(a.id)??current)-current)*fraction;
    }
  }
  function aircraftBasis(a,bank=0) {
    const h=a.heading,p=a.pitch;
    const f=[Math.cos(h)*Math.cos(p),Math.sin(h)*Math.cos(p),Math.sin(p)],l=[-Math.sin(h),Math.cos(h),0];
    const u=[-Math.cos(h)*Math.sin(p),-Math.sin(h)*Math.sin(p),Math.cos(p)];
    return [f,l.map((n,i)=>n*Math.cos(bank)-u[i]*Math.sin(bank)),u.map((n,i)=>n*Math.cos(bank)+l[i]*Math.sin(bank))];
  }
  function flightTelemetry(a,config){
    const scale=config.aero_length_scale||250,seconds=config.aero_tick_seconds||1;
    return {altitude:Math.round(a.z*scale),speed:Math.round((a.speed||0)*scale/seconds*3.6),
      heading:((Math.round(90-a.heading*180/Math.PI)%360)+360)%360};
  }
  function flightHistory(frames,id,tick,config){
    const to=clamp(tick,frames[0].tick,frames.at(-1).tick),from=Math.max(frames[0].tick,to-60/(config.aero_tick_seconds||1));
    const start=sample(frames,from),end=sample(frames,to),points=[];
    function append(frame){const a=frame.aircraft.find(a=>a.id===id);if(a)points.push({tick:frame.tick,...flightTelemetry(a,config)});}
    append(start.frame);
    for(let i=start.index+1;i<=end.index;i++)append(frames[i]);
    if(end.frame.tick>start.frame.tick&&points.at(-1)?.tick!==end.frame.tick)append(end.frame);
    return {from,to,points};
  }
  return {clamp,mix,angleMix,teamColor,clock,sample,position,bvrContacts,bvrMissiles,fitDistance,battleView,updateLossCues,wreckState,aircraftBasis,aircraftColor,aircraftLabel,createBankSampler,flightTelemetry,flightHistory};
})();
