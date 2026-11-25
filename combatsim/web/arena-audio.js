const clips={music:'score',engine:'engine',launch:'launch',hit:'impact'};
export function crossedAudioEvents(frames,from,to){
  if(from===null||to<=from)return [];
  const events=[];
  for(let i=frames.length-1;i>=0&&frames[i].tick>from;i--){
    if(frames[i].tick>to)continue;
    for(const event of frames[i].events||[])if(clips[event.type]||event.type==='crash')events.push({event,frame:frames[i]});
    if(events.length>=6)break;
  }
  return events.sort((a,b)=>Number(['hit','crash'].includes(b.event.type))-Number(['hit','crash'].includes(a.event.type))).slice(0,6);
}
export function createArenaAudio(){
  let context,master,musicGain,effectsGain,engineGain,loading,enabled=false,active=false;
  let musicVolume=.45,effectsVolume=.7;
  const buffers=new Map(),loops=[],voices=new Set();
  function ramp(gain,value){gain.gain.setTargetAtTime(value,context.currentTime,.08);}
  function update(){
    if(!context)return;
    ramp(master,enabled&&active?1:0);ramp(musicGain,musicVolume);ramp(effectsGain,effectsVolume);
    if(!enabled||!active)for(const voice of voices)voice.stop();
  }
  function source(key,destination,loop=false){
    const node=context.createBufferSource();node.buffer=buffers.get(key);node.loop=loop;node.connect(destination);node.start();return node;
  }
  async function enable(value){
    if(!value){enabled=false;update();return;}
    if(!context){
      context=new AudioContext();master=context.createGain();master.gain.value=0;
      const limiter=context.createDynamicsCompressor();limiter.threshold.value=-10;limiter.knee.value=12;limiter.ratio.value=8;limiter.attack.value=.003;limiter.release.value=.25;
      master.connect(limiter);limiter.connect(context.destination);
      musicGain=context.createGain();effectsGain=context.createGain();engineGain=context.createGain();engineGain.gain.value=.16;
      musicGain.connect(master);effectsGain.connect(master);engineGain.connect(effectsGain);
    }
    await context.resume();
    if(!loading)loading=Promise.all([...new Set(Object.values(clips))].map(async key=>{
      const response=await fetch(`/arena-assets/arena-${key}.ogg`);if(!response.ok)throw Error(`Audio HTTP ${response.status}`);
      buffers.set(key,await context.decodeAudioData(await response.arrayBuffer()));
    })).catch(error=>{loading=null;throw error;});
    await loading;
    if(!loops.length)loops.push(source('score',musicGain,true),source('engine',engineGain,true));
    enabled=true;update();
  }
  function play(type,pan=0){
    if(!enabled||!active||context?.state!=='running'||voices.size>=8)return;
    const key=type==='crash'?'impact':clips[type];if(!buffers.has(key))return;
    const panner=context.createStereoPanner();panner.pan.value=Math.max(-.75,Math.min(.75,pan));panner.connect(effectsGain);
    const voice=source(key,panner);voices.add(voice);voice.onended=()=>{voices.delete(voice);voice.disconnect();panner.disconnect();};
    if(key==='impact'){musicGain.gain.cancelScheduledValues(context.currentTime);musicGain.gain.setTargetAtTime(musicVolume*.4,context.currentTime,.025);musicGain.gain.setTargetAtTime(musicVolume,context.currentTime+.8,.4);}
  }
  return {enable,play,
    setActive(value){if(active!==value){active=value;update();}},
    volume(music,effects){musicVolume=music;effectsVolume=effects;update();},
    dispose(){for(const node of [...loops,...voices])node.stop();context?.close();}
  };
}
