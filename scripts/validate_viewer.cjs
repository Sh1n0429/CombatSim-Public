// Run with node scripts/validate_viewer.cjs; no dependencies required.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').join(__dirname,'../combatsim/web/viewer.js'),'utf8');
const missile=(x,y,heading=0)=>({id:'m',x,y,z:1,heading,pitch:0});
const frames=[{tick:0,aircraft:[],missiles:[]},
 {tick:1,aircraft:[],missiles:[missile(0,0,3.1)]},
 {tick:2,aircraft:[],missiles:[missile(10,0,-3.1)]},
 {tick:3,aircraft:[],missiles:[missile(10,10)]},
 {tick:4,aircraft:[],missiles:[]}];
const before=JSON.stringify(frames),context=vm.createContext({frames});
vm.runInContext(source.slice(source.indexOf('function lerp('),source.indexOf('function frameDuration(')),context);
const sample=t=>context.sampleRecordedFrame(t).frame;
assert.equal(sample(.5).missiles.length,0);
assert.equal(sample(1).missiles.length,1);
assert.equal(sample(1.5).missiles[0].x,5);
assert(Math.abs(sample(1.5).missiles[0].heading-Math.PI)<1e-10);
assert.equal(sample(2.5).missiles[0].x,10);
assert.equal(sample(2.5).missiles[0].y,5);
assert.equal(sample(4).missiles.length,0);
assert.equal(sample(2.5).tick,2.5);
assert.equal(JSON.stringify(frames),before);
console.log('PASS: recorded turns, spawn/removal, wrapped angles, timeline and immutable frames');

// Verify flight attitude, bank sign, geometry and repeatable seek rendering.
Object.assign(context,{config:{},direction:a=>[Math.cos(a.heading)*Math.cos(a.pitch),Math.sin(a.heading)*Math.cos(a.pitch),Math.sin(a.pitch)],byId:()=>({value:'29'}),yaw:-.65,tilt:.62,zoom:1});
const colorCode=source.slice(source.indexOf('function color('),source.indexOf('function direction('));
vm.runInContext(colorCode,context);
assert.equal(context.color({team:'blue'}),context.color({team:'blue',model_color:0}));
assert.notEqual(context.color({team:'blue',model_color:0}),context.color({team:'blue',model_color:1}));
assert.notEqual(context.color({team:'red',model_color:0}),context.color({team:'red',model_color:1}));
vm.runInContext(source.slice(source.indexOf('function targetAircraftBank('),source.indexOf('// Boundary of the actual spherical sector')),context);
const aircraft={id:'a',heading:0,pitch:0,speed:1,team:'blue',control:{heading_change:.04,pitch_before:0,pitch_after:0}};
const bank=context.targetAircraftBank(aircraft,0);
assert(bank>0&&bank<=1.22);
assert(context.targetAircraftBank({...aircraft,control:{...aircraft.control,heading_change:-.04}},0)<0);
assert.equal(context.targetAircraftBank({...aircraft,control:{...aircraft.control,heading_change:0}},0),0);
const basis=context.aircraftBasis(aircraft,bank);
assert(basis[1][2]<0,'positive heading turn lowers the left wing');
for(const axis of basis)assert(Math.abs(Math.hypot(...axis)-1)<1e-12);
for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)assert(Math.abs(basis[i].reduce((n,x,k)=>n+x*basis[j][k],0))<1e-12);
assert(context.aircraftBasis({...aircraft,pitch:.3},0)[0][2]>0);
const drawn=[];const ctx={beginPath(){},moveTo(x,y){assert(Number.isFinite(x)&&Number.isFinite(y));drawn.push([x,y]);},lineTo(x,y){assert(Number.isFinite(x)&&Number.isFinite(y));drawn.push([x,y]);},closePath(){},fill(){},stroke(){}};
context.drawAircraft3D(ctx,aircraft,[100,100,0],0);
const first=JSON.stringify(drawn);drawn.length=0;
context.drawAircraft3D(ctx,aircraft,[100,100,0],0);
assert.equal(JSON.stringify(drawn),first);
assert(drawn.length>40);
assert.equal(JSON.stringify(frames),before);
console.log('PASS: bank direction and limits, orthonormal attitude, solid mesh, deterministic rendering');

assert(Math.abs(context.targetAircraftBank({...aircraft,control:{...aircraft.control,heading_change:10}},0))<=Math.PI*35/180);
for(const view of ['top','side']){
 drawn.length=0;context.drawAircraft3D(ctx,aircraft,[100,100,0],0,view);
 assert(drawn.length>40,view+' uses solid mesh');
 assert.notEqual(JSON.stringify(drawn),first,view+' uses its own projection');
}
console.log('PASS: restrained roll and solid meshes in top/side views');

const motionFrames=[];
for(let i=0;i<24;i++)motionFrames.push({tick:i,missiles:[],aircraft:[{...aircraft,control:{...aircraft.control,heading_change:i<8?.1:i<16?-.1:0}}]});
const smooth=vm.createContext({...context,frames:motionFrames});
vm.runInContext(colorCode,smooth);
vm.runInContext(source.slice(source.indexOf('function targetAircraftBank('),source.indexOf('// Boundary of the actual spherical sector')),smooth);
const rolls=motionFrames.map((f,i)=>smooth.aircraftBank(f.aircraft[0],i));
assert.equal(rolls[0],0);
for(let i=1;i<rolls.length;i++)assert(Math.abs(rolls[i]-rolls[i-1])<=12*Math.PI/180+1e-12);
assert(rolls[8]>0,'reversal must not instantly snap to opposite bank');
assert(rolls[15]<0,'sustained opposite turn eventually reverses bank');
assert(Math.abs(rolls[23])<Math.abs(rolls[16]),'straight flight returns toward level');
assert.equal(smooth.targetAircraftBank({...aircraft,control:{...aircraft.control,heading_change:.0001}},0),0);
assert.equal(smooth.aircraftBank(motionFrames[8].aircraft[0],8,8.5),(rolls[8]+rolls[9])/2);
assert.equal(smooth.aircraftBank(motionFrames[2].aircraft[0],2),rolls[2],'rewind retains attitude');
console.log('PASS: rate limited roll, deadzone, reversal, leveling, interpolation and rewind');

const cameraCanvas={width:800,height:650,addEventListener(){},getBoundingClientRect:()=>({left:10,top:20,width:400,height:400})};
const camera=vm.createContext({yaw:0,tilt:0,zoom:1,panX:0,panY:0,draw(){},byId:id=>id==='view'?cameraCanvas:{}});
vm.runInContext(source.slice(source.indexOf('function resetCamera('),source.indexOf('if(config.models)')),camera);
camera.zoomCamera(2,600,300);
assert.equal(camera.zoom,2);assert.equal(camera.panX,-200);assert.equal(camera.panY,25);
camera.zoomCamera(.5,600,300);assert.equal(camera.panX,0);assert.equal(camera.panY,0);
camera.zoomCamera(100,400,325);assert.equal(camera.zoom,8);
camera.zoomCamera(.0001,400,325);assert.equal(camera.zoom,.5);
camera.resetCamera();assert.equal(camera.zoom,1);assert.equal(camera.panX,0);assert.equal(camera.panY,0);
const center=camera.canvasPosition({clientX:210,clientY:220});assert.equal(center[0],400);assert.equal(center[1],325);
console.log('PASS: anchored zoom, limits, camera reset and letterboxed pointer coordinates');

const rewardViewer=vm.createContext({});
vm.runInContext(source.slice(source.indexOf('function rewardRows('),source.indexOf('function drawRewards(')),rewardViewer);
const rewardRows=rewardViewer.rewardRows({
  components:{'HookReward:example':{blue_0:.1}, 'HookReward:blocked':{blue_1:0}},
  clipping:{example:{blue_0:{discarded_abs:.05}}, blocked:{blue_1:{discarded_abs:.15}}}
});
assert.equal(JSON.stringify(rewardRows),JSON.stringify([
  {reason:'example',id:'blue_0',value:.1,discarded:.05},
  {reason:'blocked',id:'blue_1',value:0,discarded:.15}
]));
console.log('PASS: applied and fully clipped reward rows');
