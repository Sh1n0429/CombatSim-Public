import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { Water } from 'three/addons/objects/Water.js';

const {position,aircraftColor,aircraftLabel,clamp}=ArenaScene;
const vector=p=>new THREE.Vector3(p[0],p[2],-p[1]);
function overviewLabel(a,config,mode,aircraft){
  if(mode==='simple')return aircraftLabel(a).match(/#\d+$/)?.[0]||`#${aircraft.filter(item=>item.team===a.team).findIndex(item=>item.id===a.id)+1}`;
  return `${aircraftLabel(a)} · ${(a.z*(config.aero_length_scale||250)/1000).toFixed(1)} KM`;
}
function noise2(x,y){
  const hash=(x,y)=>{const n=Math.sin(x*127.1+y*311.7)*43758.5453;return n-Math.floor(n);};
  const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy,u=fx*fx*(3-2*fx),v=fy*fy*(3-2*fy);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(hash(ix,iy),hash(ix+1,iy),u),THREE.MathUtils.lerp(hash(ix,iy+1),hash(ix+1,iy+1),u),v);
}
function islandGeometry(width,depth,peak,seed){
  const geometry=new THREE.PlaneGeometry(width,depth,128,96);geometry.rotateX(-Math.PI/2);
  const vertices=geometry.attributes.position,colors=new Float32Array(vertices.count*3),color=new THREE.Color();
  for(let i=0;i<vertices.count;i++){
    const x=vertices.getX(i),z=vertices.getZ(i),u=x/width*2,v=z/depth*2;
    let terrain=0,weight=.55;
    for(let k=0;k<6;k++){terrain+=noise2(u*2**(k+1)+seed*7,v*2**(k+1)+seed*13)*weight;weight*=.5;}
    const coastU=u+(noise2(u*3+seed*5,v*3)-.5)*.5+Math.sin(v*4+seed)*.15;
    const coastV=v+(noise2(u*4,v*4+seed*9)-.5)*.4+Math.sin(u*5-seed)*.12;
    const edge=Math.max(0,1-Math.hypot(coastU,coastV)*(.8+noise2(u*5+seed,v*5)*.45));
    vertices.setY(i,Math.pow(edge,1.25)*terrain*peak*2-.65);
  }
  geometry.computeVertexNormals();
  for(let i=0;i<vertices.count;i++){
    const y=vertices.getY(i),slope=geometry.attributes.normal.getY(i),grain=.86+noise2(vertices.getX(i)*.65,vertices.getZ(i)*.65)*.25;
    color.set(y<.45?0x99957c:slope<.72||y>peak*.62?0x697374:0x344c3d);
    color.multiplyScalar(grain);color.toArray(colors,i*3);
  }
  geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));return geometry;
}
function plumeTexture(){
  const size=128,data=new Uint8Array(size*size*4);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const radius=Math.hypot((x+.5)/size*2-1,(y+.5)/size*2-1),at=(y*size+x)*4;
    data[at]=data[at+1]=data[at+2]=255;
    data[at+3]=Math.max(0,1-radius)**1.6*(.65+noise2(x*.12,y*.12)*.35)*255;
  }
  const texture=new THREE.DataTexture(data,size,size);texture.needsUpdate=true;return texture;
}
function groundTexture(){
  const size=256,data=new Uint8Array(size*size*4);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const at=(y*size+x)*4,grain=noise2(x*.16,y*.16)*.45+noise2(x*.61+8,y*.61)*.35+noise2(x*1.9,y*1.9+9)*.2;
    data[at]=data[at+1]=data[at+2]=125+grain*125;data[at+3]=255;
  }
  const texture=new THREE.DataTexture(data,size,size);texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(64,48);texture.generateMipmaps=true;texture.minFilter=THREE.LinearMipmapLinearFilter;texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;return texture;
}
function particleLayer(count,texture,additive){
  const geometry=new THREE.BufferGeometry();
  for(const [name,components] of [['position',3],['color',3],['particleSize',1],['particleAlpha',1]])geometry.setAttribute(name,new THREE.BufferAttribute(new Float32Array(count*components),components));
  const material=new THREE.ShaderMaterial({transparent:true,depthWrite:false,vertexColors:true,blending:additive?THREE.AdditiveBlending:THREE.NormalBlending,
    uniforms:{plume:{value:texture},pixelHeight:{value:500}},
    vertexShader:`attribute float particleSize;attribute float particleAlpha;varying vec3 tint;varying float alpha;uniform float pixelHeight;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main(){vec4 mv=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*mv;gl_PointSize=clamp(particleSize*pixelHeight/max(.0001,-mv.z),1.,200.);tint=color;alpha=particleAlpha;
      #include <logdepthbuf_vertex>
      }`,
    fragmentShader:`uniform sampler2D plume;varying vec3 tint;varying float alpha;
      #include <logdepthbuf_pars_fragment>
      void main(){gl_FragColor=vec4(tint,texture2D(plume,gl_PointCoord).a*alpha);
      #include <logdepthbuf_fragment>
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
      }`});
  const points=new THREE.Points(geometry,material);points.frustumCulled=false;return points;
}
function updateExplosion(group,state,origin,config,height,length){
  const fire=group.children[0],smoke=group.children[1],age=state.age;
  for(const [layer,isFire] of [[fire,true],[smoke,false]]){
    const attr=layer.geometry.attributes,count=attr.position.count;
    for(let i=0;i<count;i++){
      const phase=i/count,angle=i*2.399963,spread=length*(isFire?.3+Math.min(age,1.5)*2:.5+age*.8);
      const lag=isFire?Math.max(0,age-phase*.5):Math.max(0,age-phase*3);
      const p=vector(position(origin,config,height));
      p.x+=Math.cos(angle)*spread*phase;p.z+=Math.sin(angle)*spread*phase;p.y+=length*(isFire?Math.sin(i*3.7)*phase:lag*.8);
      attr.position.setXYZ(i,p.x,p.y,p.z);
      const opacity=isFire?(Math.max(0,1-lag/2)*(.4+phase*.6)+Math.max(0,1-age/8)*.22):Math.min(1,age*.8)*Math.max(0,1-age/14)*.95;
      attr.particleAlpha.setX(i,opacity);attr.particleSize.setX(i,length*(isFire?.3+phase*.75:1+lag));
      if(isFire)attr.color.setXYZ(i,2.8,phase<.4?1.2:.35,.035);
      else{const shade=.16+phase*.24;attr.color.setXYZ(i,shade,shade*1.03,shade*1.08);}
    }
    for(const attribute of Object.values(attr))attribute.needsUpdate=true;
  }
}
// glTF local axes are converted to nose +X, up +Y before flight attitude is applied.
function aircraftQuaternion(a,bank=0){
  const basis=ArenaScene.aircraftBasis(a,bank);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(vector(basis[0]),vector(basis[2]),vector(basis[1]).negate()));
}
function normalizeAircraft(source){
  source.updateMatrixWorld(true);
  const bounds=new THREE.Box3().setFromObject(source),center=bounds.getCenter(new THREE.Vector3()),length=bounds.max.x-bounds.min.x;
  const normalized=new THREE.Group();
  new THREE.Matrix4().makeScale(1/length,1/length,1/length).multiply(new THREE.Matrix4().makeTranslation(-center.x,-center.y,-center.z)).decompose(normalized.position,normalized.quaternion,normalized.scale);
  // Keep quantized vertex buffers and their decoding node transforms intact.
  normalized.add(source);const root=new THREE.Group();root.add(normalized);return root;
}
function waterNormals(){
  const size=256,data=new Uint8Array(size*size*4);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const surface=(u,v)=>noise2(u*.07,v*.07)*.55+noise2(u*.19+10,v*.19)*.3+noise2(u*.43,v*.43+17)*.15;
    const dx=surface((x+1)%size,y)-surface((x+size-1)%size,y),dy=surface(x,(y+1)%size)-surface(x,(y+size-1)%size);
    const at=(y*size+x)*4;data[at]=128+dx*90;data[at+1]=128+dy*90;data[at+2]=255;data[at+3]=255;
  }
  const texture=new THREE.DataTexture(data,size,size);texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.minFilter=THREE.LinearMipmapLinearFilter;texture.magFilter=THREE.LinearFilter;texture.generateMipmaps=true;texture.needsUpdate=true;return texture;
}
function cloudMaterial(){
  return new THREE.ShaderMaterial({transparent:true,depthWrite:false,side:THREE.DoubleSide,
    vertexShader:`varying vec3 world;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main(){vec4 p=modelMatrix*vec4(position,1.);world=p.xyz;gl_Position=projectionMatrix*viewMatrix*p;
      #include <logdepthbuf_vertex>
      }`,
    fragmentShader:`varying vec3 world;
      #include <logdepthbuf_pars_fragment>
      float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
      float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
      float fbm(vec2 p){float n=0.,a=.5;for(int i=0;i<6;i++){n+=noise(p)*a;p=p*2.03+17.;a*=.5;}return n;}
      void main(){float n=fbm(world.xz*.011),alpha=smoothstep(.57,.79,n)*.5;vec3 color=mix(vec3(.55,.65,.71),vec3(.96,.96,.91),smoothstep(.45,.7,n));gl_FragColor=vec4(color,alpha);
      #include <logdepthbuf_fragment>
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
      }`});
}
function chaseView(a,config,height){
  return {target:position(a,config,height),yaw:a.heading-Math.PI/2+.22,elevation:.28,distance:15*1000/(config.arena_x*(config.aero_length_scale||250))*2.8};
}
function setCameraView(camera,view,aspect){
  camera.aspect=aspect;camera.fov=view.fov||42;camera.up.copy(vector(view.up||[0,0,1]));camera.updateProjectionMatrix();
  const eye=[view.target[0]+Math.sin(view.yaw)*Math.cos(view.elevation)*view.distance,view.target[1]-Math.cos(view.yaw)*Math.cos(view.elevation)*view.distance,view.target[2]+Math.sin(view.elevation)*view.distance];
  camera.position.copy(vector(eye));camera.lookAt(vector(view.target));camera.updateMatrixWorld();
}
function drawAircraftHUD(canvas,a,config,ratio,aircraft=[],missiles=[]){
  const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;
  if(w<=0||h<=0)return;
  const width=Math.round(w*ratio),pixels=Math.round(h*ratio);
  if(canvas.width!==width||canvas.height!==pixels){canvas.width=width;canvas.height=pixels;}
  const c=canvas.getContext('2d');if(!c)return;
  const contacts=ArenaScene.bvrContacts(a,aircraft,config),rockets=ArenaScene.bvrMissiles(a,missiles,config),font=clamp(h*.055,9,12),cx=w/2;
  const top=Math.min(42,h*.35),bottom=h-12,cy=(top+bottom)/2,radius=Math.max(6,Math.min((bottom-top)/2-2,w/2-14));
  const range=gridStep(Math.max(10,(a.radar_range||config.radar_range||0)*(config.aero_length_scale||250)/1000,...contacts.map(t=>t.horizontal*1.1),...contacts.map(t=>(t.rmaxHorizontal||0)*1.1),...rockets.map(t=>t.horizontal*1.1)));
  const heading=String(ArenaScene.flightTelemetry(a,config).heading).padStart(3,'0');
  c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,w,h);
  c.fillStyle='#07141d';c.fillRect(0,0,w,h);c.font=`600 ${font}px monospace`;c.textAlign='center';c.lineWidth=1;
  c.strokeStyle='#8bb7bd30';c.fillStyle='#8bb7bd';
  for(const fraction of [.25,.5,.75,1]){
    c.beginPath();c.arc(cx,cy,radius*fraction,0,Math.PI*2);c.stroke();
    if(fraction===.5&&h>=120)c.fillText(`${range/2}`,cx+radius*fraction+font,cy-3);
  }
  c.setLineDash([2,4]);c.beginPath();c.moveTo(cx-radius,cy);c.lineTo(cx+radius,cy);c.moveTo(cx,cy-radius);c.lineTo(cx,cy+radius);c.stroke();c.setLineDash([]);
  c.fillStyle='#a2bdc4';c.fillText(w>=360?`BVR · HDG ${heading} · RNG ${range} KM`:`${heading} / ${range} KM`,cx,top-9);
  if(h>=100){c.fillText('FWD',cx,cy-radius+font+2);c.fillText('AFT',cx,cy+radius-4);c.fillText('L',cx-radius-9,cy+3);c.fillText('R',cx+radius+9,cy+3);}
  c.strokeStyle=c.fillStyle='#b4ffbd';c.beginPath();c.moveTo(cx,cy-8);c.lineTo(cx-5,cy+5);c.lineTo(cx,cy+2);c.lineTo(cx+5,cy+5);c.closePath();c.stroke();
  for(const contact of contacts){
    if(!(contact.rmax>0))continue;
    const sin=Math.sin(contact.bearing),cos=Math.cos(contact.bearing),distance=contact.rmaxHorizontal/range*radius;
    const x=cx+sin*distance,y=cy-cos*distance;
    c.strokeStyle='#e8c68c45';c.beginPath();c.moveTo(cx,cy);c.lineTo(x,y);c.stroke();
    c.strokeStyle='#e8c68c';c.lineWidth=2;c.beginPath();c.moveTo(x-cos*4,y-sin*4);c.lineTo(x+cos*4,y+sin*4);c.stroke();c.lineWidth=1;
  }
  for(const missile of rockets){
    const x=cx+missile.right/range*radius,y=cy-missile.forward/range*radius;
    const color=missile.source==='own'?'#e8c68c':missile.source==='ally'?'#77e7ff':'#ff5969';
    c.save();c.translate(x,y);c.strokeStyle=c.fillStyle=color;c.rotate(missile.heading);
    c.beginPath();c.moveTo(0,-5);c.lineTo(2,3);c.lineTo(0,1);c.lineTo(-2,3);c.closePath();c.fill();
    c.beginPath();c.moveTo(0,4);c.lineTo(0,11);c.stroke();c.restore();
    c.fillStyle=color;c.textAlign='left';c.fillText('M',x+6,y+3);
  }
  c.textAlign='center';
  const plotted=contacts.map(contact=>{
    const x=cx+contact.right/range*radius,y=cy-contact.forward/range*radius;
    const color=contact.source==='ally'?'#77e7ff':'#ff5969',source={ally:'F',radar:'R',datalink:'DL'}[contact.source];
    const degrees=Math.round(Math.abs(contact.bearing)*180/Math.PI),bearing=degrees===0?'FWD':degrees===180?'AFT':`${contact.bearing<0?'L':'R'}${String(degrees).padStart(3,'0')}°`;
    const altitude=Math.round(contact.altitude),delta=`${altitude<0?'-':'+'}${Math.abs(altitude).toLocaleString()} M`;
    const baseLabel=`${source} ${overviewLabel(contact.target,config,'simple',aircraft)} · ${contact.range.toFixed(1)} KM`;
    const rangeLabel=contact.rmax===null?'':` · RMAX ${contact.rmax.toFixed(1)}`;
    const fullRangeLabel=contact.rmax!==null&&c.measureText(baseLabel+rangeLabel).width<=cx-radius-32;
    const label=baseLabel+(fullRangeLabel?rangeLabel:'');
    c.save();c.translate(x,y);c.strokeStyle=c.fillStyle=color;c.beginPath();
    if(contact.source==='ally')c.arc(0,0,4,0,Math.PI*2);
    else if(contact.source==='radar')c.rect(-4,-4,8,8);
    else{c.moveTo(0,-5);c.lineTo(5,0);c.lineTo(0,5);c.lineTo(-5,0);c.closePath();}
    c.stroke();c.rotate(contact.heading);c.beginPath();c.moveTo(0,-6);c.lineTo(0,-14);c.moveTo(-2,-11);c.lineTo(0,-14);c.lineTo(2,-11);c.stroke();c.restore();
    return {...contact,x,y,color,label,bearing,delta,fullRangeLabel};
  });
  if(w>=360&&h>=120){
    const row=font*2+9,minY=top+font,maxY=bottom-font-8,limit=Math.max(1,Math.floor((maxY-minY)/row)+1);
    for(const side of [-1,1]){
      const column=plotted.filter(p=>(p.right<0?-1:1)===side).sort((a,b)=>a.range-b.range).slice(0,limit).sort((a,b)=>a.y-b.y);
      const x=cx+side*(radius+22);let lastY=minY-row;
      for(let i=0;i<column.length;i++){
        const p=column[i],y=clamp(p.y-font/2,Math.max(minY,lastY+row),maxY-(column.length-1-i)*row);lastY=y;
        c.strokeStyle=p.color+'60';c.beginPath();c.moveTo(p.x,p.y);c.lineTo(x-side*5,y-3);c.stroke();
        c.textAlign=side<0?'right':'left';c.fillStyle=p.color;c.fillText(p.label,x,y);c.fillText(p.rmax!==null&&!p.fullRangeLabel?`RMAX${Math.round(p.rmax)} Δ${p.delta.replaceAll(' ','')}`:`${p.bearing} · Δ ${p.delta}`,x,y+font+3);
      }
    }
  }
  c.textAlign='center';
  if(!contacts.length&&!rockets.length){c.fillStyle='#8bb7bd';c.fillText(!a.alive?'TRACK ENDED':a.hud_contacts?'NO CONTACTS':'NO TRACK DATA',cx,cy+font*2+7);}
  const summary=plotted.map(p=>`${p.label}、相対方位${p.bearing}、高度差${p.delta}${p.rmax===null?'':`、推定RMAX ${p.rmax.toFixed(1)}km`}`).join('。');
  const missileSummary=rockets.map(m=>`${{own:'自機弾',ally:'味方弾',radar:'レーダー捕捉の敵弾',datalink:'データリンクの敵弾'}[m.source]} ${m.target.id}、距離${m.range.toFixed(1)}km、相対方位${Math.round(m.bearing*180/Math.PI)}度`).join('。');
  canvas.setAttribute('aria-label',`BVR戦術HSD。上が機首方向、自機方位${heading}度、表示半径${range}km。${summary||(!a.alive?'追跡終了':a.hud_contacts?'捕捉機なし':'接触情報なし')}。${missileSummary}`);
}
function overviewLength(actual,distance,height,pixels=18){
  return Math.max(actual,2*Math.tan(21*Math.PI/180)*distance*pixels/Math.max(1,height));
}
function gridStep(distance){
  const magnitude=10**Math.floor(Math.log10(Math.max(.1,distance)));
  return [1,2,5,10].map(n=>n*magnitude).find(n=>n>=Math.max(.1,distance));
}
function distanceGrid(config,view){
  const units=1000/(config.aero_length_scale||250),kmPerScene=config.arena_x/units/1000;
  const step=gridStep(view.distance*kmPerScene/14),radius=view.distance*kmPerScene*.65;
  const anchor=[view.target[0]*kmPerScene+config.arena_x/units/2,view.target[1]*kmPerScene+config.arena_y/units/2]
    .map((n,i)=>clamp(Math.round(n/step)*step,0,(i===0?config.arena_x:config.arena_y)/units));
  const lo=anchor.map(n=>Math.max(0,Math.floor((n-radius)/step)*step));
  const hi=anchor.map((n,i)=>Math.min((i===0?config.arena_x:config.arena_y)/units,Math.ceil((n+radius)/step)*step));
  const lines=[],labels=[],text=n=>Number(n.toFixed(2));
  for(let x=lo[0];x<=hi[0]+step*.001;x+=step){
    lines.push({from:[x,lo[1],0],to:[x,hi[1],0]});labels.push({at:[x,anchor[1],0],text:`X ${text(x)} KM`});
  }
  for(let y=lo[1];y<=hi[1]+step*.001;y+=step){
    lines.push({from:[lo[0],y,0],to:[hi[0],y,0]});labels.push({at:[anchor[0],y,0],text:`Y ${text(y)} KM`});
  }
  const ceiling=config.ceiling/units;
  lines.push({from:[lo[0],anchor[1],0],to:[hi[0],anchor[1],0],axis:'x'},
    {from:[anchor[0],lo[1],0],to:[anchor[0],hi[1],0],axis:'y'},
    {from:[...anchor,0],to:[...anchor,ceiling],axis:'z'});
  const verticalStep=gridStep(ceiling/4);
  for(let z=verticalStep;z<=ceiling;z+=verticalStep)labels.push({at:[...anchor,z],text:`Z ${text(z)} KM`});
  return {step,units,lines,labels};
}
function drawDistanceGrid(c,camera,config,view,height,w,h){
  const grid=distanceGrid(config,view),frustum=new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse));
  const point=([x,y,z])=>vector(position({x:x*grid.units,y:y*grid.units,z:z*grid.units},config,height));
  c.save();c.beginPath();c.rect(0,0,w,h);c.clip();c.lineWidth=1;
  for(const line of grid.lines){
    const a=point(line.from),b=point(line.to);let visible=true;
    for(const plane of frustum.planes){
      const da=plane.distanceToPoint(a),db=plane.distanceToPoint(b);
      if(da<0&&db<0){visible=false;break;}
      if(da<0)a.lerp(b,da/(da-db));else if(db<0)b.lerp(a,db/(db-da));
    }
    if(!visible)continue;
    a.project(camera);b.project(camera);c.strokeStyle=line.axis?'#c2e4e253':'#a5dce023';c.setLineDash(line.axis==='z'?[3,5]:[]);
    c.beginPath();c.moveTo((a.x*.5+.5)*w,(-a.y*.5+.5)*h);c.lineTo((b.x*.5+.5)*w,(-b.y*.5+.5)*h);c.stroke();
  }
  c.setLineDash([]);c.font='9px monospace';c.textAlign='left';c.fillStyle='#b9d5da8c';
  const placed=[];
  for(const label of grid.labels){
    const p=point(label.at);if(!frustum.containsPoint(p))continue;p.project(camera);
    const x=(p.x*.5+.5)*w+5,y=(-p.y*.5+.5)*h-5,width=c.measureText(label.text).width;
    if(x<8||x+width>w-8||y<38||y>h-50||placed.some(r=>Math.abs(y-r.y)<14&&x<r.x+r.width+8&&x+width+8>r.x))continue;
    c.fillText(label.text,x,y);placed.push({x,y,width});
  }
  c.fillText(`GRID ${grid.step} KM`,18,h-40);c.restore();
}
function createMissile(){
  const missile=new THREE.Group(),body=new THREE.Group();missile.add(body);body.scale.setScalar(1/1.42);body.position.x=-.01/1.42;
  const paint=new THREE.MeshStandardMaterial({color:0xe2e4dd,metalness:.3,roughness:.42});
  const barrel=new THREE.Mesh(new THREE.CylinderGeometry(.055,.055,1,12),paint);barrel.rotation.z=-Math.PI/2;body.add(barrel);
  const nose=new THREE.Mesh(new THREE.ConeGeometry(.055,.22,12),new THREE.MeshStandardMaterial({color:0x69747a,metalness:.4,roughness:.35}));nose.rotation.z=-Math.PI/2;nose.position.x=.61;body.add(nose);
  const fin=new THREE.BufferGeometry();fin.setAttribute('position',new THREE.Float32BufferAttribute([-.48,.045,0,-.62,.21,0,-.18,.045,0],3));fin.computeVertexNormals();
  const metal=new THREE.MeshStandardMaterial({color:0x929b9e,metalness:.45,roughness:.4,side:THREE.DoubleSide});
  for(let i=0;i<4;i++){const wing=new THREE.Mesh(fin,metal);wing.rotation.x=i*Math.PI/2;body.add(wing);}
  const flame=new THREE.Mesh(new THREE.ConeGeometry(.045,.2,12),new THREE.MeshBasicMaterial({color:0xffb34b,transparent:true,opacity:.85,depthWrite:false}));flame.rotation.z=Math.PI/2;flame.position.x=-.6;body.add(flame);
  return missile;
}
function viewportBounds(rect,root,clip=root){
  const left=Math.max(rect.left,root.left,clip.left),right=Math.min(rect.right,root.right,clip.right);
  const top=Math.max(rect.top,root.top,clip.top),bottom=Math.min(rect.bottom,root.bottom,clip.bottom);
  return {x:rect.left-root.left,y:root.bottom-rect.bottom,width:rect.width,height:rect.height,
    scissor:[left-root.left,root.bottom-bottom,Math.max(0,right-left),Math.max(0,bottom-top)]};
}
export function createArenaRenderer(canvas,overlay,mainViewport){
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,logarithmicDepthBuffer:true,powerPreference:'high-performance'});
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFShadowMap;
  renderer.autoClear=false;
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.00008,100000);
  scene.fog=new THREE.FogExp2(0x8ca4b2,.00065);
  const sunDirection=new THREE.Vector3(-.6,.29,-.72).normalize();
  const sky=new Sky();sky.scale.setScalar(90000);scene.add(sky);
  const uniforms=sky.material.uniforms;uniforms.turbidity.value=3;uniforms.rayleigh.value=1.8;uniforms.mieCoefficient.value=.003;uniforms.mieDirectionalG.value=.82;uniforms.sunPosition.value.copy(sunDirection);
  const environmentScene=new THREE.Scene();environmentScene.add(sky.clone());
  const pmrem=new THREE.PMREMGenerator(renderer);const environment=pmrem.fromScene(environmentScene,.02,.1,100000);scene.environment=environment.texture;pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xc5e2ff,0x40566a,1.1));
  const sunlight=new THREE.DirectionalLight(0xffedcd,3.1);sunlight.castShadow=true;sunlight.shadow.mapSize.set(2048,2048);sunlight.shadow.camera.near=.1;sunlight.shadow.camera.far=5;sunlight.shadow.bias=-.00006;sunlight.shadow.normalBias=.000015;scene.add(sunlight,sunlight.target);
  const impactLight=new THREE.PointLight(0xff8d38,0,2,1);scene.add(impactLight);
  const ocean=new Water(new THREE.PlaneGeometry(100000,100000),{textureWidth:512,textureHeight:512,waterNormals:waterNormals(),sunDirection,sunColor:0xffe5bc,waterColor:0x103c50,distortionScale:2.8,fog:true});
  ocean.rotation.x=-Math.PI/2;ocean.material.uniforms.size.value=24;scene.add(ocean);
  const islands=new THREE.Group(),groundMap=groundTexture(),ground=new THREE.MeshStandardMaterial({vertexColors:true,map:groundMap,bumpMap:groundMap,bumpScale:.025,roughness:.96,metalness:0});groundMap.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
  for(const [x,z,width,depth,peak,seed] of [[-380,-410,850,460,28,1],[500,-600,680,410,38,2],[-150,-1050,1000,420,45,3]]){
    const island=new THREE.Mesh(islandGeometry(width,depth,peak,seed),ground);island.position.set(x,0,z);islands.add(island);
  }
  scene.add(islands);
  const clouds=new THREE.Mesh(new THREE.PlaneGeometry(14000,14000),cloudMaterial());clouds.rotation.x=-Math.PI/2;clouds.position.y=12;clouds.renderOrder=1;scene.add(clouds);
  const aircraft=new Map(),missiles=new Map(),trails=new Map(),bursts=new Map(),losses=new Map(),reticle=new THREE.Vector3(),plume=plumeTexture();
  const paintedMaterials=new Map();
  let template=null,lastSize='',lastTick=-1,lastConfig='',disposed=false,previous=null,cinemaTime=0;
  const loader=new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  const ready=loader.loadAsync('/arena-assets/f16.glb').then(gltf=>{
    const source=gltf.scene,materials=new Map();
    source.traverse(node=>{
      if(!node.isMesh)return;
      const original=node.material;
      let material=materials.get(original);
      if(!material){
        material=original.clone();material.envMapIntensity=.7;
        if(original.name==='Black')material=new THREE.MeshPhysicalMaterial({color:0x19262e,metalness:.35,roughness:.14,clearcoat:1,clearcoatRoughness:.06,envMapIntensity:1.6,side:THREE.DoubleSide});
        else{material.metalness=.3;material.roughness=.48;material.color.set(['Grey','LightGrey'].includes(original.name)?(original.name==='Grey'?0x727e87:0x939da4):original.name==='Material'?0x394047:0x8a9398);if(material.emissive)material.emissive.multiplyScalar(.03);}
        material.name=original.name;
        if(material.map)material.map.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
        materials.set(original,material);
      }
      node.material=material;node.castShadow=node.receiveShadow=true;
    });
    template=normalizeAircraft(source);
    if(disposed)disposeObject(template);
  });
  function paintAircraft(jet,tint){
    jet.traverse(node=>{
      if(!node.isMesh||!node.material?.color||node.material.name==='Black')return;
      const base=node.material;
      if(!paintedMaterials.has(base))paintedMaterials.set(base,new Map());
      const palette=paintedMaterials.get(base);
      if(!palette.has(tint)){
        const paint=base.clone();
        paint.color.lerp(new THREE.Color(tint),.72);
        palette.set(tint,paint);
      }
      node.material=palette.get(tint);
    });
  }
  function disposeObject(root){root.traverse(node=>{node.geometry?.dispose();if(node.material)for(const material of Array.isArray(node.material)?node.material:[node.material])material.dispose();});}
  function dynamicLine(map,id,color,count){
    let line=map.get(id);
    if(!line){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(count*3),3));line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity:.55,depthWrite:false}));line.frustumCulled=false;map.set(id,line);scene.add(line);}
    return line;
  }
  function removeMissing(map,ids,shared=false){for(const [id,object] of map)if(!ids.has(id)){scene.remove(object);if(!shared)disposeObject(object);map.delete(id);}}
  function renderViewport(element,clip,root,view){
    const rect=element.getBoundingClientRect(),box=viewportBounds(rect,root,clip.getBoundingClientRect());
    if(box.scissor[2]<=0||box.scissor[3]<=0)return;
    setCameraView(camera,view,rect.width/rect.height);
    for(const burst of bursts.values())for(const layer of burst.children)layer.material.uniforms.pixelHeight.value=rect.height*renderer.getPixelRatio();
    renderer.setViewport(box.x,box.y,box.width,box.height);renderer.setScissor(...box.scissor);renderer.clear();renderer.render(scene,camera);
  }
  let aircraftBank=null;
  function draw(frame,frames,index,config,view,options,effects,feeds,dt,paused){
    aircraftBank ??= ArenaScene.createBankSampler(frames,config);
    const root=canvas.getBoundingClientRect(),rect=mainViewport.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,options.quality==='high'?2:1),size=`${root.width}:${root.height}:${rect.width}:${rect.height}:${ratio}`;
    if(rect.width<=0||rect.height<=0||disposed)return;
    if(size!==lastSize){renderer.setPixelRatio(ratio);renderer.setSize(root.width,root.height,false);overlay.width=Math.round(rect.width*ratio);overlay.height=Math.round(rect.height*ratio);lastSize=size;}
    const unit=1000/(config.arena_x*(config.aero_length_scale||250)),length=15*unit;
    if(!paused)cinemaTime+=dt/1000;
    ArenaScene.updateLossCues(losses,frame,effects,previous,cinemaTime);previous=frame;
    const wrecks=new Map([...losses].map(([id,cue])=>[id,ArenaScene.wreckState(cue,cinemaTime)]));
    const mainEye=vector([view.target[0]+Math.sin(view.yaw)*Math.cos(view.elevation)*view.distance,view.target[1]-Math.cos(view.yaw)*Math.cos(view.elevation)*view.distance,view.target[2]+Math.sin(view.elevation)*view.distance]);
    const displayPixels=options.aircraftSize??18;
    const displayLengths=new Map(frame.aircraft.map(a=>[a.id,overviewLength(length,mainEye.distanceTo(vector(position(a,config,options.height))),rect.height,displayPixels)]));
    function sceneScale(enlarged){
      const largest=enlarged?Math.max(length,...displayLengths.values()):length,shadow=sunlight.shadow.camera;
      shadow.left=shadow.bottom=-largest*1.4;shadow.right=shadow.top=largest*1.4;shadow.far=Math.max(5,largest*8);shadow.updateProjectionMatrix();
      for(const [id,jet] of aircraft)jet.scale.setScalar(enlarged?displayLengths.get(id)||length:length);
      for(const missile of missiles.values())missile.scale.setScalar(enlarged?overviewLength(3.6*unit,mainEye.distanceTo(missile.position),rect.height,displayPixels*.35):3.6*unit);
      for(const [id,burst] of bursts)updateExplosion(burst,wrecks.get(id),losses.get(id).event.position,config,options.height,enlarged?displayLengths.get(id)||length:length);
      return Math.max(2,largest*3);
    }
    sunlight.target.position.copy(vector(view.target));
    const waterTime=frame.tick*(config.aero_tick_seconds||1)*.015;ocean.material.uniforms.time.value=waterTime;
    const alive=frame.aircraft.filter(a=>a.alive),visibleAircraft=[...alive,...[...wrecks.values()].filter(s=>s.visible).map(s=>s.pose)],aircraftIds=new Set(visibleAircraft.map(a=>a.id));removeMissing(aircraft,aircraftIds,true);
    for(const a of visibleAircraft){
      let jet=aircraft.get(a.id);
      if(!jet&&template){jet=template.clone(true);paintAircraft(jet,aircraftColor(a));jet.scale.setScalar(length);aircraft.set(a.id,jet);scene.add(jet);}
      if(jet){jet.position.copy(vector(position(a,config,options.height)));jet.quaternion.copy(aircraftQuaternion(a,aircraftBank(a,index,frame.tick)));}
    }
    const missileIds=new Set(frame.missiles.map(m=>m.id));removeMissing(missiles,missileIds);
    for(const m of frame.missiles){
      let mesh=missiles.get(m.id);
      if(!mesh){mesh=createMissile();mesh.scale.setScalar(3.6*unit);missiles.set(m.id,mesh);scene.add(mesh);}
      mesh.position.copy(vector(position(m,config,options.height)));mesh.quaternion.copy(aircraftQuaternion(m));
    }
    const changed=frame.tick!==lastTick||lastConfig!==`${options.height}:${options.trails}`;
    if(changed){
      const visible=new Set();
      if(options.trails){
        for(const entity of [...visibleAircraft,...frame.missiles]){
          visible.add(entity.id);const isMissile=missileIds.has(entity.id),line=dynamicLine(trails,entity.id,isMissile?0xe7e7de:aircraftColor(entity),96);
          const points=[];
          for(let i=Math.max(0,index-(isMissile?45:90));i<=index;i++){
            const item=(isMissile?frames[i].missiles:frames[i].aircraft).find(a=>a.id===entity.id&&a.alive!==false);
            if(item)points.push(vector(position(item,config,options.height)));
          }
          points.push(vector(position(entity,config,options.height)));
          const attr=line.geometry.attributes.position;points.forEach((p,i)=>attr.setXYZ(i,p.x,p.y,p.z));attr.needsUpdate=true;line.geometry.setDrawRange(0,points.length);
          line.material.opacity=(isMissile?.85:.48)*(wrecks.has(entity.id)?1-wrecks.get(entity.id).fade:1);
        }
      }
      removeMissing(trails,visible);lastTick=frame.tick;lastConfig=`${options.height}:${options.trails}`;
    }
    const activeBursts=new Set();
    impactLight.intensity=0;
    if(options.effects)for(const [id,state] of wrecks){
      if(state.age>14)continue;activeBursts.add(id);
      let burst=bursts.get(id);if(!burst){burst=new THREE.Group();burst.add(particleLayer(36,plume,true),particleLayer(32,plume,false));scene.add(burst);bursts.set(id,burst);}
      updateExplosion(burst,state,losses.get(id).event.position,config,options.height,length);
      if(state.flash>0){impactLight.position.copy(vector(position(state.pose,config,options.height)));impactLight.intensity=state.flash*8;impactLight.distance=length*15;}
    }
    removeMissing(bursts,activeBursts);
    renderer.setScissorTest(false);renderer.setViewport(0,0,root.width,root.height);renderer.clear();renderer.setScissorTest(true);
    sunlight.position.copy(sunlight.target.position).addScaledVector(sunDirection,sceneScale(true));
    renderViewport(mainViewport,mainViewport,root,view);
    // Labels and trails identify the enlarged 3D aircraft without replacing their geometry.
    const c=overlay.getContext('2d'),w=rect.width,h=rect.height;c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,w,h);
    if(options.grid)drawDistanceGrid(c,camera,config,view,options.height,w,h);
    if(options.trails)for(const entity of [...visibleAircraft,...frame.missiles]){
      const missile=missileIds.has(entity.id);c.strokeStyle=missile?'#ffdfa0':aircraftColor(entity);c.lineWidth=missile?1.7:1.2;c.globalAlpha=missile?.85:.45;c.beginPath();let started=false;
      for(let i=Math.max(0,index-(missile?35:55));i<=index;i++){
        const point=(missile?frames[i].missiles:frames[i].aircraft).find(a=>a.id===entity.id);
        if(!point)continue;reticle.copy(vector(position(point,config,options.height))).project(camera);
        if(reticle.z<-1||reticle.z>1){started=false;continue;}
        const x=(reticle.x*.5+.5)*w,y=(-reticle.y*.5+.5)*h;if(started)c.lineTo(x,y);else c.moveTo(x,y);started=true;
      }
      reticle.copy(vector(position(entity,config,options.height))).project(camera);if(started)c.lineTo((reticle.x*.5+.5)*w,(-reticle.y*.5+.5)*h);c.stroke();c.globalAlpha=1;
    }
    const labels=[];
    for(const a of visibleAircraft){
      const center=vector(position(a,config,options.height));
      reticle.copy(center).project(camera);if(reticle.z>1||reticle.z<-1)continue;const x=(reticle.x*.5+.5)*w,y=(-reticle.y*.5+.5)*h;
      if(x<0||x>w||y<0||y>h)continue;
      const state=wrecks.get(a.id),color=state?'#e7b179':aircraftColor(a);c.globalAlpha=state?1-state.fade:1;c.strokeStyle=color;c.fillStyle=color;c.lineWidth=1;
      if(options.labels!=='off'){
        const label=overviewLabel(a,config,options.labels,frame.aircraft);
        c.font='600 9px monospace';const width=c.measureText(label).width+12,lx=clamp(x+displayPixels*.45,8,w-width-8);let ly=clamp(y-8,30,h-25);
        for(const offset of [0,22,-22,44,-44]){const test=clamp(y-8+offset,30,h-25);if(!labels.some(r=>lx<r.x+r.w&&lx+width>r.x&&test-12<r.y+17&&test+5>r.y)){ly=test;break;}}
        labels.push({x:lx,y:ly-12,w:width});c.strokeStyle=color+'70';c.beginPath();c.moveTo(x+5,y);c.lineTo(lx,ly-3);c.stroke();c.fillStyle='#07141ec4';c.fillRect(lx-4,ly-12,width,18);c.fillStyle=color;c.fillText(label,lx+2,ly);
      }
    }
    c.globalAlpha=1;
    if(options.effects)for(const state of wrecks.values()){
      if(state.age>1.8)continue;reticle.copy(vector(position(state.pose,config,options.height))).project(camera);if(reticle.z<-1||reticle.z>1)continue;
      const x=(reticle.x*.5+.5)*w,y=(-reticle.y*.5+.5)*h,r=12+state.age*18,glow=c.createRadialGradient(x,y,0,x,y,r);
      glow.addColorStop(0,'#fff4cf');glow.addColorStop(.15,'#ffb44be0');glow.addColorStop(.5,'#ff791c70');glow.addColorStop(1,'#ff791c00');c.globalAlpha=(1-state.age/1.8)**2;c.fillStyle=glow;c.fillRect(x-r,y-r,r*2,r*2);c.strokeStyle='#ffddaa80';c.beginPath();c.arc(x,y,r*.8,0,Math.PI*2);c.stroke();c.globalAlpha=1;
    }
    sceneScale(false);
    for(const feed of feeds){
      const a=frame.aircraft.find(a=>a.id===feed.id);
      if(!a)continue;
      const state=wrecks.get(a.id),pose=a;
      feed.node.classList.toggle('lost',Boolean(state&&state.age>=5));
      feed.status.textContent=state?(state.age>=8?'SIGNAL LOST':losses.get(a.id).event.type==='crash'?'AIRCRAFT DOWN':'SHOT DOWN'):'';
      feed.status.style.opacity=state?String(clamp((state.age-3)/4,0,.85)):'0';
      const selected=vector(position(pose,config,options.height));sunlight.target.position.copy(selected);sunlight.position.copy(selected).addScaledVector(sunDirection,2);
      if(feed.mode==='hud')drawAircraftHUD(feed.hud,a,config,ratio,frame.aircraft,frame.missiles);
      else renderViewport(feed.node,feed.clip,root,chaseView(pose,config,options.height));
    }
    renderer.setScissorTest(false);
  }
  function reset(){
    aircraftBank=null;
    losses.clear();previous=null;cinemaTime=0;lastTick=-1;lastConfig='';
    for(const map of [trails,bursts,missiles])removeMissing(map,new Set());removeMissing(aircraft,new Set(),true);
  }
  function dispose(){
    disposed=true;for(const map of [missiles,trails,bursts])for(const object of map.values())disposeObject(object);
    for(const palette of paintedMaterials.values())for(const material of palette.values())material.dispose();
    paintedMaterials.clear();
    if(template)disposeObject(template);plume.dispose();groundMap.dispose();disposeObject(islands);ocean.material.uniforms.normalSampler.value.dispose();disposeObject(ocean);disposeObject(clouds);sky.geometry.dispose();sky.material.dispose();environment.dispose();sunlight.shadow.dispose();renderer.dispose();
  }
  return {draw,ready,reset,dispose};
}
