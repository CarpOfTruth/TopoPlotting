let grid = null;
let cfg = null;
let contourGeneration = 0;
const yieldToMessages = () => new Promise(resolve => setTimeout(resolve, 0));

self.onmessage = async event => {
  const msg = event.data;
  if (msg.type === 'init') {
    contourGeneration++;
    cfg = msg.config;
    const response = await fetch(msg.terrainUrl || 'data/terrain-4m.bin');
    grid = new Float32Array(await response.arrayBuffer());
    self.postMessage({type: 'ready'});
    return;
  }
  if (msg.type === 'contours' && grid) {
    const generation = ++contourGeneration;
    calculate(msg, generation);
  }
};

function value(col, row) { return grid[row * cfg.width + col]; }
function interp(a, b, va, vb, level) {
  const d = vb - va;
  const t = Math.abs(d) < 1e-7 ? .5 : (level - va) / d;
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

function rangeInside(bounds) {
  const dx=(cfg.bounds[2]-cfg.bounds[0])/(cfg.width-1),dy=(cfg.bounds[3]-cfg.bounds[1])/(cfg.height-1);
  const c0=Math.max(0,Math.floor((bounds[0]-cfg.bounds[0])/dx)),c1=Math.min(cfg.width-1,Math.ceil((bounds[2]-cfg.bounds[0])/dx));
  const r0=Math.max(0,Math.floor((cfg.bounds[3]-bounds[3])/dy)),r1=Math.min(cfg.height-1,Math.ceil((cfg.bounds[3]-bounds[1])/dy));
  let landMin=Infinity,landMax=-Infinity,waterMin=Infinity,waterMax=-Infinity;
  for(let r=r0;r<=r1;r++)for(let c=c0;c<=c1;c++){
    const z=value(c,r);if(!Number.isFinite(z))continue;
    if(z>0){landMin=Math.min(landMin,z);landMax=Math.max(landMax,z);}
    else if(z<0){waterMin=Math.min(waterMin,z);waterMax=Math.max(waterMax,z);}
  }
  return {landMin,landMax,waterMin,waterMax};
}

function dynamicBanding(msg,bands,baseInterval,bathyBaseInterval) {
  if(!msg.dynamicColors)return null;
  const ranges=rangeInside(msg.colorBounds||msg.view);
  const firstLand=cfg.zeroBoundary?bands.findIndex(b=>b.max===null||b.max>0):0;
  const waterCount=Math.max(0,firstLand),landStart=waterCount,landCount=bands.length-landStart;
  const assigned=Array.from({length:bands.length},()=>[]),indexBand=new Map();
  const addLevels=(lo,hi,step,filter,out)=>{
    if(!Number.isFinite(lo)||!Number.isFinite(hi)||step<=0)return;
    for(let z=Math.ceil(lo/step)*step;z<=hi+step*1e-7;z+=step)if(filter(z))out.add(+z.toFixed(6));
  };
  const land=new Set(),water=new Set();
  addLevels(ranges.landMin,ranges.landMax,baseInterval,z=>z>0,land);
  addLevels(ranges.waterMin,ranges.waterMax,bathyBaseInterval,z=>z<0,water);
  const waterLevels=[...water].sort((a,b)=>a-b),landLevels=[...land].sort((a,b)=>a-b);
  const assign=(levels,start,count)=>{if(!count||!levels.length)return;levels.forEach((z,i)=>{const band=start+Math.min(count-1,Math.floor(i*count/levels.length));assigned[band].push(z);indexBand.set(z.toFixed(6),band);});};
  assign(waterLevels,0,waterCount);assign(landLevels,landStart,landCount);
  const waterByMagnitude=[...waterLevels].sort((a,b)=>Math.abs(a)-Math.abs(b));
  const landByMagnitude=[...landLevels].sort((a,b)=>Math.abs(a)-Math.abs(b));
  const bandFromIndexes=(level,levelsByMagnitude,fallback)=>{
    if(!levelsByMagnitude.length)return fallback;
    const magnitude=Math.abs(level);let lo=0,hi=levelsByMagnitude.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(Math.abs(levelsByMagnitude[mid])+1e-6<magnitude)lo=mid+1;else hi=mid;}
    const index=levelsByMagnitude[Math.min(lo,levelsByMagnitude.length-1)];
    return indexBand.get(index.toFixed(6))??fallback;
  };
  const fmt=z=>`${Number(z.toFixed(Math.abs(z-Math.round(z))<1e-6?0:1))} m`;
  const labels=assigned.map((a,i)=>!a.length?`${bands[i].label} · no index contour in this bin`:a.length===1?`${fmt(a[0])} index contour`:`${fmt(a[0])} to ${fmt(a[a.length-1])} index contours`);
  return {bandFor:level=>level<0?bandFromIndexes(level,waterByMagnitude,Math.max(0,waterCount-1)):bandFromIndexes(level,landByMagnitude,landStart),labels,landLevels:landLevels.length,waterLevels:waterLevels.length};
}

function stitch(data, processPath) {
  const count=data.length/4, used=new Uint8Array(count), links=new Map();
  const key=(x,y)=>`${x.toFixed(3)},${y.toFixed(3)}`;
  const addLink=(k,i)=>{const a=links.get(k);if(a)a.push(i);else links.set(k,[i]);};
  for(let i=0;i<count;i++){const o=i*4;addLink(key(data[o],data[o+1]),i);addLink(key(data[o+2],data[o+3]),i);}
  const point=(i,end)=>{const o=i*4+end*2;return[data[o],data[o+1]];};
  const walk=(start,startEnd)=>{
    const pts=[point(start,startEnd),point(start,1-startEnd)];used[start]=1;
    while(true){const tail=pts[pts.length-1],k=key(tail[0],tail[1]),next=(links.get(k)||[]).find(i=>!used[i]);if(next===undefined)break;const a=point(next,0),aMatches=key(a[0],a[1])===k;pts.push(point(next,aMatches?1:0));used[next]=1;}
    return pts;
  };
  const paths=[];
  for(let i=0;i<count;i++){if(used[i])continue;const a=point(i,0),b=point(i,1),da=(links.get(key(a[0],a[1]))||[]).length,db=(links.get(key(b[0],b[1]))||[]).length;if(da!==2||db!==2)paths.push(walk(i,da!==2?0:1));}
  for(let i=0;i<count;i++)if(!used[i])paths.push(walk(i,0));
  const out=[];
  for(const p of paths){
    const selected=processPath?processPath(p):[p];
    for(const part of selected){for(const [x,y] of part)out.push(x,y);out.push(NaN,NaN);}
  }
  return new Float32Array(out);
}

function dataPaths(data){
  const paths=[];let path=[];
  for(let i=0;i<data.length;i+=2){if(Number.isNaN(data[i])){if(path.length>1)paths.push(path);path=[];}else path.push([data[i],data[i+1]]);}
  if(path.length>1)paths.push(path);return paths;
}

function packPaths(paths){
  let length=0;for(const path of paths)length+=path.length*2+2;
  const out=new Float32Array(length);let o=0;
  for(const path of paths){for(const [x,y] of path){out[o++]=x;out[o++]=y;}out[o++]=NaN;out[o++]=NaN;}
  return out;
}

function roundedPath(path,percent){
  const amount=Math.max(0,Math.min(1,(percent||0)/100));if(!amount||path.length<3)return path;
  const closed=Math.hypot(path[0][0]-path.at(-1)[0],path[0][1]-path.at(-1)[1])<.02;
  let points=closed?path.slice(0,-1):path.slice();
  // Fair the centerline before rounding it. Marching squares can leave alternating
  // one-cell direction errors along an otherwise straight contour; corner cutting
  // alone preserves that wobble and makes a thick stroke look scalloped.
  const fairPasses=Math.max(1,Math.round(amount*4)),weight=.2*amount;
  for(let pass=0;pass<fairPasses;pass++){
    const source=points,next=source.map(p=>[p[0],p[1]]),start=closed?0:1,end=closed?source.length:source.length-1;
    for(let i=start;i<end;i++){const a=source[(i-1+source.length)%source.length],p=source[i],b=source[(i+1)%source.length];next[i]=[p[0]+weight*((a[0]+b[0])/2-p[0]),p[1]+weight*((a[1]+b[1])/2-p[1])];}
    points=next;
  }
  // Sample a C1-continuous cardinal spline through the faired centerline. This
  // makes the contour itself a smooth curve rather than a chain of softened
  // straight segments. A conservative tangent limits overshoot at tight bends.
  const source=points,curve=[],count=source.length,segments=closed?count:count-1,subdivisions=3,tangent=.35+.1*amount,get=i=>closed?source[(i+count)%count]:source[Math.max(0,Math.min(count-1,i))];
  curve.push([...source[0]]);
  for(let i=0;i<segments;i++){
    const p0=get(i-1),p1=get(i),p2=get(i+1),p3=get(i+2),m1=[(p2[0]-p0[0])*tangent,(p2[1]-p0[1])*tangent],m2=[(p3[0]-p1[0])*tangent,(p3[1]-p1[1])*tangent];
    for(let step=1;step<=subdivisions;step++){const t=step/subdivisions,t2=t*t,t3=t2*t,h00=2*t3-3*t2+1,h10=t3-2*t2+t,h01=-2*t3+3*t2,h11=t3-t2;curve.push([h00*p1[0]+h10*m1[0]+h01*p2[0]+h11*m2[0],h00*p1[1]+h10*m1[1]+h01*p2[1]+h11*m2[1]]);}
  }
  points=curve;
  if(closed&&Math.hypot(points[0][0]-points.at(-1)[0],points[0][1]-points.at(-1)[1])>1e-8)points.push([...points[0]]);return points;
}

function pointSegmentDistance(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy;if(d<1e-12)return Math.hypot(p[0]-a[0],p[1]-a[1]);const t=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)),x=a[0]+t*dx,y=a[1]+t*dy;return Math.hypot(p[0]-x,p[1]-y);}

function spacingIndex(cellSize,clearGap,maxWidth){
  const cells=new Map(),key=(x,y)=>`${x},${y}`,cell=v=>Math.floor(v/cellSize);
  const addPath=(path,width)=>{for(let i=1;i<path.length;i++){const a=path[i-1],b=path[i],x0=cell(Math.min(a[0],b[0])),x1=cell(Math.max(a[0],b[0])),y0=cell(Math.min(a[1],b[1])),y1=cell(Math.max(a[1],b[1])),segment={a,b,width};for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const k=key(x,y),bucket=cells.get(k);if(bucket)bucket.push(segment);else cells.set(k,[segment]);}}};
  const clear=(p,width)=>{const maximum=Math.max(0,clearGap+(width+maxWidth)/2);if(maximum<=0)return true;const cx=cell(p[0]),cy=cell(p[1]),reach=Math.ceil(maximum/cellSize)+1,seen=new Set();for(let y=cy-reach;y<=cy+reach;y++)for(let x=cx-reach;x<=cx+reach;x++)for(const segment of cells.get(key(x,y))||[]){if(seen.has(segment))continue;seen.add(segment);const required=Math.max(0,clearGap+(width+segment.width)/2);if(required>0&&pointSegmentDistance(p,segment.a,segment.b)+1e-6<required)return false;}return true;};
  return {addPath,clear};
}

async function calculate(msg, generation) {
  const [vl, vb, vr, vt] = msg.view;
  const dx = (cfg.bounds[2] - cfg.bounds[0]) / (cfg.width - 1);
  const dy = (cfg.bounds[3] - cfg.bounds[1]) / (cfg.height - 1);
  const c0 = Math.max(0, Math.floor((vl - cfg.bounds[0]) / dx));
  const c1 = Math.min(cfg.width - 1, Math.ceil((vr - cfg.bounds[0]) / dx));
  const r0 = Math.max(0, Math.floor((cfg.bounds[3] - vt) / dy));
  const r1 = Math.min(cfg.height - 1, Math.ceil((cfg.bounds[3] - vb) / dy));
  const interval=msg.interval,bathyInterval=msg.bathyInterval||interval;
  const baseInterval=msg.baseInterval||interval*5;
  const bathyBaseInterval=msg.bathyBaseInterval||baseInterval;
  const landRange=Math.max(0,cfg.maxElevation-Math.max(0,cfg.minElevation)),waterRange=Math.max(0,Math.min(0,cfg.maxElevation)-cfg.minElevation);
  const levels=Math.max(1,Math.ceil(landRange/interval)+Math.ceil(landRange/baseInterval)+Math.ceil(waterRange/bathyInterval)+Math.ceil(waterRange/bathyBaseInterval));
  const target = 7500000;
  const stride = msg.samplingStride||Math.max(1, Math.ceil(Math.sqrt(Math.max(1, ((c1-c0)*(r1-r0)*levels) / target))));
  const bands=cfg.colorBands||[{max:20},{max:35},{max:50},{max:65},{max:80},{max:null}];
  const minor = null, major = null, zero=[], minorBands=Array.from({length:bands.length},()=>[]), minorLevelBands=Array.from({length:bands.length},()=>new Map()), majorBands=Array.from({length:bands.length},()=>[]);
  const add = (arr, x1,y1,x2,y2) => arr.push(x1,y1,x2,y2);
  const dynamic=dynamicBanding(msg,bands,baseInterval,bathyBaseInterval);
  const bandFor = dynamic?.bandFor||function(level){const i=bands.findIndex(x=>x.max===null||level<x.max);return i<0?bands.length-1:i;};
  const segments = (tl,tr,br,bl,level,xL,xR,yTop,yBottom) => {
    let mask=0;
    if (tl>=level) mask|=1; if (tr>=level) mask|=2; if (br>=level) mask|=4; if (bl>=level) mask|=8;
    if (mask===0 || mask===15) return [];
    const p={
      t:[interp(xL,xR,tl,tr,level),yTop],
      r:[xR,interp(yTop,yBottom,tr,br,level)],
      b:[interp(xL,xR,bl,br,level),yBottom],
      l:[xL,interp(yTop,yBottom,tl,bl,level)]
    };
    const pairs={1:[['l','t']],2:[['t','r']],3:[['l','r']],4:[['r','b']],6:[['t','b']],7:[['l','b']],8:[['b','l']],9:[['b','t']],11:[['b','r']],12:[['r','l']],13:[['r','t']],14:[['t','l']]};
    let found=pairs[mask];
    if (!found) {
      const center=(tl+tr+br+bl)/4;
      if (mask===5) found=center>=level?[["t","r"],["b","l"]]:[["l","t"],["r","b"]];
      else found=center>=level?[["l","t"],["r","b"]]:[["t","r"],["b","l"]];
    }
    return found.map(([a,b])=>[p[a][0],p[a][1],p[b][0],p[b][1]]);
  };
  for (let r = r0; r + stride <= r1; r += stride) {
    if (((r-r0)/stride)%12===0) {
      await yieldToMessages();
      if (generation !== contourGeneration) return;
    }
    const yTop = cfg.bounds[3] - r * dy;
    const yBottom = cfg.bounds[3] - (r + stride) * dy;
    for (let c = c0; c + stride <= c1; c += stride) {
      const tl=value(c,r), tr=value(c+stride,r), br=value(c+stride,r+stride), bl=value(c,r+stride);
      const lo=Math.min(tl,tr,br,bl), hi=Math.max(tl,tr,br,bl);
      const xL=cfg.bounds[0]+c*dx, xR=cfg.bounds[0]+(c+stride)*dx;
      const fineBounds=msg.fineBounds,fineHere=!fineBounds||(xL+xR)/2>=fineBounds[0]&&(xL+xR)/2<=fineBounds[2]&&(yTop+yBottom)/2>=fineBounds[1]&&(yTop+yBottom)/2<=fineBounds[3];
      const emitLevels=(step,arr,bands,filter) => {
        for(let level=Math.ceil(lo/step)*step;level<=hi;level+=step){
          if(filter && !filter(level)) continue;
          for(const s of segments(tl,tr,br,bl,level,xL,xR,yTop,yBottom)){if(cfg.zeroBoundary&&Math.abs(level)<1e-6)add(zero,...s);else{if(arr)add(arr,...s);const band=bandFor(level);if(bands===minorBands){const key=level.toFixed(6),byLevel=minorLevelBands[band];if(!byLevel.has(key))byLevel.set(key,{level,data:[]});add(byLevel.get(key).data,...s);}else add(bands[band],...s);}}
        }
      };
      if(cfg.zeroBoundary&&lo<=0&&hi>=0)for(const s of segments(tl,tr,br,bl,0,xL,xR,yTop,yBottom))add(zero,...s);
      if(msg.adaptive&&cfg.zeroBoundary){
        emitLevels(baseInterval,major,majorBands,level=>level>0);
        emitLevels(bathyBaseInterval,major,majorBands,level=>level<0);
        if(fineHere)emitLevels(interval,minor,minorBands,level=>level>0&&level<=msg.detailCeiling&&Math.abs(level/baseInterval-Math.round(level/baseInterval))>1e-6);
        if(fineHere)emitLevels(bathyInterval,minor,minorBands,level=>level<0&&level>=-(msg.bathyDetailDepth??Infinity)&&Math.abs(level/bathyBaseInterval-Math.round(level/bathyBaseInterval))>1e-6);
      }else if(msg.adaptive){
        emitLevels(baseInterval,major,majorBands);
        if(fineHere)emitLevels(interval,minor,minorBands,level=>level<=msg.detailCeiling&&Math.abs(level/baseInterval-Math.round(level/baseInterval))>1e-6);
      }else if(cfg.zeroBoundary){
        if(fineHere)emitLevels(interval,minor,minorBands,level=>level>0&&Math.abs(level/baseInterval-Math.round(level/baseInterval))>1e-6);
        emitLevels(baseInterval,major,majorBands,level=>level>0);
        if(fineHere)emitLevels(bathyInterval,minor,minorBands,level=>level<0&&level>=-(msg.bathyDetailDepth??Infinity)&&Math.abs(level/bathyBaseInterval-Math.round(level/bathyBaseInterval))>1e-6);
        emitLevels(bathyBaseInterval,major,majorBands,level=>level<0);
      }else{
        if(fineHere)emitLevels(interval,minor,minorBands,level=>Math.round(level/interval)%5!==0);
        emitLevels(baseInterval,major,majorBands);
      }
    }
  }
  const pathLength=path=>{let n=0;for(let i=1;i<path.length;i++)n+=Math.hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]);return n;};
  const densityFilter=!!msg.indexGapFilter&&msg.adaptive,fastDensity=!!msg.fastPreview;
  const splitPrintable=(path,printable,density)=>{
    const minimum=density?(msg.indexGapMinimumGroundLength||0):(msg.minPathGroundLength||0);
    if(density&&path.length>2){
      const distance=new Float64Array(path.length),goodDistance=new Float64Array(path.length);
      for(let i=1;i<path.length;i++){
        const length=Math.hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]);
        distance[i]=distance[i-1]+length;goodDistance[i]=goodDistance[i-1]+(printable[i-1]&&printable[i]?length:0);
      }
      const halfWindow=(msg.indexGapWindowGroundLength||0)/2,averaged=new Array(path.length);let left=0,right=0;
      for(let i=0;i<path.length;i++){
        while(left<i&&distance[left]<distance[i]-halfWindow)left++;
        while(right+1<path.length&&distance[right+1]<=distance[i]+halfWindow)right++;
        const span=distance[right]-distance[left],usable=goodDistance[right]-goodDistance[left];averaged[i]=span>0&&usable/span>=.5;
      }
      printable=averaged;
    }
    for(let i=0;i<printable.length;){
      if(printable[i]){i++;continue;}
      const start=i;while(i<printable.length&&!printable[i])i++;const end=i-1;
      let run=0;for(let j=Math.max(1,start);j<=Math.min(path.length-1,end+1);j++)run+=Math.hypot(path[j][0]-path[j-1][0],path[j][1]-path[j-1][1]);
      const bridge=density?msg.indexGapBridgeGroundLength:msg.bridgeGroundLength;
      if(run<=(bridge||0))for(let j=start;j<=end;j++)printable[j]=true;
    }
    const parts=[];let part=[];
    const finish=()=>{if(part.length>1&&pathLength(part)>=minimum)parts.push(part);part=[];};
    for(let i=0;i<path.length;i++){
      if(printable[i]){if(!part.length&&i>0)part.push(path[i-1]);part.push(path[i]);}
      else if(part.length){part.push(path[i]);finish();}
    }
    finish();return parts;
  };
  const processFinePath=(path)=>{
    const minimum=msg.minPathGroundLength||0;
    if(msg.preserveContours)return path.length>1?[path]:[];
    if(!msg.adaptive)return pathLength(path)>=minimum?[path]:[];
    const printable=path.map(([x,y])=>{
      const col=Math.max(2,Math.min(cfg.width-3,Math.round((x-cfg.bounds[0])/dx)));
      const row=Math.max(2,Math.min(cfg.height-3,Math.round((cfg.bounds[3]-y)/dy)));
      const gx=(value(col+2,row)-value(col-2,row))/(4*dx),gy=(value(col,row+2)-value(col,row-2))/(4*dy);
      const localFine=value(col,row)<0?bathyInterval:interval,slope=Math.hypot(gx,gy),spacing=slope<1e-6?Infinity:localFine/slope;
      return spacing>=msg.minGroundSpacing;
    });
    return splitPrintable(path,printable,false);
  };
  const merge=arrays=>{const length=arrays.reduce((n,a)=>n+a.length,0),out=new Float32Array(length);let offset=0;for(const a of arrays){out.set(a,offset);offset+=a.length;}return out;};
  const roundData=data=>packPaths(dataPaths(stitch(data)).map(path=>roundedPath(path,msg.smoothing)));
  const a=new Float32Array(), b=new Float32Array(), z=roundData(zero),xb=[];
  for(const band of majorBands){
    xb.push(roundData(band));
    await yieldToMessages();
    if(generation!==contourGeneration)return;
  }
  let mb;
  if(densityFilter){
    const fineWidth=msg.fineLineGroundWidth||0,indexWidth=msg.indexLineGroundWidth||fineWidth,clearGap=msg.indexGapGroundSpacing||0,groundPerMm=msg.groundPerPaperMm||Math.max(1,fineWidth/.3),maximum=Math.max(fineWidth,indexWidth),cellSize=Math.max(groundPerMm*.1,Math.max(0,clearGap+(fineWidth+maximum)/2),1),occupied=spacingIndex(cellSize,clearGap,maximum);
    for(const path of dataPaths(z))occupied.addPath(path,indexWidth);for(const band of xb)for(const path of dataPaths(band))occupied.addPath(path,indexWidth);
    const entries=[];for(let band=0;band<minorLevelBands.length;band++)for(const {level,data} of minorLevelBands[band].values()){const localFine=level<0?bathyInterval:interval,localIndex=level<0?bathyBaseInterval:baseInterval,ratio=Math.max(1,Math.round(localIndex/localFine)),ordinal=Math.round(level/localFine),position=((ordinal%ratio)+ratio)%ratio;let priority=0,n=position;while(n>0&&n%2===0){priority++;n/=2;}entries.push({band,level,data,priority});}
    entries.sort((u,v)=>v.priority-u.priority||Math.abs(u.level)-Math.abs(v.level));const accepted=Array.from({length:bands.length},()=>[]);
    for(let entryIndex=0;entryIndex<entries.length;entryIndex++){
      if(entryIndex%6===0){await yieldToMessages();if(generation!==contourGeneration)return;}
      const entry=entries[entryIndex];
      for(const raw of dataPaths(stitch(entry.data))){
      const path=roundedPath(raw,msg.smoothing),minimum=msg.indexGapMinimumGroundLength||0,total=pathLength(path);if(total<minimum)continue;
      let clearLength=0,sampledLength=0;
      for(let i=1;i<path.length;i++){
        const a=path[i-1],b=path[i],length=Math.hypot(b[0]-a[0],b[1]-a[1]);if(!length||fastDensity&&i%3!==1)continue;
        const samples=fastDensity?[.5]:[.2,.5,.8];let clearSamples=0;sampledLength+=length;
        for(const f of samples)if(occupied.clear([a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f],fineWidth))clearSamples++;
        clearLength+=length*clearSamples/samples.length;
      }
      // Density control selects complete contour strokes. A retained marching-squares
      // contour is never cut into internal fragments; the crop boundary is its only cap.
      if(sampledLength&&clearLength/sampledLength>=.5){accepted[entry.band].push(path);if(!fastDensity)occupied.addPath(path,fineWidth);}
      }
    }
    mb=accepted.map(packPaths);
  }else mb=minorLevelBands.map(byLevel=>merge([...byLevel.values()].map(({data})=>packPaths(dataPaths(stitch(data)).flatMap(path=>processFinePath(roundedPath(path,msg.smoothing)))))));
  const transfer=[a.buffer,b.buffer,z.buffer,...mb.map(x=>x.buffer),...xb.map(x=>x.buffer)];
  if(generation!==contourGeneration)return;
  self.postMessage({type:'result', id:msg.id, minorPaths:a, majorPaths:b, zeroPaths:z, minorBandPaths:mb, majorBandPaths:xb, bandLabels:dynamic?.labels||bands.map(x=>x.label), colorStats:dynamic?{landLevels:dynamic.landLevels,waterLevels:dynamic.waterLevels}:null, stride},transfer);
}
