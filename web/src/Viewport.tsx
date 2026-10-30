import {useEffect, useId, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {Maximize2, Scan, RotateCcw, Move, LockKeyhole} from 'lucide-react';
import type {Frame} from './types';
import './viewport-editing.css';

const ROBOT_TAGS:Record<string,string>={'franka-panda-v1':'FRANKA PANDA','cartesian-gripper-v1':'CARTESIAN GRIPPER','ur5e-2f85-v1':'UR5E + ROBOTIQ 2F-85','ur5e-allegro-v1':'UR5E + ALLEGRO HAND','tidybot-v1':'TIDYBOT MOBILE'};

let pandaVisuals:Promise<THREE.Group>|undefined;
function loadPandaVisuals(){
  if(!pandaVisuals)pandaVisuals=fetch('/api/robot-visuals/franka-panda-v1.glb').then(async response=>{
    if(!response.ok)throw new Error('Optional Panda CAD is unavailable.');
    return (await new GLTFLoader().parseAsync(await response.arrayBuffer(),'')).scene;
  }).catch(error=>{pandaVisuals=undefined;throw error;});
  return pandaVisuals;
}

const armVisuals=new Map<string,Promise<THREE.Group>>();
function loadArmVisuals(embodiment:string){
  let pending=armVisuals.get(embodiment);
  if(!pending){
    pending=fetch('/api/robot-visuals/arm/'+encodeURIComponent(embodiment)+'.glb').then(async response=>{
      if(!response.ok)throw new Error('Optional robot CAD is unavailable.');
      return (await new GLTFLoader().parseAsync(await response.arrayBuffer(),'')).scene;
    }).catch(error=>{armVisuals.delete(embodiment);throw error;});
    armVisuals.set(embodiment,pending);
  }
  return pending;
}

export type EditableSceneObject = {
  id:string;
  label:string;
  kind:string;
  position:number[];
  size:number[];
  editable:boolean;
  geom_names:string[];
  color?:number[];
  position_axes?:number[];
  dimensions_editable?:boolean;
  constraints?:string[];
};

type ViewportProps = {
  frame?:Frame;
  compact?:boolean;
  objects?:EditableSceneObject[];
  selectedObjectId?:string|null;
  onSelectObject?:(id:string|null)=>void;
  onMoveObject?:(id:string,position:number[])=>void;
  editing?:boolean;
  disabled?:boolean;
};

export default function Viewport({frame,compact=false,objects=[],selectedObjectId=null,onSelectObject,onMoveObject,editing=false,disabled=false}:ViewportProps) {
  const host=useRef<HTMLDivElement>(null);
  const update=useRef<(frame?:Frame)=>void>(()=>{});
  const syncEditing=useRef<()=>void>(()=>{});
  const reset=useRef<(top:boolean)=>void>(()=>{});
  const latest=useRef({objects,selectedObjectId,onSelectObject,onMoveObject,editing,disabled});
  latest.current={objects,selectedObjectId,onSelectObject,onMoveObject,editing,disabled};
  const [failure,setFailure]=useState('');
  const [dragging,setDragging]=useState(false);
  const instructionsId=useId();
  const selected=objects.find(object=>object.id===selectedObjectId);

  useEffect(()=>{
    const el=host.current!;
    let renderer:THREE.WebGLRenderer;
    try {renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});} catch {setFailure('WebGL is unavailable. Recorded states and judge evidence are still accessible.');return;}
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.shadowMap.enabled=true;
    renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    const background=new THREE.Color(getComputedStyle(el).getPropertyValue('--viewport-background').trim()||'#131312');
    renderer.setClearColor(background);
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.domElement.tabIndex=0;
    el.appendChild(renderer.domElement);
    const scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(39,1,.01,20);
    camera.up.set(0,0,1);
    camera.position.set(1.2,-1.5,1.05);
    const controls=new OrbitControls(camera,renderer.domElement);
    controls.target.set(0,0,.15);controls.enableDamping=true;controls.minDistance=.6;controls.maxDistance=3.8;controls.maxPolarAngle=Math.PI*.49;
    scene.fog=new THREE.Fog(background,3.2,6.5);
    // Designs restyle the viewport backdrop; follow theme/design switches without remounting.
    const restyle=new MutationObserver(()=>{const next=getComputedStyle(el).getPropertyValue('--viewport-background').trim();if(!next)return;background.set(next);renderer.setClearColor(background);(scene.fog as THREE.Fog).color.copy(background);});
    restyle.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme','data-design']});
    scene.add(new THREE.HemisphereLight(0xf4efe6,0x2c2a26,2.2));
    const rim=new THREE.DirectionalLight(0xffb487,.55);rim.position.set(.9,1.3,.7);scene.add(rim);
    const light=new THREE.DirectionalLight(0xfff1e0,3);light.position.set(-.5,-.8,2);light.castShadow=true;
    light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-1;light.shadow.camera.right=1;light.shadow.camera.top=1;light.shadow.camera.bottom=-1;light.shadow.normalBias=.004;scene.add(light);
    const grid=new THREE.GridHelper(2.8,40,0x57544d,0x2a2926);grid.rotation.x=Math.PI/2;grid.position.z=-.073;scene.add(grid);
    const meshes=new Map<string,THREE.Mesh>();
    const signatures=new Map<string,string>();
    let disposed=false;
    let requestedPanda=false;
    let pendingPanda:THREE.Group|undefined;
    let robotRoot:THREE.Group|undefined;
    let lastFrame:Frame|undefined;
    const robotNodes:{node:THREE.Object3D;relative:THREE.Matrix4;anchorOffset:number}[]=[];
    const robotPickMeshes:THREE.Mesh[]=[];
    const installPanda=()=>{
      if(!pendingPanda||robotRoot)return;
      robotRoot=pendingPanda.clone(true);pendingPanda=undefined;
      robotRoot.traverse(object=>{
        if(object instanceof THREE.Mesh){
          object.geometry=object.geometry.clone();
          object.material=Array.isArray(object.material)?object.material.map(material=>material.clone()):object.material.clone();
          object.castShadow=true;object.receiveShadow=true;robotPickMeshes.push(object);
        }
      });
      for(const node of robotRoot.children){
        node.updateMatrix();
        robotNodes.push({node,relative:node.matrix.clone(),anchorOffset:Number(node.userData.anchor_offset)});
        node.matrixAutoUpdate=false;
      }
      scene.add(robotRoot);
    };
    // Registry arms: CAD meshes placed on recorded link poses (frame.robot_bodies).
    let requestedArm='';
    let armRoot:THREE.Group|undefined;
    const armNodes:{node:THREE.Object3D;relative:THREE.Matrix4;body:string}[]=[];
    const installArm=(group:THREE.Group)=>{
      if(armRoot){scene.remove(armRoot);armNodes.length=0;}
      armRoot=group.clone(true);
      armRoot.traverse(object=>{if(object instanceof THREE.Mesh){object.castShadow=true;object.receiveShadow=true;robotPickMeshes.push(object);}});
      for(const node of armRoot.children){node.updateMatrix();armNodes.push({node,relative:node.matrix.clone(),body:String(node.userData.body)});node.matrixAutoUpdate=false;}
      scene.add(armRoot);
    };
    const linkPose=new THREE.Matrix4(),linkQuat=new THREE.Quaternion(),linkPos=new THREE.Vector3(),unitScale=new THREE.Vector3(1,1,1);
    const syncArm=(next?:Frame)=>{
      const hidden=new Set<string>();
      const poses=(next as (Frame&{robot_bodies?:Record<string,number[]>})|undefined)?.robot_bodies;
      if(!armRoot)return hidden;
      const metadata=armRoot.userData;
      const matches=!!poses&&next?.embodiment===metadata.embodiment&&armNodes.every(item=>poses[item.body]);
      armRoot.visible=matches;
      if(!matches||!poses)return hidden;
      for(const {node,relative,body} of armNodes){
        const [x,y,z,qw,qx,qy,qz]=poses[body];
        linkPose.compose(linkPos.set(x,y,z),linkQuat.set(qx,qy,qz,qw),unitScale);
        node.matrix.multiplyMatrices(linkPose,relative);node.matrixWorldNeedsUpdate=true;
      }
      const prefix=String(metadata.geom_prefix||'');
      if(prefix)next?.geoms.forEach(geom=>{if(geom.name.startsWith(prefix))hidden.add(geom.name);});
      return hidden;
    };
    const robotAnchorMatrix=new THREE.Matrix4();
    const syncRobot=(next?:Frame)=>{
      const hidden=new Set<string>();
      if(!robotRoot)return hidden;
      const metadata=robotRoot.userData;
      const robotFrame=next as (Frame&{robot_asset_commit?:string})|undefined;
      const anchors=next?.geoms.filter(geom=>/^panda_geom_\d+$/.test(geom.name)).sort((a,b)=>Number(a.name.slice(11))-Number(b.name.slice(11)))??[];
      const first=anchors.length?Number(anchors[0].name.slice(11)):0;
      const offsets=metadata.collision_offsets as number[]|undefined;
      const matches=next?.embodiment==='franka-panda-v1'&&robotFrame?.robot_asset_commit===metadata.robot_asset_commit&&offsets?.length===anchors.length&&anchors.every((geom,index)=>Number(geom.name.slice(11))-first===offsets[index]);
      robotRoot.visible=!!matches;
      if(!matches)return hidden;
      const byName=new Map(anchors.map(geom=>[geom.name,geom]));
      for(const {node,relative,anchorOffset} of robotNodes){
        const geom=byName.get('panda_geom_'+(first+anchorOffset));
        if(!geom){robotRoot.visible=false;return hidden;}
        const m=geom.matrix,p=geom.position;
        robotAnchorMatrix.set(m[0],m[1],m[2],p[0],m[3],m[4],m[5],p[1],m[6],m[7],m[8],p[2],0,0,0,1);
        node.matrix.multiplyMatrices(robotAnchorMatrix,relative);node.matrixWorldNeedsUpdate=true;
      }
      anchors.forEach(geom=>hidden.add(geom.name));
      return hidden;
    };
    const disposeMesh=(mesh:THREE.Mesh)=>{
      scene.remove(mesh);
      mesh.traverse(obj=>{if(obj instanceof THREE.Mesh||obj instanceof THREE.LineSegments){obj.geometry.dispose();const materials=Array.isArray(obj.material)?obj.material:[obj.material];materials.forEach(m=>m.dispose());}});
    };
    const matrix=new THREE.Matrix4();

    // The gizmo moves an anchor. Mapped geoms retain their relative offsets, so a
    // multipart assembly and its collider outlines move as one scene object.
    const anchor=new THREE.Object3D();scene.add(anchor);
    const transform=new TransformControls(camera,renderer.domElement);
    transform.setMode('translate');transform.setSpace('world');transform.setSize(.8);
    const transformHelper=transform.getHelper();scene.add(transformHelper);
    const selectionBox=new THREE.Box3();
    const outline=new THREE.Box3Helper(selectionBox,0xa7bfff);
    const outlineMaterials=Array.isArray(outline.material)?outline.material:[outline.material];
    outlineMaterials.forEach(material=>{material.depthTest=false;material.transparent=true;material.opacity=.95;});
    outline.renderOrder=1000;outline.visible=false;scene.add(outline);
    type Drag={id:string;start:THREE.Vector3;positions:Map<string,THREE.Vector3>;axes:number[]};
    let drag:Drag|null=null;
    let pointerStart:{x:number;y:number;transformed:boolean}|null=null;

    const selectedObject=()=>latest.current.objects.find(object=>object.id===latest.current.selectedObjectId);
    const updateOutline=()=>{
      const object=selectedObject();
      selectionBox.makeEmpty();
      if(latest.current.editing&&object){
        for(const name of object.geom_names){const mesh=meshes.get(name);if(mesh){mesh.updateWorldMatrix(true,false);selectionBox.expandByObject(mesh);}}
      }
      outline.visible=!selectionBox.isEmpty();
      if(outline.visible){selectionBox.expandByScalar(.003);outline.updateMatrixWorld(true);}
    };
    const restoreDrag=()=>{
      if(!drag)return;
      for(const [name,position] of drag.positions)meshes.get(name)?.position.copy(position);
      anchor.position.copy(drag.start);
      drag=null;setDragging(false);updateOutline();
    };
    const cancelDrag=()=>{
      restoreDrag();
      // Clear the library's gesture state as well, so Escape cannot commit on
      // the subsequent physical pointer-up.
      if(transform.dragging)transform.pointerUp(null);
      controls.enabled=true;
    };
    const syncSelection=()=>{
      const config=latest.current;
      const object=selectedObject();
      renderer.domElement.setAttribute('aria-label',config.editing?'3D scene editor. Select an object and drag its colored axes to move it. Press Escape to cancel a move.':'Interactive 3D replay of recorded MuJoCo geometry');
      if(drag)return;
      const axes=object?.position_axes??[0,1,2];
      transform.showX=axes.includes(0);transform.showY=axes.includes(1);transform.showZ=axes.includes(2);
      const canMove=config.editing&&!config.disabled&&object?.editable&&axes.length>0&&!!config.onMoveObject&&object.geom_names.some(name=>meshes.has(name));
      transform.enabled=!!canMove;
      if(canMove&&object){anchor.position.fromArray(object.position);anchor.updateMatrixWorld(true);transform.attach(anchor);}
      else transform.detach();
      renderer.domElement.style.cursor=config.editing&&!config.disabled?'crosshair':'grab';
      updateOutline();
    };
    syncEditing.current=()=>{cancelDrag();syncSelection();};

    transform.addEventListener('dragging-changed',event=>{controls.enabled=!event.value;});
    transform.addEventListener('mouseDown',()=>{
      const config=latest.current;
      const object=selectedObject();
      if(!object?.editable||!config.editing||config.disabled)return;
      drag={id:object.id,start:new THREE.Vector3().fromArray(object.position),positions:new Map(),axes:object.position_axes??[0,1,2]};
      for(const name of object.geom_names){const mesh=meshes.get(name);if(mesh)drag.positions.set(name,mesh.position.clone());}
      if(pointerStart)pointerStart.transformed=true;
      setDragging(true);
    });
    transform.addEventListener('objectChange',()=>{
      if(!drag)return;
      // Axis permissions also apply to the gizmo's center/plane handles.
      for(let axis=0;axis<3;axis++)if(!drag.axes.includes(axis))anchor.position.setComponent(axis,drag.start.getComponent(axis));
      const delta=anchor.position.clone().sub(drag.start);
      for(const [name,position] of drag.positions)meshes.get(name)?.position.copy(position).add(delta);
      updateOutline();
    });
    transform.addEventListener('mouseUp',()=>{
      if(!drag)return;
      const {id,start}=drag;
      const position=anchor.position.toArray();
      const changed=anchor.position.distanceToSquared(start)>1e-10;
      const config=latest.current;
      // Only validated frame data is persistent. A failed or cancelled backend
      // update cannot leave the viewport displaying an unsaved position.
      restoreDrag();
      if(changed&&position.every(Number.isFinite)&&config.editing&&!config.disabled&&config.selectedObjectId===id)config.onMoveObject?.(id,position);
    });

    const raycaster=new THREE.Raycaster();
    const pointer=new THREE.Vector2();
    const pointerDown=(event:PointerEvent)=>{
      if(event.button!==0)return;
      renderer.domElement.focus({preventScroll:true});
      pointerStart={x:event.clientX,y:event.clientY,transformed:transform.dragging};
    };
    const pointerUp=(event:PointerEvent)=>{
      const start=pointerStart;pointerStart=null;
      const config=latest.current;
      if(event.button!==0||!start||start.transformed||!config.editing||config.disabled||Math.hypot(event.clientX-start.x,event.clientY-start.y)>5)return;
      const bounds=renderer.domElement.getBoundingClientRect();
      if(!bounds.width||!bounds.height)return;
      pointer.set((event.clientX-bounds.left)/bounds.width*2-1,-(event.clientY-bounds.top)/bounds.height*2+1);
      raycaster.setFromCamera(pointer,camera);
      const hit=raycaster.intersectObjects([...meshes.values(),...(robotRoot?.visible?robotPickMeshes:[])],false)[0];
      const hitName=hit&&[...meshes].find(([,mesh])=>mesh===hit.object)?.[0];
      config.onSelectObject?.(hitName?config.objects.find(object=>object.geom_names.includes(hitName))?.id??null:null);
    };
    const pointerCancel=()=>{pointerStart=null;cancelDrag();syncSelection();};
    const keyDown=(event:KeyboardEvent)=>{
      if(event.key==='Escape'&&latest.current.editing){event.preventDefault();if(drag){cancelDrag();syncSelection();}else latest.current.onSelectObject?.(null);}
    };
    renderer.domElement.addEventListener('pointerdown',pointerDown);
    renderer.domElement.addEventListener('pointerup',pointerUp);
    renderer.domElement.addEventListener('pointercancel',pointerCancel);
    renderer.domElement.addEventListener('keydown',keyDown);
    window.addEventListener('blur',pointerCancel);

    update.current=(next)=>{
      cancelDrag();
      lastFrame=next;
      if(next?.embodiment==='franka-panda-v1'&&!requestedPanda){
        requestedPanda=true;
        loadPandaVisuals().then(group=>{if(!disposed)pendingPanda=group;}).catch(()=>{});
      }
      const armFrame=next as (Frame&{robot_bodies?:Record<string,number[]>})|undefined;
      if(armFrame?.robot_bodies&&armFrame.embodiment&&requestedArm!==armFrame.embodiment){
        const wanted=armFrame.embodiment;requestedArm=wanted;
        loadArmVisuals(wanted).then(group=>{if(!disposed&&requestedArm===wanted){installArm(group);update.current?.(lastFrame);}}).catch(()=>{});
      }
      const hiddenRobotGeoms=new Set([...syncRobot(next),...syncArm(next)]);
      const visible=new Set(next?.geoms.filter(geom=>geom.color[3]>0&&!hiddenRobotGeoms.has(geom.name)).map(geom=>geom.name)||[]);
      for(const [name,mesh] of meshes){if(!visible.has(name)){disposeMesh(mesh);meshes.delete(name);signatures.delete(name);}}
      if(next)for(const geom of next.geoms){
        if(geom.color[3]===0||hiddenRobotGeoms.has(geom.name))continue;
        let mesh=meshes.get(geom.name);
        const signature=JSON.stringify([geom.type,geom.size,geom.color,geom.mesh_vertices,geom.mesh_faces]);
        if(mesh&&signatures.get(geom.name)!==signature){disposeMesh(mesh);meshes.delete(geom.name);mesh=undefined;}
        if(!mesh){
          let shape:THREE.BufferGeometry;
          if(geom.type==='mesh'&&geom.mesh_vertices&&geom.mesh_faces){shape=new THREE.BufferGeometry();shape.setAttribute('position',new THREE.Float32BufferAttribute(geom.mesh_vertices.flat(),3));shape.setIndex(geom.mesh_faces.flat());shape.computeVertexNormals();}
          else if(geom.type==='sphere')shape=new THREE.SphereGeometry(geom.size[0],20,12);
          else if(geom.type==='capsule')shape=new THREE.CapsuleGeometry(geom.size[0],geom.size[1]*2,6,16).rotateX(Math.PI/2);
          else shape=geom.type==='cylinder'?new THREE.CylinderGeometry(geom.size[0],geom.size[0],geom.size[1]*2,24).rotateX(Math.PI/2):new THREE.BoxGeometry(...geom.size.map(n=>n*2) as [number,number,number]);
          const material=new THREE.MeshStandardMaterial({color:new THREE.Color(...geom.color.slice(0,3) as [number,number,number]),opacity:geom.color[3]??1,transparent:(geom.color[3]??1)<1,roughness:.65,metalness:geom.name.includes('rail')||geom.name==='shaft'?.6:.15});
          mesh=new THREE.Mesh(shape,material);mesh.castShadow=true;mesh.receiveShadow=true;meshes.set(geom.name,mesh);scene.add(mesh);
          signatures.set(geom.name,signature);
          if(geom.name==='block'||geom.name.startsWith('object_')){
            const edges=new THREE.LineSegments(new THREE.EdgesGeometry(shape),new THREE.LineBasicMaterial({color:0xffbe78,transparent:true,opacity:.6}));mesh.add(edges);
          }
        }
        mesh.position.fromArray(geom.position);
        const m=geom.matrix;
        matrix.set(m[0],m[1],m[2],0,m[3],m[4],m[5],0,m[6],m[7],m[8],0,0,0,0,1);
        mesh.quaternion.setFromRotationMatrix(matrix);
      }
      syncSelection();
    };
    reset.current=(top)=>{camera.position.set(...(top?[0,-.001,1.85]:[1.2,-1.5,1.05]) as [number,number,number]);controls.target.set(0,0,.15);controls.update();};
    const resize=new ResizeObserver(()=>{const {width,height}=el.getBoundingClientRect();if(width&&height){renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();}});resize.observe(el);
    let animation=0;
    const draw=()=>{
      animation=requestAnimationFrame(draw);
      // Let an active object move finish before replacing optional robot bounds.
      if(pendingPanda&&!drag){installPanda();update.current(lastFrame);}
      controls.update();renderer.render(scene,camera);
    };draw();
    update.current(frame);
    return ()=>{
      disposed=true;
      cancelAnimationFrame(animation);resize.disconnect();
      renderer.domElement.removeEventListener('pointerdown',pointerDown);
      renderer.domElement.removeEventListener('pointerup',pointerUp);
      renderer.domElement.removeEventListener('pointercancel',pointerCancel);
      renderer.domElement.removeEventListener('keydown',keyDown);
      window.removeEventListener('blur',pointerCancel);
      controls.dispose();scene.remove(transformHelper);transform.dispose();
      scene.traverse(obj=>{if(obj instanceof THREE.Mesh||obj instanceof THREE.LineSegments){obj.geometry.dispose();const materials=Array.isArray(obj.material)?obj.material:[obj.material];materials.forEach(m=>m.dispose());}});
      restyle.disconnect();
      renderer.dispose();renderer.domElement.remove();update.current=()=>{};syncEditing.current=()=>{};reset.current=()=>{};
    };
  },[]);
  useEffect(()=>update.current(frame),[frame]);
  useEffect(()=>syncEditing.current(),[objects,selectedObjectId,editing,disabled]);

  return <div className={'viewport '+(compact?'compact ':'')+(editing?'viewport-editing ':'')+(dragging?'viewport-dragging':'')}>
    <div ref={host} className="canvas-host" aria-label={editing?'Interactive 3D scene editor':'Interactive 3D replay of recorded MuJoCo geometry'} aria-describedby={editing?instructionsId:undefined}/>
    {failure&&<div className="viewport-error">{failure}</div>}
    <div className="viewport-top"><span className="sim-tag"><span className="dot"/> MUJOCO <span className="separator">/</span> {!frame?'NO RECORDING':ROBOT_TAGS[frame.embodiment??'']??(frame.embodiment?frame.embodiment.toUpperCase():'CARTESIAN GRIPPER')}</span><span className="view-tag">{editing?'Edit scene':'Perspective'}</span></div>
    {editing&&selected&&<div className="viewport-selection" aria-live="polite">{selected.editable?<Move size={13}/>:<LockKeyhole size={13}/>}<strong>{selected.label}</strong><span>{disabled?'Updating scene…':dragging?'Release to apply':selected.editable?'Drag an axis to move':'Fixed object'}</span></div>}
    <div className="viewport-bottom"><span><span className="axis x">X</span><span className="axis y">Y</span><span className="axis z">Z</span><span className="orbit-hint">Drag to orbit · Scroll to zoom</span></span><div className="view-buttons"><button title="Overhead view" aria-label="Overhead view" onClick={()=>reset.current(true)}><Scan size={16}/></button><button title="Reset view" aria-label="Reset view" onClick={()=>reset.current(false)}><RotateCcw size={16}/></button><button title="Fullscreen" aria-label="Fullscreen" onClick={()=>host.current?.parentElement?.requestFullscreen().catch(()=>{})}><Maximize2 size={16}/></button></div></div>
    {editing?<div className="viewport-edit-hint" id={instructionsId}>Select an object · Drag colored axes · Esc cancels</div>:(frame?.geoms.some(g=>g.name==='block')||frame?.geoms.some(g=>g.name==='tray_base'))&&<div className="scene-legend">{frame?.geoms.some(g=>g.name==='block')&&<span><i className="legend-block"/> Target object</span>}{frame?.geoms.some(g=>g.name==='tray_base')&&<span><i className="legend-tray"/> Placement zone</span>}</div>}
  </div>;
}
