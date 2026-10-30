import {useEffect,useState} from 'react';
import {Crosshair,FileCode,Move3D,Plus,X} from 'lucide-react';
import type {EditableSceneObject} from '../Viewport';
import type {SceneStudio} from './useSceneStudio';

export default function ObjectInspector({studio:s,onAsk}:{studio:SceneStudio;onAsk:()=>void}){
  const [adding,setAdding]=useState(false);
  return <div className="agent-inspector">
    <div className="agent-object-list"><header><span>Scene objects <small>{s.objects.length}</small></span><button className={'agent-icon '+(adding?'active':'')} aria-label="Add component" disabled={!s.ready||s.blocked} onClick={()=>setAdding(!adding)}><Plus size={14}/></button></header>
      {adding?<div className="agent-component-list"><label className="agent-mjcf-import" title="Bodies, joints, primitive and mesh geoms, sites, defaults, equality constraints, tendons, and motor/position/velocity actuators. Select the XML together with the OBJ/STL files it names. Materials, textures, and plugins are rejected."><FileCode size={13}/><span>Import MuJoCo XML…<small>Articulated object from MJCF, with any OBJ/STL meshes it names</small></span><input type="file" multiple accept=".xml,.mjcf,.obj,.stl,text/xml" hidden disabled={s.blocked} onChange={e=>{const files=[...(e.target.files??[])];e.target.value='';if(files.length){s.importMjcf(files);setAdding(false);}}}/></label>{s.components.map(item=><button key={item.id} disabled={s.blocked||item.available===false} title={typeof item.constraint==='string'?item.constraint:undefined} onClick={()=>{s.insertComponent(item.id);setAdding(false);}}><Plus size={13}/><span>{item.label??item.name??item.id}<small>{typeof item.constraint==='string'?item.constraint:item.description}</small></span></button>)}{!s.components.length&&<p>No component presets available.</p>}</div>:<div className="agent-object-scroll">{s.objects.map(item=><button key={item.id} aria-pressed={item.id===s.selectedId} onClick={()=>{s.setSelectedId(item.id);s.startEditing();}}><Crosshair size={13}/><span>{item.label}</span><small>{item.kind.replaceAll('_',' ')}</small></button>)}{!s.objects.length&&<p>{s.recordingLoading?'Loading this revision…':'Objects appear after the scene is built.'}</p>}</div>}
    </div>
    {s.selected?<ObjectProperties key={s.selected.id} object={s.selected} busy={s.blocked} onClose={()=>s.setSelectedId(null)} onApply={changes=>s.editObject(s.selected!.id,changes)} onAsk={onAsk}/>:<div className="agent-inspector-empty"><Move3D size={20}/><strong>Select something to edit</strong><p>Click the scene or choose an object. Drag its arrows, enter a position, or ask the agent.</p></div>}
  </div>;
}
function ObjectProperties({object,busy,onClose,onApply,onAsk}:{object:EditableSceneObject;busy:boolean;onClose:()=>void;onApply:(changes:{position?:number[];size?:number[]})=>void;onAsk:()=>void}){
  const [position,setPosition]=useState(object.position.map(String));
  const [size,setSize]=useState(object.size.map(value=>String(value*2)));
  const [sizeDirty,setSizeDirty]=useState(false);const [positionDirty,setPositionDirty]=useState(false);
  useEffect(()=>{setPosition(object.position.map(String));setSize(object.size.map(value=>String(value*2)));setSizeDirty(false);setPositionDirty(false);},[object]);
  const valid=position.every(value=>value.trim()!==''&&Number.isFinite(Number(value)))&&size.every(value=>value.trim()!==''&&Number(value)>0);
  const changed=position.some((value,index)=>Math.abs(Number(value)-object.position[index])>1e-6)||size.some((value,index)=>Math.abs(Number(value)/2-object.size[index])>1e-6);
  return <div className="agent-object-properties"><header><span>{object.label}</span><button className="agent-icon" aria-label="Deselect object" onClick={onClose}><X size={14}/></button></header>
    <span className="agent-property-label">Position <small>metres</small></span><div className="agent-xyz">{['X','Y','Z'].map((axis,index)=><label key={axis}><span>{axis}</span><input aria-label={object.label+' position '+axis} type="number" step="0.01" value={position[index]} disabled={busy||!object.editable||!(object.position_axes??[0,1,2]).includes(index)} onChange={e=>{setPositionDirty(true);setPosition(previous=>previous.map((v,i)=>i===index?e.target.value:v));}}/></label>)}</div>
    {object.dimensions_editable&&<><span className="agent-property-label">Dimensions <small>metres · full size</small></span><div className="agent-xyz">{['X','Y','Z'].map((axis,index)=><label key={axis}><span>{axis}</span><input aria-label={object.label+' size '+axis} type="number" min="0.01" step="0.01" value={size[index]} disabled={busy||!object.editable} onChange={e=>{setSizeDirty(true);setSize(previous=>previous.map((v,i)=>i===index?e.target.value:v));}}/></label>)}</div></>}
    {!!object.constraints?.length&&<p className="agent-object-constraint">{object.constraints.join(' ')}</p>}
    <div className="agent-property-actions"><button disabled={busy} onClick={onAsk}>Ask agent about this</button><button className="agent-primary" disabled={busy||!object.editable||!valid||!changed} onClick={()=>onApply({...(positionDirty?{position:position.map(Number)}:{}),...(object.dimensions_editable&&sizeDirty?{size:size.map(v=>Number(v)/2)}:{})})}>Apply & simulate</button></div>
  </div>;
}
