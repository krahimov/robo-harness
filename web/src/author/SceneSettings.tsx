import {useRef} from 'react';
import {Box,Camera,FileUp,Plus,RefreshCw,Upload,X} from 'lucide-react';
import type {TaskSpec} from '../types';
import type {SceneStudio} from './useSceneStudio';

export default function SceneSettings({studio:s,onClose,onEditTask,onCamera}:{studio:SceneStudio;onClose:()=>void;onEditTask:(task:TaskSpec)=>void;onCamera:()=>void}){
  const meshInput=useRef<HTMLInputElement>(null);
  const {config,patchConfig,blocked}=s;
  return <section className="agent-settings" aria-label="Scene settings">
    <header><h2>Scene settings</h2><button className="agent-icon" aria-label="Close scene settings" onClick={onClose}><X size={16}/></button></header>
    <div className="agent-settings-body">
      <label>Scene name<input value={config.name} disabled={blocked} maxLength={100} onChange={e=>patchConfig({name:e.target.value})}/></label>
      <div className="agent-field-row">
        <label>Author connection<select value={config.provider} disabled={blocked} onChange={e=>patchConfig({provider:e.target.value as typeof config.provider})}><option value="codex">Codex subscription</option><option value="openai">OpenAI API</option></select></label>
        <label>Scene generation<select value={config.builder} disabled={blocked} onChange={e=>patchConfig({builder:e.target.value as typeof config.builder})}><option value="blender">Blender + MuJoCo</option><option value="mujoco">MuJoCo primitives</option></select></label>
      </div>
      <p>{s.connection?.message??'Checking author connection…'}</p>
      <p>{config.builder==='blender'?(s.author?.blender?.configured?`Blender ${s.author.blender.version} ready`:s.author?.blender?.message??'Checking Blender…'):'Geometry and physics are generated directly in MuJoCo.'}</p>
      <button className="agent-text" disabled={blocked} onClick={s.refreshConnection}><RefreshCw size={13}/>Check connections</button>
      <label>Robot<select value={config.embodiment} disabled={blocked} onChange={e=>patchConfig({embodiment:e.target.value as typeof config.embodiment})}><option value="franka-panda-v1">Franka Panda · finger contacts</option><option value="cartesian-gripper-v1">Cartesian gripper · idealized grasp</option></select></label>
      <label>Task requirements<select value={config.taskId} disabled={blocked} onChange={e=>{const task=s.tasks.find(t=>t.id===e.target.value);patchConfig({taskId:e.target.value,...(task?.suggested_builder?{builder:task.suggested_builder}:{}),...(task?.suggested_scene?.embodiment?{embodiment:task.suggested_scene.embodiment}:{})});}}><option value="">Original block-to-tray task</option>{s.tasks.map(task=><option key={task.id} value={task.id}>{task.name} · v{task.version}</option>)}</select></label>
      {s.task&&<details><summary>{s.task.requirements.length} human-owned outcome checks</summary><p>{s.task.description}</p><ul>{s.task.requirements.map(item=><li key={item.id}><strong>{item.label}</strong><p>{item.definition}</p></li>)}</ul><button className="agent-text" disabled={blocked} onClick={()=>onEditTask(s.task!)}>Edit as a new task version</button></details>}
      <div className="agent-settings-actions"><button disabled={blocked} onClick={()=>s.template(false)}><Box size={14}/>Run task example</button><button disabled={blocked||!s.author?.blender?.configured} onClick={()=>s.template(true)}><Box size={14}/>Blender example</button></div>
      <details><summary>Known measurements <span>{config.measurements.length}</span></summary><p>Give the agent a scale reference from your workspace.</p>{config.measurements.map((item,index)=><div className="agent-measurement" key={index}>
        <input aria-label={'Measurement '+(index+1)+' name'} value={item.label} disabled={blocked} onChange={e=>patchConfig({measurements:config.measurements.map((value,i)=>i===index?{...value,label:e.target.value}:value)})}/>
        <input aria-label={'Measurement '+(index+1)+' value'} type="number" min="0.001" step="any" value={item.value} disabled={blocked} onChange={e=>patchConfig({measurements:config.measurements.map((value,i)=>i===index?{...value,value:Number(e.target.value)}:value)})}/>
        <select aria-label={'Measurement '+(index+1)+' unit'} value={item.unit} disabled={blocked} onChange={e=>patchConfig({measurements:config.measurements.map((value,i)=>i===index?{...value,unit:e.target.value}:value)})}><option>cm</option><option>mm</option><option>m</option></select>
        <button className="agent-icon" aria-label={'Remove measurement '+(index+1)} disabled={blocked} onClick={()=>patchConfig({measurements:config.measurements.filter((_,i)=>i!==index)})}><X size={12}/></button>
      </div>)}<button className="agent-text" disabled={blocked||config.measurements.length>=8} onClick={()=>patchConfig({measurements:[...config.measurements,{label:'Conveyor width',value:40,unit:'cm'}]})}><Plus size={13}/>Add measurement</button></details>
      <details><summary>Mesh assets <span>{config.assetIds.length} attached</span></summary><p>Import a geometry-only OBJ up to 256 KiB, 400 vertices and 800 triangles.</p><button className="agent-text" disabled={blocked} onClick={()=>meshInput.current?.click()}><FileUp size={13}/>Import OBJ</button>{s.assets.map(asset=><label className="agent-check" key={asset.id}><input type="checkbox" checked={config.assetIds.includes(asset.id)} disabled={blocked||(!config.assetIds.includes(asset.id)&&config.assetIds.length>=4)} onChange={e=>patchConfig({assetIds:e.target.checked?[...config.assetIds,asset.id]:config.assetIds.filter(id=>id!==asset.id)})}/><span>{asset.name}<small>{asset.vertex_count} vertices · {asset.face_count} triangles</small></span></label>)}</details>
      <details><summary>Saved reference photos</summary><div className="agent-photo-library">{s.references.map(photo=><button key={photo.id} disabled={blocked||config.imageIds.includes(photo.id)||config.imageIds.length>=4} aria-label={'Attach '+photo.name} onClick={()=>patchConfig({imageIds:[...config.imageIds,photo.id]})}><img src={photo.url} alt={photo.name}/><span>{photo.name}</span></button>)}</div><button className="agent-text" disabled={blocked} onClick={onCamera}><Camera size={13}/>Take a reference photo</button></details>
      <p className="agent-muted">Photos are stored locally. Sending a request shares the attached photos through your selected author connection.</p>
      <button className="agent-text" disabled={blocked||!s.prompt.trim()} onClick={()=>s.perform('draft',async()=>{await s.saveDraft();s.setNotice('Draft saved.');})}><Upload size={13}/>Save request as draft</button>
      <input type="file" ref={meshInput} accept=".obj" hidden onChange={e=>{const file=e.target.files?.[0];if(file)s.uploadMesh(file);e.target.value='';}}/>
    </div>
  </section>;
}
