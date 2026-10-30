import {useEffect,useRef,useState} from 'react';
import {ArrowDown,ArrowUp,Box,Camera,Check,ChevronDown,Download,FileImage,GitBranch,History,Layers3,Loader2,MessageSquare,MousePointer2,Paperclip,Pause,Play,Plus,RotateCcw,Settings2,Sparkles,Square,Upload,X} from 'lucide-react';
import Viewport from './Viewport';
import {PhotoCamera,TaskEditor} from './SceneDialogs';
import SceneSettings from './author/SceneSettings';
import ObjectInspector from './author/ObjectInspector';
import {activeScene,useSceneStudio} from './author/useSceneStudio';
import {api,MODEL_LABELS,type AuthoredScene,type AuthorProvider,type Demo,type TaskSpec} from './types';
import './scene-builder.css';
import './author/agent-studio.css';

const dateLabel=(date:string)=>new Date(date).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const starter='Use this photo to build a conveyor workstation. Add a robot that picks up each box and places it in the tray. The conveyor is 40 cm wide.';

export default function SceneBuilder({onDemoCreated}:{onDemoCreated:(demo:Demo)=>Promise<void>}){
  const s=useSceneStudio();
  const [settings,setSettings]=useState(false);
  const [history,setHistory]=useState(false);
  const [cameraOpen,setCameraOpen]=useState(false);
  const [editingTask,setEditingTask]=useState<TaskSpec>();
  const [mobilePane,setMobilePane]=useState<'conversation'|'scene'>('conversation');
  const [sceneTab,setSceneTab]=useState<'simulation'|'render'|'details'>('simulation');
  const [showObjects,setShowObjects]=useState(true);
  const [recordName,setRecordName]=useState('');
  const [recordOpen,setRecordOpen]=useState(false);
  const [source,setSource]=useState('');
  const [historyFilter,setHistoryFilter]=useState('');
  const [dragging,setDragging]=useState(false);
  const photoInput=useRef<HTMLInputElement>(null);
  const cameraInput=useRef<HTMLInputElement>(null);
  const packageInput=useRef<HTMLInputElement>(null);
  const composer=useRef<HTMLTextAreaElement>(null);
  const chatEnd=useRef<HTMLDivElement>(null);
  const chatScroll=useRef<HTMLDivElement>(null);
  const nearBottom=useRef(true);
  useEffect(()=>{setRecordName(s.current?.spec?.name??s.config.name);setSource(s.current?.spec?JSON.stringify(s.current.spec,null,2):'');setRecordOpen(false);},[s.current?.id,s.current?.status]);
  useEffect(()=>{if(nearBottom.current)chatEnd.current?.scrollIntoView({block:'end',behavior:'instant'});},[s.conversation,s.current?.progress.length]);
  useEffect(()=>{if(composer.current){composer.current.style.height='auto';composer.current.style.height=Math.min(190,Math.max(84,composer.current.scrollHeight))+'px';}},[s.prompt]);
  const photos=s.config.imageIds.map(id=>s.references.find(photo=>photo.id===id)).filter(photo=>!!photo);
  const displayedTurns=s.conversation.filter(turn=>turn.status!=='draft');
  const frame=s.artifact?.frames[s.editing?0:s.frameIndex];
  const ready=!!s.ready;
  const sendDisabled=s.blocked||s.prompt.trim().length<3||!s.connection?.configured||(s.config.builder==='blender'&&!s.author?.blender?.configured);
  function takePhoto(){if(typeof navigator.mediaDevices?.getUserMedia==='function')setCameraOpen(true);else cameraInput.current?.click();}
  function askSelected(){setMobilePane('conversation');setSettings(false);composer.current?.focus();}
  async function send(){nearBottom.current=true;await s.send();}
  async function openRevision(scene:AuthoredScene){await s.perform('open',()=>s.selectScene(scene));setHistory(false);setSceneTab('simulation');}

  return <div className={'agent-studio mobile-'+mobilePane}>
    <header className="agent-toolbar">
      <div className="agent-scene-title"><Box size={17}/><button disabled={s.blocked} onClick={()=>setHistory(!history)} aria-expanded={history}>{s.current?.name??s.config.name}<ChevronDown size={13}/></button>{s.current&&<span className="agent-revision"><GitBranch size={12}/>{Math.max(1,s.conversation.length)}</span>}</div>
      <div className="agent-toolbar-actions"><button className="agent-icon" aria-label="Scene history" title="Scene history" onClick={()=>setHistory(!history)}><History size={16}/></button><button className="agent-icon" disabled={s.blocked} aria-label="Import scene package" title="Import scene package" onClick={()=>packageInput.current?.click()}><Upload size={16}/></button><button className="agent-icon" disabled={s.blocked} aria-label="New scene" title="New scene" onClick={()=>{s.newScene();setMobilePane('conversation');setSettings(false);setHistory(false);}}><Plus size={17}/></button></div>
    </header>
    {history&&<section className="agent-history" aria-label="Scene history"><header><strong>Scenes & revisions</strong><button className="agent-icon" aria-label="Close scene history" onClick={()=>setHistory(false)}><X size={15}/></button></header><input aria-label="Find a scene" placeholder="Find a scene…" value={historyFilter} onChange={e=>setHistoryFilter(e.target.value)}/><div>{s.scenes.filter(scene=>scene.name.toLowerCase().includes(historyFilter.toLowerCase())).map(scene=><button key={scene.id} disabled={s.blocked} aria-current={scene.id===s.current?.id?'true':undefined} onClick={()=>openRevision(scene)}><GitBranch size={14}/><span>{scene.name}<small>{scene.parent_id?'Revision':'Original'} · {dateLabel(scene.created_at)}</small></span><span className={'agent-state '+scene.status}>{scene.status}</span></button>)}{!s.scenes.length&&<p>Your scenes will appear here.</p>}</div></section>}
    <div className="agent-mobile-tabs"><button className={mobilePane==='conversation'?'active':''} onClick={()=>setMobilePane('conversation')}><MessageSquare size={14}/>Agent</button><button className={mobilePane==='scene'?'active':''} onClick={()=>setMobilePane('scene')}><Box size={14}/>Scene{ready&&<span className="agent-dot"/>}</button></div>
    {(s.error||s.notice)&&<div className={'agent-banner '+(s.error?'error':'')} role={s.error?'alert':'status'}><span>{s.error||s.notice}</span><button className="agent-icon" aria-label="Dismiss message" onClick={()=>{s.setError('');s.setNotice('');}}><X size={14}/></button></div>}
    <div className="agent-workspace">
      <section className="agent-conversation" aria-label="Scene author conversation">
        <div className="agent-pane-heading"><span><Sparkles size={14}/>Scene author</span><button className={'agent-icon '+(settings?'active':'')} title="Scene settings" aria-label="Scene settings" onClick={()=>setSettings(!settings)}><Settings2 size={15}/></button></div>
        {settings?<SceneSettings studio={s} onClose={()=>setSettings(false)} onEditTask={setEditingTask} onCamera={takePhoto}/>:<>
          <div className="agent-messages" ref={chatScroll} onScroll={()=>{const el=chatScroll.current;if(el)nearBottom.current=el.scrollHeight-el.scrollTop-el.clientHeight<110;}}>
            {s.loading?<div className="agent-loading"><Loader2 className="spin" size={18}/>Loading your workspace…</div>:!displayedTurns.length?<div className="agent-welcome"><div className="agent-welcome-icon"><Box size={28} strokeWidth={1.3}/></div><h1>What should we build?</h1><p>Describe the space and what the robot should do. A photo helps bring your workspace into the scene.</p><div className="agent-starters"><button onClick={()=>{s.setPrompt(starter);composer.current?.focus();}}><FileImage size={15}/><span>Build from a factory photo<small>Recreate a workspace and its operation</small></span></button><button disabled={s.blocked} onClick={()=>s.template(false)}><Play size={15}/><span>Open a working example<small>Try the simulator without a model call</small></span></button></div></div>:displayedTurns.map((turn,index)=><div className="agent-turn" key={turn.id}>
              <div className="agent-user-message"><div className="agent-message-byline"><span>{turn.source==='manual'?'You · scene edit':turn.source==='template'?'Example':turn.source==='imported'?'Imported scene':'You'}</span><time>{dateLabel(turn.created_at)}</time></div>{!!turn.image_ids?.length&&<div className="agent-message-photos">{turn.image_ids.map(id=>s.references.find(photo=>photo.id===id)).filter(photo=>!!photo).map(photo=><img key={photo.id} src={photo.url} alt={photo.name}/>)}</div>}<p>{turn.prompt}</p></div>
              <div className="agent-assistant-message"><div className="agent-message-byline"><span><Sparkles size={13}/>{turn.source==='astra'?(turn.model&&MODEL_LABELS[turn.model])||'GPT-6 Astra':'Studio'}</span><span>{activeScene(turn)?'Working':turn.status==='ready'?'Completed':turn.status}</span></div>
                {activeScene(turn)?<div className="agent-working"><Loader2 size={15} className="spin"/><p>{turn.progress.at(-1)?.message??'Preparing the scene…'}</p></div>:<p>{turn.error??turn.summary??(turn.status==='ready'?'Scene built and recorded. Inspect the simulation and measured checks.':'This request has not completed.')}</p>}
                {!!turn.progress.length&&<details className="agent-activity"><summary>{activeScene(turn)?'Live activity':`${turn.progress.length} step${turn.progress.length===1?'':'s'}`}</summary><ol>{turn.progress.map((entry,i)=><li key={i}><span>{String(i+1).padStart(2,'0')}</span>{entry.message}</li>)}</ol></details>}
                {turn.status==='ready'&&<button className={'agent-scene-card '+(turn.id===s.current?.id?'current':'')} disabled={s.blocked} onClick={()=>{if(turn.id!==s.current?.id)openRevision(turn);setMobilePane('scene');}}><Box size={17}/><span>{turn.name}<small>Revision {index+1} · {turn.builder==='blender'?'Blender + MuJoCo':'MuJoCo'}{turn.verification?' · '+turn.verification.status:''}</small></span>{turn.id===s.current?.id?<Check size={15}/>:<ArrowDown size={15}/>}</button>}
              </div>
            </div>)}
            <div ref={chatEnd}/>
          </div>
          <div className="agent-composer-area">
            {s.selected&&<div className="agent-context-chip"><MousePointer2 size={12}/><span>Selected: {s.selected.label}</span><button className="agent-icon" aria-label="Remove selected object context" onClick={()=>s.setSelectedId(null)}><X size={12}/></button></div>}
            <div className={'agent-composer '+(dragging?'dragging':'')} onDragOver={e=>{e.preventDefault();if(!s.blocked)setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);if(!s.blocked)s.upload(Array.from(e.dataTransfer.files));}}>
              {!!photos.length&&<div className="agent-attachments">{photos.map(photo=><div key={photo.id}><img src={photo.url} alt={photo.name}/><span>{photo.name}</span><button className="agent-icon" aria-label={'Remove '+photo.name} disabled={s.blocked} onClick={()=>s.patchConfig({imageIds:s.config.imageIds.filter(id=>id!==photo.id)})}><X size={12}/></button></div>)}</div>}
              <textarea ref={composer} aria-label="Message scene author" placeholder={s.selected?`What should change about ${s.selected.label}?`:s.current?'Ask for a change to the scene…':'Describe a scene, or drop a photo here…'} value={s.prompt} maxLength={8000} disabled={s.blocked} onChange={e=>s.setPrompt(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();if(!sendDisabled)send();}}}/>
              <div className="agent-composer-bottom"><div><button className="agent-icon" aria-label="Attach photos" title="Attach photos" disabled={s.blocked||photos.length>=4} onClick={()=>photoInput.current?.click()}><Paperclip size={16}/></button><button className="agent-icon" aria-label="Take photo" title="Take photo" disabled={s.blocked||photos.length>=4} onClick={takePhoto}><Camera size={16}/></button><label className="agent-model-choice"><select aria-label="Scene author model" title="Switch the model for the next message" value={s.config.provider} disabled={s.blocked} onChange={e=>s.patchConfig({provider:e.target.value as AuthorProvider})}>{([['codex','GPT-6 Astra'],['claude','Claude Opus 5.5'],['openai','GPT-6 Astra · API']] as const).map(([id,label])=><option key={id} value={id}>{label}{s.author?.providers.find(p=>p.id===id)?.configured===false?' · offline':''}</option>)}</select><ChevronDown size={12}/></label></div>{s.running?<button className="agent-send stopping" aria-label="Stop generation" onClick={s.cancel}><Square size={13} fill="currentColor"/></button>:<button className="agent-send" disabled={sendDisabled} aria-label="Send to scene author" title="Send · Enter" onClick={send}>{s.busy?<Loader2 size={16} className="spin"/>:<ArrowUp size={18}/>}</button>}</div>
            </div>
            <div className="agent-composer-meta"><span><i className={'agent-dot '+(s.connection?.configured?'':'offline')}/>{s.config.provider==='codex'?'Codex subscription':s.config.provider==='claude'?'Claude Code subscription':'OpenAI API'}</span><button onClick={()=>setSettings(true)}>{s.config.builder==='blender'?'Blender + MuJoCo':'MuJoCo'}<ChevronDown size={10}/></button><span className="agent-enter-hint">Enter to send</span></div>
            {s.author&&!s.connection?.configured&&<button className="agent-connection-hint" onClick={()=>setSettings(true)}>Connect the author in settings. Examples and manual edits work locally.</button>}
          </div>
        </>}
      </section>
      <section className="agent-scene-pane" aria-label="Simulation and scene editor">
        <div className="agent-scene-tabs"><div role="tablist" aria-label="Scene views">{(['simulation','render','details'] as const).map(tab=><button role="tab" key={tab} aria-selected={sceneTab===tab} className={sceneTab===tab?'active':''} onClick={()=>setSceneTab(tab)}>{tab==='simulation'?'Simulator':tab==='render'?'Blender render':'Details'}</button>)}</div><button className={'agent-icon '+(showObjects?'active':'')} aria-label="Toggle object inspector" title="Object inspector" onClick={()=>setShowObjects(!showObjects)}><Layers3 size={15}/></button></div>
        <div className="agent-scene-content">
          {sceneTab==='simulation'?<>
            <div className="agent-simulation-toolbar"><div><button className={s.editing?'active':''} disabled={!ready||s.blocked} onClick={s.startEditing}><MousePointer2 size={13}/>Edit scene</button><button className={!s.editing?'active':''} disabled={!ready||s.blocked} onClick={()=>{s.setEditing(false);s.setPlaying(false);}}><Play size={12}/>Recording</button></div><span>{s.running?'Building revision…':s.editing?'Initial state':`${s.artifact?.frames.length??0} recorded frames`}</span></div>
            <div className="agent-view-area"><Viewport frame={frame} objects={s.objects} selectedObjectId={s.selectedId} onSelectObject={s.setSelectedId} onMoveObject={(id,position)=>s.editObject(id,{position})} editing={s.editing&&ready} disabled={s.blocked}/>{!s.artifact&&<div className="agent-viewport-empty">{s.running||s.recordingLoading?<Loader2 size={28} className="spin"/>:<Box size={32} strokeWidth={1}/>}<h2>{s.running?'Building your scene':s.recordingLoading?'Loading the recording':'Your scene starts here'}</h2><p>{s.running?'Geometry, physics, and a recorded operation.':s.recordingLoading?'Fetching this revision\'s simulated frames.':'Ask the agent to create a scene, or open an example.'}</p></div>}</div>
            <div className="agent-playback"><button className="agent-icon" aria-label={s.playing?'Pause scene playback':'Play scene recording'} disabled={!s.artifact||s.blocked} onClick={s.startPlayback}>{s.playing?<Pause size={15}/>:<Play size={15}/>}</button><span>{(frame?.t??0).toFixed(2)}<small> / {(s.artifact?.frames.at(-1)?.t??0).toFixed(2)}s</small></span><input aria-label="Scene recording timeline" type="range" min={0} max={Math.max(1,(s.artifact?.frames.length??1)-1)} value={s.editing?0:s.frameIndex} disabled={!s.artifact||s.blocked} onChange={e=>{s.setEditing(false);s.setPlaying(false);s.setFrameIndex(Number(e.target.value));}}/><button className="agent-icon" aria-label="Restart scene recording" disabled={!s.artifact} onClick={()=>{s.setFrameIndex(0);s.setPlaying(false);}}><RotateCcw size={14}/></button></div>
            {showObjects&&<ObjectInspector studio={s} onAsk={askSelected}/>}
          </>:sceneTab==='render'?<div className="agent-render">{s.current?.blender?<><img src={s.current.blender.preview_url} alt="Blender layout at the initial simulation state"/><div><strong>Blender {s.current.blender.version}</strong><p>Static layout render. Inspect the recorded simulation for robot motion.</p><span>{s.current.blender.visual_meshes} visual meshes · {s.current.blender.collision_shapes} collision shapes</span><a className="agent-button" href={s.current.blender.blend_url} download><Download size={14}/>Download Blender file</a></div></>:<div className="agent-panel-empty"><Box size={25}/><h2>No Blender render yet</h2><p>Choose Blender in scene settings to generate a rendered layout alongside the simulation.</p><button onClick={()=>{setSettings(true);setMobilePane('conversation');}}>Scene settings</button></div>}</div>:<div className="agent-details">
            <h2>{s.current?.name??'Scene details'}</h2><p>{s.current?.spec?.description??'Scene details and task checks appear after a build.'}</p>
            {s.current?.verification&&<section><h3>Measured task checks <span className={'agent-state '+s.current.verification.status}>{s.current.verification.status}</span></h3>{s.current.verification.outcomes.map(outcome=><div className="agent-outcome" key={outcome.id}><span className={'agent-state '+outcome.status}>{outcome.status}</span><div><strong>{outcome.id.replaceAll('_',' ')}</strong><p>{outcome.explanation}</p></div></div>)}</section>}
            {!!s.current?.spec?.assumptions.length&&<section><h3>Assumptions</h3><ul>{s.current.spec.assumptions.map((item,index)=><li key={index}>{item}</li>)}</ul></section>}
            {s.current?.package_source?.attachments&&<section><h3>Imported source files</h3><p>{s.current.package_source.note}</p><div className="agent-source-files">{Object.entries(s.current.package_source.attachments).map(([filename,url])=><a className="agent-button" key={filename} href={url} download><Download size={13}/>{filename}</a>)}</div></section>}
            {ready&&<details><summary>Advanced scene specification</summary><p>Editing this resolved specification creates a MuJoCo revision. Use object controls to preserve Blender assemblies.</p><textarea aria-label="Scene specification JSON" value={source} onChange={e=>setSource(e.target.value)} rows={16}/><button disabled={s.blocked} onClick={()=>s.perform('source',async()=>{await s.acceptScene(await api<AuthoredScene>('/scenes/'+s.current!.id+'/preview',JSON.parse(source)));})}>Apply specification & simulate</button></details>}
          </div>}
        </div>
        <footer className="agent-scene-footer"><span>{s.current?.verification?<><i className={'agent-dot '+(s.current.verification.status==='pass'?'':'warning')}/>{s.current.verification.status==='pass'?'Task checks passed':s.current.verification.status==='fail'?'Task needs revision':'Review task checks'}</>:s.running?<><Loader2 size={12} className="spin"/>Building</>:'No simulation recorded'}</span><div>{ready&&<a className="agent-icon" href={'/api/scenes/'+s.current!.id+'/package'} download title="Export portable scene package" aria-label="Export portable scene package"><Download size={15}/></a>}<button className="agent-primary" disabled={!ready||s.blocked} onClick={()=>setRecordOpen(!recordOpen)}><Layers3 size={14}/>Save demonstration</button></div></footer>
        {recordOpen&&<div className="agent-record-popover"><header><strong>Save demonstration</strong><button className="agent-icon" aria-label="Close save demonstration" onClick={()=>setRecordOpen(false)}><X size={14}/></button></header><label>Demonstration name<input value={recordName} maxLength={100} onChange={e=>setRecordName(e.target.value)}/></label><p>Saves this recorded operation to the library. Model learning is measured in a separate experiment.</p><button className="agent-primary" disabled={s.blocked||!recordName.trim()} onClick={()=>s.perform('record',async()=>{const demo=await api<Demo>('/scenes/'+s.current!.id+'/record',{name:recordName.trim()});await onDemoCreated(demo);})}>{s.busy==='record'?<Loader2 className="spin" size={14}/>:<Check size={14}/>}Save & open workspace</button></div>}
      </section>
    </div>
    <input ref={photoInput} type="file" hidden accept="image/jpeg,image/png,image/webp" multiple onChange={e=>{s.upload(Array.from(e.target.files??[]));e.target.value='';}}/>
    <input ref={cameraInput} type="file" hidden accept="image/jpeg,image/png,image/webp" capture="environment" onChange={e=>{s.upload(Array.from(e.target.files??[]));e.target.value='';}}/>
    <input ref={packageInput} type="file" hidden accept=".zip" onChange={e=>{const file=e.target.files?.[0];if(file)s.importPackage(file);e.target.value='';}}/>
    {cameraOpen&&<PhotoCamera onClose={()=>setCameraOpen(false)} onCapture={file=>{setCameraOpen(false);s.upload([file]);}} onFallback={()=>{setCameraOpen(false);cameraInput.current?.click();}}/>}
    {editingTask&&<TaskEditor task={editingTask} onClose={()=>setEditingTask(undefined)} onSave={task=>{s.setTasks(previous=>[task,...previous]);s.patchConfig({taskId:task.id??''});setEditingTask(undefined);s.setNotice('New task version selected for the next request.');}}/>}
  </div>;
}
