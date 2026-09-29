import {useEffect,useRef,useState} from 'react';
import {Camera,Loader2,Save,X} from 'lucide-react';
import {api,type TaskSpec} from './types';
export function PhotoCamera({onClose,onCapture,onFallback}:{onClose:()=>void;onCapture:(file:File)=>void;onFallback:()=>void}){
  const dialog=useRef<HTMLDialogElement>(null);
  const video=useRef<HTMLVideoElement>(null);
  const [error,setError]=useState('');
  const [ready,setReady]=useState(false);
  const [capturing,setCapturing]=useState(false);
  useEffect(()=>{
    dialog.current?.showModal();let live=true;let stream:MediaStream|undefined;
    navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1600},height:{ideal:1200}},audio:false}).then(async media=>{stream=media;if(!live){media.getTracks().forEach(track=>track.stop());return;}if(video.current){video.current.srcObject=media;await video.current.play();}}).catch(reason=>{if(live)setError(reason.name==='NotAllowedError'?'Camera access was not allowed. Use your device camera or upload a photo instead.':'Could not open a camera. Use your device camera or upload a photo instead.');});
    return()=>{live=false;stream?.getTracks().forEach(track=>track.stop());};
  },[]);
  function capture(){const source=video.current;if(!source?.videoWidth)return;setCapturing(true);const canvas=document.createElement('canvas');canvas.width=source.videoWidth;canvas.height=source.videoHeight;const context=canvas.getContext('2d');if(!context){setError('Could not capture this camera frame.');setCapturing(false);return;}context.drawImage(source,0,0);canvas.toBlob(blob=>{if(blob){onCapture(new File([blob],`workspace-${new Date().toISOString().replaceAll(':','-')}.jpg`,{type:'image/jpeg'}));}else{setError('Could not save the photo. Please try again.');setCapturing(false);}},'image/jpeg',.9);}
  return <dialog className="sb-camera-dialog" ref={dialog} onCancel={onClose} onClick={event=>{if(event.target===dialog.current)onClose();}}><div className="modal-heading"><h2>Take a reference photo</h2><button className="icon-button" aria-label="Close camera" onClick={onClose}><X size={18}/></button></div><div className="sb-camera-preview"><video ref={video} autoPlay playsInline muted onLoadedData={()=>setReady(true)}/>{!ready&&!error&&<span><Loader2 className="spin" size={22}/>Opening camera…</span>}</div>{error&&<p className="sb-camera-error" role="alert">{error}</p>}<div className="modal-footer"><button className="button" onClick={onFallback}>Use device camera / file</button><button className="button primary" disabled={!ready||capturing||!!error} onClick={capture}><Camera size={15}/>Capture photo</button></div></dialog>;
}

export function TaskEditor({task,onClose,onSave}:{task:TaskSpec;onClose:()=>void;onSave:(task:TaskSpec)=>void}){
  const dialog=useRef<HTMLDialogElement>(null);
  const [name,setName]=useState(task.name);
  const [description,setDescription]=useState(task.description);
  const [requirements,setRequirements]=useState(JSON.stringify(task.requirements,null,2));
  const [operations,setOperations]=useState(JSON.stringify(task.operations??[],null,2));
  const [error,setError]=useState('');
  const [saving,setSaving]=useState(false);
  useEffect(()=>{dialog.current?.showModal();},[]);
  async function save(){setSaving(true);setError('');try{const saved=await api<TaskSpec>('/tasks',{name,description,requirements:JSON.parse(requirements),operations:JSON.parse(operations),parent_id:task.id});onSave(saved);}catch(reason){setError((reason as Error).message);}finally{setSaving(false);}}
  return <dialog className="sb-task-dialog" ref={dialog} onCancel={()=>!saving&&onClose()}><div className="modal-heading"><h2>New task version</h2><button className="icon-button" aria-label="Close task editor" disabled={saving} onClick={onClose}><X size={18}/></button></div><div className="sb-task-editor"><p>These are your success criteria. The scene author and judge cannot edit them. Existing recordings keep the previous version.</p><label className="field-label">Task name<input value={name} maxLength={100} disabled={saving} onChange={event=>setName(event.target.value)}/></label><label className="field-label">Operation description<textarea value={description} maxLength={3000} rows={3} disabled={saving} onChange={event=>setDescription(event.target.value)}/></label><label className="field-label">Outcome checks<textarea className="sb-task-json" value={requirements} rows={12} disabled={saving} onChange={event=>setRequirements(event.target.value)}/></label><label className="field-label">Teacher operations<textarea className="sb-task-json" value={operations} rows={6} disabled={saving} onChange={event=>setOperations(event.target.value)}/></label><p>Checks use object IDs, numerical regions and thresholds. Teacher operations define up to two sequential transfers or a lift-and-hold example. JSON validation catches unsupported fields.</p>{error&&<p className="sb-camera-error" role="alert">{error}</p>}</div><div className="modal-footer"><button className="button" disabled={saving} onClick={onClose}>Cancel</button><button className="button primary" disabled={saving||!name.trim()||!description.trim()} onClick={save}>{saving?<Loader2 size={14} className="spin"/>:<Save size={14}/>}Save task version</button></div></dialog>;
}
