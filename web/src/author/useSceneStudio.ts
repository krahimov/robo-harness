import {useEffect,useRef,useState} from 'react';
import {api,type Artifact,type AuthoredScene,type AuthorStatus,type ReferenceImage,type SceneAsset,type SceneComponent,type SceneConfig,type TaskSpec} from '../types';
import type {EditableSceneObject} from '../Viewport';

export const activeScene=(scene?:AuthoredScene)=>!!scene&&['queued','running'].includes(scene.status);
const emptyConfig:SceneConfig={name:'Untitled scene',provider:'codex',builder:'blender',embodiment:'franka-panda-v1',taskId:'block-to-tray',measurements:[],assetIds:[],imageIds:[]};
const configFrom=(scene:AuthoredScene):SceneConfig=>({name:scene.name,provider:scene.provider??'codex',builder:scene.builder??'mujoco',embodiment:scene.embodiment??scene.spec?.scene.embodiment??'cartesian-gripper-v1',taskId:scene.task_id??'',measurements:scene.measurements??[],assetIds:scene.asset_ids??[],imageIds:scene.image_ids});
const dataUrl=(file:File)=>new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error('Could not read the file.'));reader.readAsDataURL(file);});

export function useSceneStudio(){
  const [author,setAuthor]=useState<AuthorStatus>();
  const [scenes,setScenes]=useState<AuthoredScene[]>([]);
  const [current,setCurrent]=useState<AuthoredScene>();
  const [conversation,setConversation]=useState<AuthoredScene[]>([]);
  const [references,setReferences]=useState<ReferenceImage[]>([]);
  const [tasks,setTasks]=useState<TaskSpec[]>([]);
  const [assets,setAssets]=useState<SceneAsset[]>([]);
  const [components,setComponents]=useState<SceneComponent[]>([]);
  const [objects,setObjects]=useState<EditableSceneObject[]>([]);
  const [config,setConfig]=useState<SceneConfig>(emptyConfig);
  const [prompt,setPrompt]=useState('');
  const [artifact,setArtifact]=useState<Artifact>();
  const [selectedId,setSelectedId]=useState<string|null>(null);
  const [busy,setBusy]=useState('');
  const [error,setError]=useState('');
  const [notice,setNotice]=useState('');
  const [loading,setLoading]=useState(true);
  const [frameIndex,setFrameIndex]=useState(0);
  const [playing,setPlaying]=useState(false);
  const [editing,setEditing]=useState(true);
  const mounted=useRef(true);
  const selection=useRef(0);
  const currentId=useRef<string|undefined>(undefined);
  const draftText=useRef(prompt);draftText.current=prompt;
  const draftKey=(id?:string)=>'studio.author.composer.'+(id??'new');
  function persistDraft(){localStorage.setItem(draftKey(currentId.current),draftText.current);}
  const running=activeScene(current);
  const blocked=loading||!!busy||running;
  const ready=current?.status==='ready'&&!!artifact;
  const selected=objects.find(item=>item.id===selectedId);
  const connection=author?.providers.find(item=>item.id===config.provider);
  const task=(current?.task_id===config.taskId?current.task:undefined)??tasks.find(item=>item.id===config.taskId);
  const patchConfig=(patch:Partial<SceneConfig>)=>setConfig(previous=>({...previous,...patch}));
  const remember=(scene:AuthoredScene)=>setScenes(previous=>[scene,...previous.filter(item=>item.id!==scene.id)]);

  async function perform(label:string,action:()=>Promise<void>){
    setBusy(label);setError('');setNotice('');
    try{await action();}catch(reason){if(mounted.current)setError((reason as Error).message);}
    finally{if(mounted.current)setBusy('');}
  }
  async function loadSceneData(scene:AuthoredScene,token:number){
    const [history,recording,listed,catalog]=await Promise.all([
      api<{scenes:AuthoredScene[]}>('/scenes/'+scene.id+'/conversation'),
      scene.artifact_id?api<Artifact>('/artifacts/'+scene.artifact_id):Promise.resolve(undefined),
      scene.spec?api<{objects:EditableSceneObject[]}>('/scenes/'+scene.id+'/objects'):Promise.resolve({objects:[]}),
      scene.spec?api<{components:SceneComponent[]}>('/scene-components?scene_id='+scene.id):Promise.resolve(undefined)
    ]);
    if(!mounted.current||token!==selection.current)return;
    setConversation(history.scenes);setArtifact(recording);setObjects(listed.objects);if(catalog)setComponents(catalog.components);
    setSelectedId(previous=>listed.objects.some(item=>item.id===previous)?previous:null);
  }
  async function selectScene(scene:AuthoredScene,keepPrompt=false){
    persistDraft();
    const token=++selection.current;currentId.current=scene.id;
    setCurrent(scene);setConfig(configFrom(scene));setPlaying(false);setFrameIndex(0);setEditing(true);setError('');setNotice('');
    if(!keepPrompt)setPrompt(localStorage.getItem(draftKey(scene.id))??(scene.status==='draft'?scene.prompt:''));
    setArtifact(undefined);setObjects([]);setConversation([scene]);
    await loadSceneData(scene,token);
  }
  async function acceptScene(scene:AuthoredScene,keepPrompt=false){remember(scene);await selectScene(scene,keepPrompt);}
  useEffect(()=>{
    mounted.current=true;const initialization=++selection.current;
    async function load(){
      try{
        const [saved,images,knownTasks,knownAssets]=await Promise.all([api<AuthoredScene[]>('/scenes'),api<ReferenceImage[]>('/references'),api<TaskSpec[]>('/tasks'),api<SceneAsset[]>('/assets')]);
        if(!mounted.current||initialization!==selection.current)return;
        setScenes(saved);setReferences(images);setTasks(knownTasks);setAssets(knownAssets);
        const remembered=localStorage.getItem('studio.author.scene');
        const initial=saved.find(item=>item.id===remembered)??saved[0];
        if(initial)await selectScene(initial);
      }catch(reason){if(mounted.current)setError((reason as Error).message);}
      finally{if(mounted.current)setLoading(false);}
    }
    load();
    api<AuthorStatus>('/author/status').then(value=>{if(mounted.current)setAuthor(value);}).catch(()=>{});
    api<{components:SceneComponent[]}|SceneComponent[]>('/scene-components').then(value=>{if(mounted.current)setComponents(Array.isArray(value)?value:value.components);}).catch(()=>{});
    return()=>{mounted.current=false;selection.current++;};
  },[]);
  useEffect(()=>{if(current?.id)localStorage.setItem('studio.author.scene',current.id);},[current?.id]);
  useEffect(()=>{if(!loading&&!running)localStorage.setItem(draftKey(current?.id),prompt);},[current?.id,prompt,loading,running]);
  useEffect(()=>{
    if(!running||!current)return;
    let cancelled=false;let timer:ReturnType<typeof setTimeout>;
    const id=current.id;
    async function poll(){
      try{
        const next=await api<AuthoredScene>('/scenes/'+id);
        if(cancelled||currentId.current!==id)return;
        setCurrent(next);remember(next);setConversation(previous=>[...previous.filter(item=>item.id!==id),next]);
        if(activeScene(next)){timer=setTimeout(poll,1000);return;}
        setConfig(configFrom(next));setFrameIndex(0);
        await loadSceneData(next,selection.current);
        const knownAssets=await api<SceneAsset[]>('/assets');
        if(mounted.current&&currentId.current===id)setAssets(knownAssets);
      }catch(reason){if(mounted.current&&currentId.current===id){setError((reason as Error).message+' Reopen this scene to reload its evidence.');if(!cancelled)timer=setTimeout(poll,2500);}}
    }
    timer=setTimeout(poll,750);
    return()=>{cancelled=true;clearTimeout(timer);};
  },[current?.id,running]);
  useEffect(()=>{
    if(!playing||!artifact)return;
    const timer=setInterval(()=>setFrameIndex(index=>{if(index>=artifact.frames.length-1){setPlaying(false);return index;}return index+1;}),1000/artifact.frequency);
    return()=>clearInterval(timer);
  },[playing,artifact]);

  function newScene(){persistDraft();selection.current++;currentId.current=undefined;localStorage.removeItem('studio.author.scene');setCurrent(undefined);setConversation([]);setArtifact(undefined);setObjects([]);setSelectedId(null);setConfig({...emptyConfig,provider:config.provider});setPrompt('');setError('');setNotice('');setPlaying(false);setFrameIndex(0);setEditing(true);}
  async function saveDraft(){
    const draft=await api<AuthoredScene>('/scenes',{name:config.name.trim()||'Untitled scene',prompt:prompt.trim(),provider:config.provider,builder:config.builder,embodiment:config.embodiment,task_id:config.taskId||null,measurements:config.measurements,asset_ids:config.assetIds,image_ids:config.imageIds,selected_object_id:selectedId,...(current?{parent_id:current.id}:{})});
    await acceptScene(draft);return draft;
  }
  async function send(){await perform('send',async()=>{
    if(!connection?.configured)throw new Error(connection?.message??'Check the author connection in settings.');
    if(config.builder==='blender'&&!author?.blender?.configured)throw new Error(author?.blender?.message??'Blender is not ready. Check settings.');
    if(prompt.trim().length<3)throw new Error('Describe what you want to build or change.');
    const draft=await saveDraft();
    const queued=await api<AuthoredScene>('/scenes/'+draft.id+'/generate',{});
    remember(queued);setCurrent(queued);setPrompt('');localStorage.removeItem(draftKey(draft.id));setConversation(previous=>[...previous.filter(item=>item.id!==queued.id),queued]);
  });}
  async function cancel(){if(current)await perform('cancel',async()=>{const next=await api<AuthoredScene>('/scenes/'+current.id+'/cancel',{});setCurrent(next);remember(next);setConversation(previous=>previous.map(item=>item.id===next.id?next:item));});}
  async function upload(files:File[]){await perform('upload',async()=>{
    if(files.length+config.imageIds.length>(author?.max_images??4))throw new Error('Attach up to four photos.');
    for(const file of files){if(!['image/jpeg','image/png','image/webp'].includes(file.type)||file.size>(author?.max_image_bytes??8388608))throw new Error('Choose JPEG, PNG or WebP photos up to 8 MiB each.');}
    for(const file of files){const reference=await api<ReferenceImage>('/references',{name:file.name,data_url:await dataUrl(file)});setReferences(previous=>[reference,...previous]);setConfig(previous=>({...previous,imageIds:[...previous.imageIds,reference.id]}));}
  });}
  async function uploadMesh(file:File){await perform('mesh',async()=>{
    if(!file.name.toLowerCase().endsWith('.obj')||file.size>262144)throw new Error('Choose a geometry-only OBJ up to 256 KiB.');
    const asset=await api<SceneAsset>('/assets',{name:file.name,obj:await file.text()});setAssets(previous=>[asset,...previous]);setConfig(previous=>({...previous,assetIds:previous.assetIds.length<4?[...previous.assetIds,asset.id]:previous.assetIds}));
  });}
  async function template(blender=false){await perform('template',async()=>{
    const scene=await api<AuthoredScene>('/scenes/template?task_id='+encodeURIComponent(blender?'block-to-tray':config.taskId)+'&embodiment='+config.embodiment+'&builder='+(blender?'blender':'mujoco'),{});
    await acceptScene(scene);setNotice('Example built and recorded. No model call was made.');
  });}
  async function editObject(id:string,changes:{position?:number[];size?:number[];color?:number[]}){if(!current)return;await perform('edit',async()=>{
    const next=await api<AuthoredScene>('/scenes/'+current.id+'/edit',{object_id:id,...changes});await acceptScene(next,true);setSelectedId(id);
  });}
  async function insertComponent(id:string){if(!current)return;await perform('component',async()=>{await acceptScene(await api<AuthoredScene>('/scenes/'+current.id+'/components',{component_id:id}),true);});}
  async function importPackage(file:File){await perform('import',async()=>{
    if(!file.name.toLowerCase().endsWith('.zip')||file.size>64*1024*1024)throw new Error('Choose a scene package ZIP up to 64 MiB.');
    const encoded=(await dataUrl(file)).split(',')[1];await acceptScene(await api<AuthoredScene>('/scenes/import',{data_base64:encoded}));
    const [images,meshes,knownTasks]=await Promise.all([api<ReferenceImage[]>('/references'),api<SceneAsset[]>('/assets'),api<TaskSpec[]>('/tasks')]);setReferences(images);setAssets(meshes);setTasks(knownTasks);setNotice('Package imported and simulation rebuilt. Blender source files remain archived in the package.');
  });}
  async function refreshConnection(){await perform('connection',async()=>setAuthor(await api<AuthorStatus>('/author/status?refresh=true')));}
  function startPlayback(){setEditing(false);if(artifact&&frameIndex===artifact.frames.length-1)setFrameIndex(0);setPlaying(previous=>!previous);}
  function startEditing(){setPlaying(false);setFrameIndex(0);setEditing(true);}
  return {author,scenes,current,conversation,references,tasks,assets,components,objects,config,prompt,artifact,selectedId,selected,busy,error,notice,loading,frameIndex,playing,editing,running,blocked,ready,connection,task,patchConfig,setPrompt,setSelectedId,setError,setNotice,setFrameIndex,setPlaying,setEditing,setTasks,perform,selectScene,acceptScene,newScene,saveDraft,send,cancel,upload,uploadMesh,template,editObject,insertComponent,importPackage,refreshConnection,startPlayback,startEditing};
}
export type SceneStudio=ReturnType<typeof useSceneStudio>;
