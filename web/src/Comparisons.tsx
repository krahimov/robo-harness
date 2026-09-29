import {useEffect,useState} from 'react';
import {ArrowRight,Download,FlaskConical,Loader2,Plus,Square,X} from 'lucide-react';
import {api,type Bootstrap,type Scene} from './types';
import './comparisons.css';

type Condition={name:string;demo_ids:string[]};
type Row=Condition&{experiment_id:string;status:string;error?:string;recorded:number;planned:number;resolved:number;unresolved:number;counts:Record<string,number>;success_rate:number|null;confidence_interval:number[]|null;paired:{baseline:string;count:number;gained:number;lost:number;delta:number|null};attempts:{seed:number;status:string}[]};
type Comparison={id:string;name:string;status:string;adapter:string;seeds:number[];conditions:Row[];horizon_steps:number;note:string;scene:Scene};
const rate=(v:number|null)=>v===null?'—':`${Math.round(v*100)}%`;

export default function Comparisons({data,scene,judgeId,selected,taskId,instructions,onOpenRun}:{data:Bootstrap;scene:Scene;judgeId:string;selected:string[];taskId:string;instructions:string;onOpenRun:(id:string)=>void}){
  const compatibleDemos=data.demos.filter(d=>(d.scene.embodiment||'cartesian-gripper-v1')===(scene.embodiment||'cartesian-gripper-v1'));
  const compatibleSelected=selected.filter(id=>compatibleDemos.some(d=>d.id===id));
  const [name,setName]=useState('Demonstration context comparison');
  const [adapter,setAdapter]=useState('replay');
  const [judge,setJudge]=useState(judgeId||data.judges[0]?.id||'');
  const [seed,setSeed]=useState(Math.min(scene.seed+1000,2147483500));
  const [variation,setVariation]=useState(Math.max(scene.variation,.04));
  const [episodes,setEpisodes]=useState(8);
  const [steps,setSteps]=useState(Math.min(1200,Math.max(180,...data.demos.filter(d=>selected.includes(d.id)).map(d=>d.frames-1))));
  const [conditions,setConditions]=useState<Condition[]>([{name:'One demonstration',demo_ids:(compatibleSelected.length?compatibleSelected:compatibleDemos.map(d=>d.id)).slice(0,1)},{name:'Several demonstrations',demo_ids:(compatibleSelected.length>1?compatibleSelected:compatibleDemos.map(d=>d.id)).slice(0,3)}]);
  const [saved,setSaved]=useState<Comparison[]>([]);
  const [current,setCurrent]=useState<Comparison>();
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const model=data.adapters.find(a=>a.id===adapter);
  const active=current?.status==='running';
  useEffect(()=>{let live=true;api<Comparison[]>('/comparisons').then(items=>{if(live){setSaved(items);setCurrent(items[0]);}}).catch(e=>{if(live)setError(e.message);});return()=>{live=false;};},[]);
  useEffect(()=>{if(!active||!current)return;let live=true;const timer=setInterval(()=>{api<Comparison>('/comparisons/'+current.id).then(next=>{if(live){setCurrent(next);setSaved(old=>old.map(c=>c.id===next.id?next:c));}}).catch(e=>{if(live)setError(e.message);});},1200);return()=>{live=false;clearInterval(timer);};},[current?.id,active]);
  function edit(index:number,changes:Partial<Condition>){setConditions(old=>old.map((c,i)=>i===index?{...c,...changes}:c));}
  async function submit(){setBusy(true);setError('');try{const next=await api<Comparison>('/comparisons',{name,conditions,adapter,judge_id:judge,scene:{...scene,seed,variation},episodes,horizon_steps:steps,task_id:taskId||null,instructions});setCurrent(next);setSaved(old=>[next,...old]);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function exportRun(){if(!current)return;try{const result=await api('/comparisons/'+current.id+'/export');const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='context-comparison-'+current.id+'.json';link.click();URL.revokeObjectURL(url);}catch(e){setError((e as Error).message);}}
  return <div className="comparisons">
    {error&&<div className="inline-error" role="alert">{error}</div>}
    <section className="comparison-setup">
      <div className="section-title"><FlaskConical size={18}/><h2>Change context. Keep the evaluation conditions fixed.</h2></div>
      <p className="field-help">Each condition uses the same robot, task, judge version, starting seeds, and time budget. Only its ordered demonstration context changes.</p>
      <div className="comparison-fields"><label className="field-label">Comparison name<input value={name} maxLength={100} onChange={e=>setName(e.target.value)}/></label><label className="field-label">Robot model<select value={adapter} onChange={e=>setAdapter(e.target.value)}>{data.adapters.filter(a=>a.type==='model'&&a.supports_comparisons!==false).map(a=><option key={a.id} value={a.id}>{a.name}</option>)}</select></label><label className="field-label">Judge version<select value={judge} onChange={e=>setJudge(e.target.value)}>{data.judges.map(j=><option key={j.id} value={j.id}>{j.name} · v{j.version}</option>)}</select></label></div>
      <p className="field-help">{model?.description}</p>
      <div className="comparison-fields four"><label className="field-label">First evaluation seed<input type="number" min={0} max={2147483500} value={seed} onChange={e=>setSeed(+e.target.value)}/></label><label className="field-label">Attempts per condition<input type="number" min={1} max={24} value={episodes} onChange={e=>setEpisodes(+e.target.value)}/></label><label className="field-label">Position variation · metres<input type="number" min={0} max={.15} step={.01} value={variation} onChange={e=>setVariation(+e.target.value)}/></label><label className="field-label">Control steps<input type="number" min={1} max={1200} value={steps} onChange={e=>setSteps(+e.target.value)}/></label></div>
      <div className="comparison-conditions">{conditions.map((c,i)=><article key={i}><div className="condition-title"><input aria-label={'Condition '+(i+1)+' name'} maxLength={60} value={c.name} onChange={e=>edit(i,{name:e.target.value})}/>{conditions.length>2&&<button className="icon-button" aria-label={'Remove condition '+(i+1)} onClick={()=>setConditions(old=>old.filter((_,j)=>j!==i))}><X size={14}/></button>}</div><div className="condition-demo-list">{compatibleDemos.map(d=><label key={d.id}><input type="checkbox" checked={c.demo_ids.includes(d.id)} disabled={!c.demo_ids.includes(d.id)&&c.demo_ids.length>=(adapter==='icrt'?4:8)} onChange={()=>edit(i,{demo_ids:c.demo_ids.includes(d.id)?c.demo_ids.filter(id=>id!==d.id):[...c.demo_ids,d.id]})}/><span>{d.name}<small>Source seed {d.scene.seed}</small></span>{c.demo_ids.includes(d.id)&&<span className="condition-order">{c.demo_ids.indexOf(d.id)+1}</span>}</label>)}</div><p>{c.demo_ids.length?`${c.demo_ids.length} demonstrations, in selection order`:'Zero-demo condition'}</p></article>)}</div>
      <div className="comparison-actions"><button className="button small" disabled={conditions.length>=5} onClick={()=>setConditions(old=>[...old,{name:'No demonstrations',demo_ids:[]}])}><Plus size={14}/>Add condition</button><span className="field-help">{adapter==='replay'?'Replay needs a demonstration; zero-demo runs require an inference model.':'An empty selection runs the model without demonstration context.'}</span><button className="button primary" disabled={busy||!name.trim()||!judge||!model?.ready} onClick={submit}>{busy?<Loader2 className="spin" size={15}/>:<FlaskConical size={15}/>}Run comparison</button></div>
      {!model?.ready&&<p className="connection-hint">Connect the selected model in Adapters before running this comparison.</p>}
    </section>
    {!!saved.length&&<label className="field-label comparison-history">Saved comparison<select value={current?.id||''} onChange={e=>setCurrent(saved.find(c=>c.id===e.target.value))}>{saved.map(c=><option key={c.id} value={c.id}>{c.name} · {c.status}</option>)}</select></label>}
    {current&&<section className="comparison-results"><div className="section-header"><div className="section-title"><h2>{current.name}</h2><span className={'badge '+current.status}>{current.status}</span></div><div className="comparison-actions"><button className="button small" onClick={exportRun}><Download size={14}/>Export evidence</button>{active&&<button className="button small" onClick={async()=>{try{await api('/comparisons/'+current.id+'/cancel',{});}catch(e){setError((e as Error).message);}}}><Square size={12}/>Cancel all</button>}</div></div>
      <div className="comparison-summary">{current.conditions.map(c=><article key={c.experiment_id}><span className="eyebrow">{c.demo_ids.length} DEMONSTRATIONS</span><h3>{c.name}</h3><strong className="comparison-rate">{rate(c.success_rate)}</strong><p>{c.counts.pass} passes / {c.resolved} resolved attempts</p><div className="comparison-bar"><span style={{width:rate(c.success_rate??0)}}/></div><p>{c.confidence_interval?`95% interval: ${rate(c.confidence_interval[0])}–${rate(c.confidence_interval[1])}`:'No resolved outcomes yet'}</p><p>{c.recorded} / {c.planned} recorded · {c.unresolved} unresolved</p><p>{c.paired.delta===null?'No resolved seed pairs yet':`${c.paired.delta>=0?'+':''}${Math.round(c.paired.delta*100)} points vs ${c.paired.baseline} across ${c.paired.count} paired seeds`}</p>{c.error&&<p className="inline-error">{c.error}</p>}<button className="text-button" onClick={()=>onOpenRun(c.experiment_id)}>Inspect attempts <ArrowRight size={13}/></button></article>)}</div>
      <div className="comparison-table"><table><thead><tr><th>Seed</th>{current.conditions.map(c=><th key={c.experiment_id}>{c.name}</th>)}</tr></thead><tbody>{current.seeds.map(s=><tr key={s}><td>{s}</td>{current.conditions.map(c=>{const status=c.attempts.find(a=>a.seed===s)?.status||'pending';return <td key={c.experiment_id}><span className={'badge '+status}>{status}</span></td>;})}</tr>)}</tbody></table></div><p className="comparison-note">Rates and intervals use resolved pass/fail judgments. Unknown outcomes and errors remain visible and are excluded from those rates. Results retain the original evaluation even after a later rejudge. {current.note}</p>
    </section>}
  </div>;
}
