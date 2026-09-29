import {useEffect,useRef,useState} from 'react';
import {Check,ClipboardCheck,FileText,Loader2,Pause,Play,RefreshCw,Save,Sparkles} from 'lucide-react';
import Viewport from './Viewport';
import CalibrationSets from './CalibrationSets';
import {api,type Artifact,type Evaluation,type Judge} from './types';
import './judge-lab.css';

type OutcomeStatus='pass'|'fail'|'unknown';
type HumanLabel={id:string;status:OutcomeStatus;note:string};
type Task={requirements:{id:string;label?:string;definition:string;required:boolean}[]};
type ReviewItem={evaluation:Evaluation;episode:{id:string;artifact_id:string;index:number};task:Task;labels:HumanLabel[];reason:string};
type Queue={items:ReviewItem[];pending:number;sampling:string};
type Calibration={judge_id:string;judge_version:number;evaluation_count:number;label_count:number;labelled_evaluations:number;agreement:number|null;false_passes:number;false_pass_rate:number|null;human_unknown_labels:number;judge_unknowns:number;sampling_note:string;per_outcome:{id:string;label_count:number;agreement:number|null;false_passes:number}[]};
type Proposal={id:string;status:'draft'|'accepted';rubric:string;rationale:string;note:string;calibration:Calibration};
const percent=(value:number|null)=>value===null?'No labels':`${Math.round(value*100)}%`;
const pretty=(value:string)=>value.replaceAll('_',' ');

export default function JudgeLab({judges,onJudgesChanged}:{judges:Judge[];onJudgesChanged:()=>Promise<unknown>}){
  const [hideVerdicts,setHideVerdicts]=useState(true);
  const [judgeId,setJudgeId]=useState(judges.find(judge=>judge.kind==='vlm')?.id||judges[0]?.id||'');
  const [metrics,setMetrics]=useState<Calibration>();
  const [queue,setQueue]=useState<Queue>();
  const [proposals,setProposals]=useState<Proposal[]>([]);
  const [selected,setSelected]=useState<string>();
  const [artifact,setArtifact]=useState<Artifact>();
  const [frame,setFrame]=useState(0);
  const [playing,setPlaying]=useState(false);
  const [labels,setLabels]=useState<Record<string,{status:OutcomeStatus|'';note:string}>>({});
  const [proposalId,setProposalId]=useState('');
  const [rubric,setRubric]=useState('');
  const [busy,setBusy]=useState('');
  const [error,setError]=useState('');
  const [notice,setNotice]=useState('');
  const request=useRef(0);
  const recordingRequest=useRef(0);
  const activeJudge=judges.find(judge=>judge.id===judgeId);
  const item=queue?.items.find(item=>item.evaluation.id===selected);
  const proposal=proposals.find(item=>item.id===proposalId);

  async function refresh(){
    const token=++request.current;
    const [nextMetrics,nextQueue,nextProposals]=await Promise.all([
      api<Calibration>('/judges/'+judgeId+'/calibration'),api<Queue>('/judges/'+judgeId+'/review-queue?limit=12'),api<Proposal[]>('/judges/'+judgeId+'/rubric-proposals')]);
    if(token!==request.current)return;
    setMetrics(nextMetrics);setQueue(nextQueue);setProposals(nextProposals);
    setSelected(previous=>nextQueue.items.some(item=>item.evaluation.id===previous)?previous:nextQueue.items[0]?.evaluation.id);
  }
  async function perform(name:string,action:()=>Promise<void>){setBusy(name);setError('');setNotice('');try{await action();}catch(reason){setError((reason as Error).message);}finally{setBusy('');}}
  useEffect(()=>{if(!judges.some(judge=>judge.id===judgeId))setJudgeId(judges[0]?.id||'');},[judges,judgeId]);
  useEffect(()=>{
    setMetrics(undefined);setQueue(undefined);setProposals([]);setProposalId('');setRubric('');setError('');setNotice('');
    if(judgeId)refresh().catch(reason=>setError(reason.message));
    return()=>{request.current++;};
  },[judgeId]);
  useEffect(()=>{
    const token=++recordingRequest.current;setArtifact(undefined);setFrame(0);setPlaying(false);
    const selectedItem=queue?.items.find(item=>item.evaluation.id===selected);
    setLabels(Object.fromEntries((selectedItem?.task.requirements||[]).map(requirement=>{const previous=selectedItem?.labels.find(label=>label.id===requirement.id);return [requirement.id,{status:previous?.status||'',note:previous?.note||''}];})));
    if(selectedItem)api<Artifact>('/artifacts/'+selectedItem.episode.artifact_id).then(next=>{if(token===recordingRequest.current)setArtifact(next);}).catch(reason=>{if(token===recordingRequest.current)setError(reason.message);});
    return()=>{recordingRequest.current++;};
  },[selected,queue]);
  useEffect(()=>{if(!playing||!artifact)return;const timer=setInterval(()=>setFrame(previous=>{if(previous>=artifact.frames.length-1){setPlaying(false);return previous;}return previous+1;}),1000/artifact.frequency);return()=>clearInterval(timer);},[playing,artifact]);
  if(!judges.length)return null;

  return <section className="judge-lab">
    <div className="jl-heading"><div><span className="eyebrow">HUMAN CALIBRATION</span><h2><ClipboardCheck size={20}/>Compare judge decisions with your review</h2><p>Labels belong to an exact evaluation and rubric version. Corrections never overwrite the original decision.</p></div><button className="button small" disabled={!!busy} onClick={()=>perform('refresh',refresh)}><RefreshCw size={14}/>Refresh</button></div>
    <label className="field-label jl-selector">Judge version<select value={judgeId} disabled={!!busy} onChange={event=>setJudgeId(event.target.value)}>{judges.map(judge=><option key={judge.id} value={judge.id}>{judge.name} · v{judge.version} · {judge.kind==='state'?'state checks':judge.model}</option>)}</select></label>
    {error&&<p className="jl-error" role="alert">{error}</p>}{notice&&<p className="jl-notice" role="status"><Check size={15}/>{notice}</p>}
    {metrics&&<><div className="jl-metrics"><div><span>Human agreement</span><strong>{percent(metrics.agreement)}</strong><small>{metrics.label_count} resolved outcome labels</small></div><div><span>False passes</span><strong>{metrics.false_passes}</strong><small>Judge passed, human marked fail</small></div><div><span>Attempts labelled</span><strong>{metrics.labelled_evaluations}<small> / {metrics.evaluation_count}</small></strong><small>{metrics.human_unknown_labels} human labels unresolved</small></div><div><span>Awaiting labels</span><strong>{queue?.pending??0}</strong><small>This exact judge version</small></div></div><p className="jl-help">{metrics.sampling_note}</p>{metrics.per_outcome.length>0&&<details className="jl-outcome-metrics"><summary>Agreement by outcome</summary><table><thead><tr><th>Outcome</th><th>Labels</th><th>Agreement</th><th>False passes</th></tr></thead><tbody>{metrics.per_outcome.map(outcome=><tr key={outcome.id}><td>{pretty(outcome.id)}</td><td>{outcome.label_count}</td><td>{percent(outcome.agreement)}</td><td>{outcome.false_passes}</td></tr>)}</tbody></table></details>}</>}
    <CalibrationSets judgeId={judgeId} labelCount={metrics?.label_count||0} onReview={item=>{setQueue(q=>q?{...q,items:[item,...q.items.filter(x=>x.evaluation.id!==item.evaluation.id)]}:{items:[item],pending:1,sampling:'Frozen review set'});setSelected(item.evaluation.id);}}/><label className="checkbox-label"><input type="checkbox" checked={hideVerdicts} onChange={e=>setHideVerdicts(e.target.checked)}/>Hide judge verdicts while labeling</label><div className="jl-section-top"><h3>Review sample</h3><span>Uncertain attempts plus a reserved sample of passes</span></div>
    {queue?.items.length?<div className="jl-review-grid"><div className="jl-samples">{queue.items.map(sample=><button key={sample.evaluation.id} className={selected===sample.evaluation.id?'selected':''} disabled={!!busy} onClick={()=>setSelected(sample.evaluation.id)}><span><strong>Attempt {sample.episode.index}</strong><small>{hideVerdicts?'Review the recording before revealing the judge':sample.reason}</small></span>{!hideVerdicts&&<span className={'badge '+sample.evaluation.status}>{sample.evaluation.status}</span>}</button>)}</div><div className="jl-review-main">{item&&<>
      {artifact?<><Viewport frame={artifact.frames[frame]}/><div className="jl-playback"><button className="icon-button" aria-label={playing?'Pause review playback':'Play review playback'} onClick={()=>{if(frame===artifact.frames.length-1)setFrame(0);setPlaying(!playing);}}>{playing?<Pause size={17}/>:<Play size={17}/>}</button><input aria-label="Review recording timeline" type="range" min={0} max={artifact.frames.length-1} value={frame} onChange={event=>{setFrame(+event.target.value);setPlaying(false);}}/><span>{artifact.frames[frame]?.t.toFixed(2)}s</span></div></>:<div className="jl-loading"><Loader2 size={20} className="spin"/>Loading recording</div>}
      {!hideVerdicts&&<p className="jl-summary">{item.evaluation.summary}</p>}
      <form onSubmit={event=>{event.preventDefault();perform('labels',async()=>{const outcomes=Object.entries(labels).filter(([,label])=>label.status).map(([id,label])=>({id,...label}));await api('/evaluations/'+item.evaluation.id+'/labels',{outcomes});await refresh();setNotice('Human labels saved. The original judge decision is preserved.');});}}>
        {item.task.requirements.map(requirement=>{const predicted=item.evaluation.outcomes.find(outcome=>outcome.id===requirement.id);return <div className="jl-label-row" key={requirement.id}><div className="jl-label-heading"><strong>{requirement.label||pretty(requirement.id)}</strong>{!hideVerdicts&&<span className={'badge '+(predicted?.status||'unknown')}>Judge: {predicted?.status||'unknown'}</span>}</div><p>{requirement.definition}</p>{!hideVerdicts&&<small>{predicted?.explanation||'The judge did not return an outcome.'}</small>}<div className="jl-label-fields"><label className="field-label">Your judgment<select aria-label={'Your judgment for '+requirement.id} value={labels[requirement.id]?.status||''} disabled={!!busy} onChange={event=>setLabels(previous=>({...previous,[requirement.id]:{...previous[requirement.id],status:event.target.value as OutcomeStatus|''}}))}><option value="">Choose verdict</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="unknown">Unknown</option></select></label><label className="field-label">Evidence / correction<input aria-label={'Correction for '+requirement.id} value={labels[requirement.id]?.note||''} maxLength={2000} disabled={!!busy} placeholder="What did you observe?" onChange={event=>setLabels(previous=>({...previous,[requirement.id]:{...previous[requirement.id],note:event.target.value}}))}/></label></div></div>;})}
        <button type="submit" className="button primary" disabled={!!busy||!Object.values(labels).some(label=>label.status)}>{busy==='labels'?<Loader2 className="spin" size={15}/>:<Save size={15}/>}Save outcome labels</button>
      </form>
    </>}</div></div>:<div className="jl-empty"><ClipboardCheck size={23}/><p>{metrics?.evaluation_count?'All available outcomes for this version have human labels.':'Run an experiment or re-evaluate an attempt with this judge to start calibration.'}</p></div>}
    <div className="jl-section-top jl-proposal-top"><div><h3>Rubric proposals</h3><p>Use human corrections to draft inspection guidance. Review it before saving a new judge version.</p></div><button className="button" disabled={!!busy||activeJudge?.kind!=='vlm'||!metrics?.label_count} onClick={()=>perform('proposal',async()=>{const next=await api<Proposal>('/judges/'+judgeId+'/rubric-proposals',{});await refresh();setProposalId(next.id);setRubric(next.rubric);setNotice('Rubric draft generated. Review the changes before accepting.');})}>{busy==='proposal'?<Loader2 className="spin" size={15}/>:<Sparkles size={15}/>}Propose from corrections</button></div>
    {activeJudge?.kind==='state'&&<p className="jl-help">State checks measure task requirements directly. Choose a VLM judge to revise how a model inspects evidence.</p>}
    {!!proposals.length&&<div className="jl-proposals"><label className="field-label">Saved proposal<select value={proposalId} disabled={!!busy} onChange={event=>{setProposalId(event.target.value);setRubric(proposals.find(item=>item.id===event.target.value)?.rubric||'');}}><option value="">Select a proposal to inspect</option>{proposals.map((item,index)=><option key={item.id} value={item.id}>{item.status==='accepted'?'Accepted':'Draft'} · {item.calibration.label_count} source labels · {proposals.length-index}</option>)}</select></label>{proposal&&<><p>{proposal.rationale}</p><div className="jl-rubric-comparison"><div><h4><FileText size={14}/>Current rubric · v{activeJudge?.version}</h4><pre>{activeJudge?.rubric}</pre></div><label className="field-label"><span>Proposed rubric.md</span><textarea value={rubric} minLength={10} maxLength={16000} spellCheck={false} disabled={!!busy||proposal.status==='accepted'} onChange={event=>setRubric(event.target.value)}/></label></div><p className="jl-help">{proposal.note}</p>{proposal.status==='draft'&&<button className="button primary" disabled={!!busy||rubric.trim().length<10} onClick={()=>perform('accept',async()=>{const next=await api<Judge>('/rubric-proposals/'+proposal.id+'/accept',{rubric});await onJudgesChanged();await refresh();setNotice('Saved '+next.name+' v'+next.version+'. Run fresh evaluations to measure its agreement.');})}>{busy==='accept'?<Loader2 className="spin" size={15}/>:<Check size={15}/>}Accept as new version</button>}</>}</div>}
  </section>;
}
