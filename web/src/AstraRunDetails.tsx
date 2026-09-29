import type {Experiment} from './types';

export default function AstraRunDetails({run}:{run:Experiment}) {
  if(run.adapter!=='astra')return null;
  if(run.fault_injection)return <div className="astra-run-details"><strong>Controlled recovery diagnostic</strong><p>{run.fault_injection}</p><p>One recorded episode. No development search or final holdout batch was run. Inspect the episode’s state checks and camera checkpoint evidence.</p></div>;
  return <div className="astra-run-details">
    <div className="astra-run-summary" role="status">
      <strong>{run.phase?.replaceAll('_',' ')||'Queued'}</strong>
      <span>{run.message||'Waiting for the actor'}</span>
      <span>Development: {run.attempts?.length||0}/{run.max_attempts||3} proposals</span>
      <span>Final evaluation: {run.holdout_passes||0} passes / {run.holdout_completed||0} completed / {run.episodes} planned</span>
    </div>
    <p className="field-help">{run.feedback_mode==='checkpoints'?'Camera checkpoints can revise remaining motions within each episode. The starting program and feedback rules stay frozen for final evaluation.':'Astra revises only the motion program between development attempts.'} Task, scene, context and judge stay fixed. Final cases never feed back into development or other episodes. Human acceptance is separate.</p>
    {run.outcome==='budget_exhausted'&&<p>No development program passed both checks within this budget. Final evaluation was not run.</p>}
    {!!run.attempts?.length&&<details open><summary>Development proposals</summary><ol className="astra-proposals">{run.attempts.map(a=><li key={a.number}>
      <div><strong>Proposal {a.number} · {a.status}</strong><span>{a.summary?`Model proposal: ${a.summary}`:null}</span></div>
      {a.proposal_artifact_id&&<a href={'/api/artifacts/'+a.proposal_artifact_id} target="_blank" rel="noreferrer">Saved model response</a>}
      {!!a.program&&<details><summary>Motion program</summary><pre>{JSON.stringify(a.program,null,2)}</pre></details>}
    </li>)}</ol></details>}
    {!!run.frozen_program&&<details><summary>Frozen program and final seeds</summary><p>These seeds are excluded from development feedback. Further tuning on their results would make them development cases.</p><code>{run.holdout_seeds?.join(', ')}</code><pre>{JSON.stringify(run.frozen_program,null,2)}</pre></details>}
  </div>;
}
