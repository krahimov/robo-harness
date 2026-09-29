"""Fixed review samples with development/holdout separation and no invented labels."""
from fastapi import APIRouter,HTTPException
from studio import storage as db
from studio.calibration import latest_labels,evaluation_task
router=APIRouter(prefix='/api')


def detail(id):
    item=db.get('calibration_set',id);labels=latest_labels(item['judge_id']);evaluations={e['id']:e for e in db.all_records('evaluation')}
    parts={}
    for part,ids in item['splits'].items():
        rows=[];expected=0
        for eid in ids:
            evaluation=evaluations[eid];expected+=len(evaluation_task(evaluation)['requirements'])
            for outcome in evaluation_task(evaluation)['requirements']:
                label=labels.get((eid,outcome['id']))
                if label and label['status']!='unknown':
                    predicted=next((o['status'] for o in evaluation['outcomes'] if o['id']==outcome['id']),'unknown')
                    rows.append((label['status'],predicted))
        parts[part]={'evaluation_ids':ids,'resolved_labels':len(rows),'expected_labels':expected,
            'agreement':sum(a==b for a,b in rows)/len(rows) if rows else None,
            'false_passes':sum(a=='fail' and b=='pass' for a,b in rows),
            'complete':len(rows)==expected and expected>0}
    items=[{'evaluation':evaluations[eid],'episode':db.get('episode',evaluations[eid]['episode_id']),'task':evaluation_task(evaluations[eid]),'labels':[v for (e,_),v in labels.items() if e==eid],'reason':part+' calibration sample'} for part,ids in item['splits'].items() for eid in ids]
    return {**item,'items':items,'metrics':parts,'status':'labelled' if all(p['complete'] for p in parts.values()) else 'awaiting_human_labels'}


@router.get('/judges/{judge_id}/calibration-sets')
def list_sets(judge_id:str):
    db.get('judge',judge_id)
    return [detail(s['id']) for s in db.all_records('calibration_set') if s['judge_id']==judge_id]


@router.post('/judges/{judge_id}/calibration-sets',status_code=201)
def create_set(judge_id:str):
    judge=db.get('judge',judge_id)
    existing=db.all_records('calibration_set')
    used={eid for s in existing for ids in s['splits'].values() for eid in ids}
    labels=latest_labels(judge_id)
    pool=[e for e in db.all_records('evaluation') if e['judge']['id']==judge_id and e['id'] not in used
        and not any(eid==e['id'] for eid,_ in labels) and e['status'] in ('pass','fail','unknown')]
    passed=[e for e in pool if e['status']=='pass'];failed=[e for e in pool if e['status']=='fail'];unknown=[e for e in pool if e['status']=='unknown']
    # Interleave declared strata so both splits get passes and failures when available.
    chosen=[]
    for group in (failed,passed,unknown):chosen.extend(group[:2])
    if len(chosen)<4:raise HTTPException(409,'Need at least four unlabelled evaluations including varied verdicts to freeze a review sample.')
    ids=[e['id'] for e in chosen]
    item=db.save('calibration_set',{'judge_id':judge_id,'judge_version':judge['version'],
        'splits':{'development':ids[::2],'holdout':ids[1::2]},
        'sampling':'Up to two failures, two passes, two unknowns; no prior labels. Stratified diagnostic sample, not population accuracy.',
        'rule':'Holdout labels are excluded from automatic rubric proposals. Only user-entered labels count as human judgments.'})
    return detail(item['id'])


def reserved_holdouts(judge_id):
    return {eid for s in db.all_records('calibration_set') if s['judge_id']==judge_id for eid in s['splits']['holdout']}
