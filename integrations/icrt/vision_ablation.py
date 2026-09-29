"""Frozen reference-image interventions; offline diagnostic, not task success."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from reference_eval import load_episode,metrics,validate_split
from runtime import ICRTRuntime,ROOT


def intervene(episode,mode):
    frames=[]
    for frame in episode['frames']:
        f=dict(frame)
        if mode=='blank':
            f['side']=Image.new('RGB',frame['side'].size,(128,128,128))
            f['wrist']=Image.new('RGB',frame['wrist'].size,(128,128,128))
        elif mode=='swap':f['side'],f['wrist']=frame['wrist'],frame['side']
        frames.append(f)
    return {**episode,'frames':frames}


def run(output):
    folder=ROOT/'.data/validation/icrt-reference-1'
    manifest=json.loads((folder/'manifest.json').read_text());context_item,queries=validate_split(manifest)
    output.mkdir(parents=True,exist_ok=False)
    conditions=[('real','real','real'),('query_blank','real','blank'),('both_blank','blank','blank'),('both_swapped','swap','swap')]
    plan={'conditions':conditions,'queries':[q['id'] for q in queries], 'context':context_item['id'],
        'kind':'offline controlled camera intervention; proprioception/actions/runtime fixed; own predicted action history',
        'limits':'Not a simulation-transfer or task-success score. Vision changes can affect subsequent predicted-action history.'}
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    context=load_episode(folder,context_item);runtime=ICRTRuntime(device='cpu');results=[]
    for q in queries:
        query=load_episode(folder,q)
        for name,cm,qm in conditions:
            start=time.monotonic();runtime.reset([intervene(context,cm)])
            predictions=[runtime.act(f) for f in intervene(query,qm)['frames']]
            row={'query':q['id'],'condition':name,'metrics':metrics(predictions,query['actions'],query['proprio']),'seconds':time.monotonic()-start}
            (output/f"{q['id']}-{name}.json").write_text(json.dumps({**row,'predictions':predictions}))
            results.append(row);(output/'report.json').write_text(json.dumps({'plan':plan,'model':runtime.metadata(),'results':results},indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
