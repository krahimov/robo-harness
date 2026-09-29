"""Retain every simulated demonstration frame, with a bounded offline query."""
import argparse,json,time,hashlib
from pathlib import Path
from runtime import ICRTRuntime
from transfer_probe import load
from reference_eval import metrics

def run(folder):
 out=folder/'density';out.mkdir(exist_ok=False)
 full=load(folder/'context-full.npz');query=load(folder/'query.npz')
 count=len(full['frames']);assert count+len(query['frames'])+2<=512
 plan={'context_frames':count,'query_frames':len(query['frames']),'context_sha256':hashlib.sha256((folder/'context-full.npz').read_bytes()).hexdigest(),'kind':'offline complete context with no subsampling; increase only wrapper context allowance, not transformer capacity or weights; compare existing sampled-context probe on same query'}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));rt=ICRTRuntime(device='cpu');rt.max_context_frames=count;rt.reset([full]);start=time.monotonic()
 pred=[rt.act(f) for f in query['frames']]
 report={'plan':plan,'model':rt.metadata(),'metrics':metrics(pred,query['actions'],query['proprio']),'seconds':time.monotonic()-start,'predictions':pred};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['metrics']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder)
