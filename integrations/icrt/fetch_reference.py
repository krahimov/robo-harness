"""Fetch four complete, fixed reference episodes using bounded HTTP range reads.

Run from the repository root with uv --with h5py --with numpy.
The pinned dataset, selection, and no-overwrite checks are intentional.
"""
import io,json,urllib.request,collections,time
from pathlib import Path
import numpy as np,h5py
URL='https://huggingface.co/datasets/Ravenh97/ICRT-MT/resolve/7eee568a981d2f58b5425beb2e0ea87eb9d30e1d/merged_data_part1.hdf5'
SIZE=44485934388
def open_range(request):
 for attempt in range(4):
  try:return urllib.request.urlopen(request,timeout=30)
  except (OSError,urllib.error.URLError):
   if attempt==3:raise
   time.sleep(attempt+1)

class Remote(io.RawIOBase):
 def __init__(self):
  self.pos=0;self.cache={};self.downloaded=0;self.deadline=time.monotonic()+900
  self.disk=Path('.data/validation/icrt-reference-1/range-cache');self.disk.mkdir(parents=True,exist_ok=True)
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset+(self.pos if whence==1 else SIZE if whence==2 else 0);return self.pos
 def readinto(self,b):
  data=self.read(len(b));b[:len(data)]=data;return len(data)
 def read(self,n=-1):
  if n<0:raise ValueError('Full download forbidden')
  end=min(SIZE,self.pos+n);chunks=[];block=1024*1024
  while self.pos<end:
   index=self.pos//block;start=index*block;last=min(SIZE-1,start+block-1)
   path=self.disk/str(index)
   if index not in self.cache:
    if path.exists() and path.stat().st_size==last-start+1:
     self.cache[index]=path.read_bytes()
    else:
     if self.downloaded+block>300*1024**2 or time.monotonic()>self.deadline:raise RuntimeError('Sample transfer budget exceeded')
     for attempt in range(4):
      req=urllib.request.Request(URL+f'?download=true&reference_block={index}&attempt={attempt}',headers={'Range':f'bytes={start}-{last}'})
      with open_range(req) as response:
       if response.status!=206:raise ValueError('Server did not honor Range')
       content_range=response.headers.get('Content-Range','')
       data=response.read(block+1)
       self.downloaded+=len(data)
      if content_range==f'bytes {start}-{last}/{SIZE}' and len(data)==last-start+1:break
      print('Retry incomplete range',index,content_range,len(data),flush=True)
     else:raise ValueError('Repeated incomplete range')
     self.cache[index]=data;path.write_bytes(data)
     if len(self.cache)%20==0:print('Cached MB',len(self.cache),'new bytes',self.downloaded,flush=True)
   data=self.cache[index];offset=self.pos-start;take=min(len(data)-offset,end-self.pos)
   if take<=0:raise ValueError(f'Empty range: position={self.pos}, block={index}, offset={offset}, bytes={len(data)}')
   chunks.append(data[offset:offset+take]);self.pos+=take
  return b''.join(chunks)

import hashlib
r=Remote();out=Path('.data/validation/icrt-reference-1');out.mkdir(exist_ok=True)
assert not list(out.glob('episode-*.npz')), 'Refusing to replace downloaded episodes'
manifest={'dataset_revision':'7eee568a981d2f58b5425beb2e0ea87eb9d30e1d','url':URL,'selection':'Fixed before prediction: episode 5 context; episodes 6,7,8 query. Separate episodes, not guaranteed unseen during training.','episodes':[]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
with h5py.File(r,'r') as f:
 for suffix in [5,6,7,8]:
  episode=f'real_episode_2024-05-31-tiger-to-black-bowl_{suffix}'
  g=f[episode]
  p=np.concatenate([g['observation/cartesian_position'][:],g['observation/gripper_position'][:]],axis=-1)
  a=np.concatenate([g['action/cartesian_position'][:],g['action/gripper_position'][:]],axis=-1)
  print('Fetching',episode,len(a),flush=True)
  assert len(p)==len(a)
  cameras={}
  for name,key in [('side','exterior_image_1_left'),('wrist','wrist_image_left')]:
   images=[]
   for raw in g['observation/'+key][:]:
    v=np.frombuffer(raw,dtype='uint8');assert v.size==180*320*3
    images.append(v.reshape(180,320,3))
   cameras[name]=np.array(images)
  dest=out/f'episode-{suffix}.npz'
  np.savez_compressed(dest,**cameras,proprio=p,actions=a,indices=np.arange(len(a)))
  manifest['episodes'].append({'id':episode,'file':dest.name,'frames':len(a),'role':'context' if suffix==5 else 'query','sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
  manifest['bytes_transferred']=r.downloaded
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
  print('Saved',dest.name,'bytes transferred',r.downloaded,flush=True)
print('Complete',flush=True)
