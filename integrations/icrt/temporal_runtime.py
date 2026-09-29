"""Experimental current-pose action ensemble; production runtime stays unchanged."""
from collections import deque
import numpy as np
import torch
from runtime import ICRTRuntime


class TemporalRuntime(ICRTRuntime):
    horizon = 10

    def reset(self, demonstrations):
        super().reset(demonstrations)
        self.chunks = deque(maxlen=self.horizon)

    def metadata(self):
        return {**super().metadata(), 'runtime_adapter_version':'icrt-temporal-diagnostic-v1',
            'supports_executed_action_feedback':False,
            'temporal_ensemble':True,'action_horizon':self.horizon,'action_reference':'current_observation_per_chunk'}

    def act(self, observation):
        from icrt.data.utils import convert_abs_action, convert_delta_action, rot_6d_to_euler
        if self.context_frames+self.steps+2>self.max_sequence_frames:
            raise ValueError('ICRT context capacity reached')
        with torch.inference_mode():
            current=self._prepare([observation])
            history={k:torch.cat([self.model.last_observation[k],v],dim=1) if self.model.last_observation is not None and v is not None else v for k,v in current.items()}
            history['action']=self.model.last_action
            predicted,_=self.model.forward_inference(history,self.model.start_pos)
            current_pose=current['proprio'].cpu().numpy()
            absolute=convert_abs_action(predicted.cpu().numpy(),current_pose)[0]
            self.chunks.append(deque(absolute[:self.horizon]))
            proposals=np.stack([q.popleft() for q in self.chunks])
            weights=np.exp(.05*np.arange(len(proposals)));weights/=weights.sum()
            action=(proposals*weights[:,None]).sum(axis=0)
            cached=convert_delta_action(action[None,None],current_pose).reshape(-1)
            self.model.last_action=torch.tensor(np.r_[cached,0.],device=self.device,dtype=torch.float32)[None,None]
            self.model.last_observation=current
            if self.model.first_obs:self.model.first_obs=False
            else:self.model.start_pos+=2
            self.steps+=1
            result=np.r_[action[:3],rot_6d_to_euler(action[3:9]).reshape(-1),action[-1]]
            if not np.isfinite(result).all():raise ValueError('Invalid ensemble output')
            return result.tolist()

if __name__=='__main__':
    import uvicorn
    from worker import create_app
    uvicorn.run(create_app(lambda:TemporalRuntime(device='cpu')),host='127.0.0.1',port=8791,log_level='warning')
