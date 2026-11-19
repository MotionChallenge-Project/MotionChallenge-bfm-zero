"""BFM-Zero GPU histories and adapter-owned reference body encoding."""
import json
from pathlib import Path
import torch
from tensor_policy import TensorBFM
from reference_kinematics import ReferenceKinematics
from motion_challenge.simulation.tensor_math import normalize,conjugate,rotate,matrix,quaternion_velocity


class TensorBFMZero:
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        self.device=torch.device(device);root=Path(root);self.cfg=json.loads((root/'policy.json').read_text())
        self.order=[joint_names.index(n) for n in self.cfg['joint_names']]
        self.inverse=[self.cfg['joint_names'].index(n) for n in joint_names]
        self.default=torch.tensor(self.cfg['default_joint_pos'],device=device,dtype=torch.float64)
        self.kp=torch.tensor(self.cfg['kp'],device=device,dtype=torch.float32)[self.inverse]
        self.kd=torch.tensor(self.cfg['kd'],device=device,dtype=torch.float32)[self.inverse]
        self.scale=torch.tensor(self.cfg['action_scales'],device=device,dtype=torch.float32)
        self.actor=TensorBFM(converted_weights,device,runtime['weights_sha256']).eval()
        self.fk=ReferenceKinematics(root/'reference_model.json',joint_names,device)
        self.body_ids=[self.fk.names.index(n) for n in self.cfg['body_names'] if n!='head_link']
        self.head_parent=self.fk.names.index(self.cfg['head_parent'])
        self.head_offset=torch.tensor(self.cfg['head_position_parent_m'],device=device,dtype=torch.float64)
        self.histories=[torch.zeros((slots,4,n),device=device) for n in (29,3,29,29,3)]
        self.last=torch.zeros((slots,29),device=device)

    def reset(self,slots):
        self.last[slots]=0
        for h in self.histories:h[slots]=0

    def actor_observation(self,state):
        q=state.joint_position_rad[:,self.order].double()-self.default
        qd=state.joint_velocity_rad_s[:,self.order].double()
        inv=conjugate(normalize(state.root_orientation_xyzw.double()))
        down=torch.tensor([0.,0.,-1.],device=self.device,dtype=torch.float64)
        gravity=rotate(inv,down);omega=.25*rotate(inv,state.root_angular_velocity_world_rad_s.double())
        for history,value in zip(self.histories,(self.last,omega,q,qd,gravity)):
            history[:,1:]=history[:,:-1].clone();history[:,0]=value.float()
        return torch.cat((q,qd,gravity,omega),dim=-1).float(),self.last.clone(),torch.cat([h.flatten(1) for h in self.histories],dim=-1)

    def reference_observations(self,reference):
        pos,quat,_,_=self.fk.compute(reference)
        head=pos[:,:,self.head_parent]+rotate(quat[:,:,self.head_parent],self.head_offset)
        pos=torch.cat((pos[:,:,self.body_ids],head[:,:,None]),dim=-2)
        quat=torch.cat((quat[:,:,self.body_ids],quat[:,:,self.head_parent,None]),dim=-2)
        dt=reference.source_time_s[:,1:]-reference.source_time_s[:,:-1];valid=dt>1e-10
        safe=torch.where(valid,dt,1.)
        q=reference.joint_position_rad[:,:,self.order].double()
        qd=(q[:,1:]-q[:,:-1])/safe[...,None]*valid[...,None]
        roots=normalize(reference.root_orientation_xyzw.double())
        omega=quaternion_velocity(roots[:,:-1],roots[:,1:],safe)*valid[...,None]
        down=torch.tensor([0.,0.,-1.],device=self.device,dtype=torch.float64)
        gravity=rotate(conjugate(roots[:,1:]),down)
        state=torch.cat((q[:,1:]-self.default,qd,gravity,omega),dim=-1).float()
        current=pos[:,1:];rot=normalize(quat[:,1:]);root=current[:,:,0]
        x,y,z,w=rot[:,:,0].unbind(-1)
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));zero=torch.zeros_like(yaw)
        heading=torch.stack((zero,zero,(-yaw/2).sin(),(-yaw/2).cos()),dim=-1)
        relative=rotate(heading[:,:,None],current-root[:,:,None])[:,:,1:].flatten(2)
        from motion_challenge.simulation.tensor_math import multiply
        matrices=matrix(multiply(heading[:,:,None],rot))
        rotation=torch.cat((matrices[...,0],matrices[...,2]),dim=-1).flatten(2)
        velocity=(current-pos[:,:-1])/safe[...,None,None]*valid[...,None,None]
        angular=quaternion_velocity(normalize(quat[:,:-1]),rot,safe[...,None])*valid[...,None,None]
        privileged=torch.cat((root[...,2:3],relative,rotation,rotate(heading[:,:,None],velocity).flatten(2),rotate(heading[:,:,None],angular).flatten(2)),dim=-1).float()
        weights=torch.ones((*dt.shape,1),device=self.device)
        weights[:,1:,0]=valid[:,1:]
        return state,privileged,weights

    @torch.inference_mode()
    def act(self,state,reference):
        state_obs,last,history=self.actor_observation(state)
        rs,rp,weights=self.reference_observations(reference)
        action,_=self.actor(state_obs,last,history,rs,rp,weights)
        self.actor_observation(state)
        self.last.copy_(action*self.cfg['action_rescale'])
        target=(action*self.scale*self.cfg['action_rescale']+self.default.float())[:,self.inverse]
        zero=torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1)
