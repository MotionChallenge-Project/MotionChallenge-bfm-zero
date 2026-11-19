"""Adapter-owned G1 reference kinematics in double precision."""
import json
from pathlib import Path
import torch
from motion_challenge.simulation.tensor_math import normalize, multiply, rotate


class ReferenceKinematics:
    def __init__(self,path,joint_names,device):
        self.device=torch.device(device);spec=json.loads(Path(path).read_text())
        self.names=tuple(spec['body_names']);self.parents=torch.tensor(spec['parents'],device=device)
        self.translation=torch.tensor(spec['translation'],device=device,dtype=torch.float64)
        self.rotation=torch.tensor(spec['rotation_xyzw'],device=device,dtype=torch.float64)
        self.axes=torch.tensor(spec['axes'],device=device,dtype=torch.float64)
        indices=[joint_names.index(x) if x else 0 for x in spec['joints']]
        self.joints=torch.tensor(indices,device=device)
        self.movable=torch.tensor([bool(x) for x in spec['joints']],device=device)
        self.levels=[torch.tensor(x,device=device) for x in spec['levels']]

    def compute(self,reference):
        shape=reference.joint_position_rad.shape[:-1];n=reference.joint_position_rad.numel()//reference.joint_position_rad.shape[-1]
        joints=reference.joint_position_rad.reshape(n,-1).double()
        jvel=reference.joint_velocity_rad_s.reshape(n,-1).double()
        count=len(self.names)
        pos=torch.zeros((n,count,3),device=self.device,dtype=torch.float64)
        quat=torch.zeros((n,count,4),device=self.device,dtype=torch.float64)
        linear=torch.zeros_like(pos);angular=torch.zeros_like(pos)
        pos[:,0]=reference.root_position_world_m.reshape(n,3)
        quat[:,0]=normalize(reference.root_orientation_xyzw.reshape(n,4).double())
        linear[:,0]=reference.root_linear_velocity_world_m_s.reshape(n,3)
        angular[:,0]=reference.root_angular_velocity_world_rad_s.reshape(n,3)
        for rows in self.levels:
            parents=self.parents[rows];pq=quat[:,parents]
            offset=rotate(pq,self.translation[rows])
            pos[:,rows]=pos[:,parents]+offset
            linear[:,rows]=linear[:,parents]+torch.linalg.cross(angular[:,parents],offset)
            base=multiply(pq,self.rotation[rows][None].expand_as(pq))
            half=joints[:,self.joints[rows]]*self.movable[rows]/2
            delta=torch.cat((self.axes[rows]*half.sin()[...,None],half.cos()[...,None]),dim=-1)
            quat[:,rows]=multiply(base,delta)
            angular[:,rows]=angular[:,parents]+rotate(base,self.axes[rows])*jvel[:,self.joints[rows],None]*self.movable[rows,None]
        return tuple(x.reshape(*shape,*x.shape[1:]) for x in (pos,quat,linear,angular))
