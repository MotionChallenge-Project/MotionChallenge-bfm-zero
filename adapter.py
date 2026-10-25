"""BFM-Zero official minimal-inference encoding and frozen actor/backward map.

The source runtime's duplicated history insertion and local finite-difference
angular-velocity convention are preserved explicitly, including in regression
checks. This adapter owns neither dataset loading nor benchmark physics.
"""
from motion_challenge.adapters.onnx import cpu_session
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file


def privileged_encoding(positions,quaternions,previous_positions,previous_quaternions,dt):
    positions=np.asarray(positions);rot=R.from_quat(quaternions)
    root=positions[0]
    x,y,z,w=np.asarray(quaternions)[0]
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    heading=R.from_rotvec([0,0,-yaw])
    relative=heading.apply(positions-root)[1:].reshape(-1)
    matrices=(heading*rot).as_matrix()
    rotation=np.concatenate([matrices[:,:,0],matrices[:,:,2]],axis=-1).reshape(-1)
    if dt<=1e-10:
        velocity=np.zeros_like(positions);angular=np.zeros_like(positions)
    else:
        velocity=(positions-previous_positions)/dt
        angular=(R.from_quat(previous_quaternions).inv()*rot).as_rotvec()/dt
    return np.r_[root[2],relative,rotation,heading.apply(velocity).reshape(-1),heading.apply(angular).reshape(-1)].astype(np.float32)


class BFMZero:
    def __init__(self,root,config):
        self.cfg=json.loads((root/'policy.json').read_text())
        if sha256_file(root/'policy.json')!=config['inference_config_sha256']:raise ValueError('BFM config hash mismatch')
        weights=root/config['weights']
        if sha256_file(weights)!=config['weights_sha256']:raise ValueError('BFM checkpoint hash mismatch')
        self.session=cpu_session(weights,config['weights_sha256'])
        self.default=np.asarray(self.cfg['default_joint_pos'])

    def reset(self,state,reference,seed):
        self.order=[state.joint_names.index(n) for n in self.cfg['joint_names']]
        self.inverse=[self.cfg['joint_names'].index(n) for n in state.joint_names]
        self.histories=[np.zeros((4,n),np.float32) for n in (29,3,29,29,3)]
        self.last=np.zeros(29,np.float32)

    def actor_observation(self,state):
        q=np.asarray(state.joint_position_rad)[self.order]-self.default
        qd=np.asarray(state.joint_velocity_rad_s)[self.order]
        inv=R.from_quat(state.root_orientation_xyzw).inv()
        gravity=inv.apply([0,0,-1]);omega=.25*inv.apply(state.root_angular_velocity_world_rad_s)
        terms=[self.last,omega,q,qd,gravity]
        self.histories=[np.concatenate([np.asarray(v,np.float32)[None],h[:-1]],axis=0) for h,v in zip(self.histories,terms)]
        return {'state':np.r_[q,qd,gravity,omega][None].astype(np.float32),'last_action':self.last[None].copy(),
                'history_actor':np.concatenate([h.reshape(-1) for h in self.histories])[None]}

    def body_arrays(self,frame):
        bodies={b.name:b for b in frame.bodies};positions=[];quats=[]
        for name in self.cfg['body_names']:
            if name=='head_link':
                b=bodies[self.cfg['head_parent']]
                pos=np.asarray(b.position_world_m)+R.from_quat(b.orientation_xyzw).apply(self.cfg['head_position_parent_m'])
            else:
                b=bodies[name];pos=b.position_world_m
            positions.append(pos);quats.append(b.orientation_xyzw)
        return np.asarray(positions),np.asarray(quats)

    def reference_observation(self,frame,previous):
        dt=frame.source_time_s-previous.source_time_s
        q=np.asarray(frame.joint_position_rad)[self.order]
        if dt<=1e-10:
            qd=np.zeros(29);omega=np.zeros(3)
        else:
            qd=(q-np.asarray(previous.joint_position_rad)[self.order])/dt
            omega=(R.from_quat(previous.root_orientation_xyzw).inv()*R.from_quat(frame.root_orientation_xyzw)).as_rotvec()/dt
        gravity=R.from_quat(frame.root_orientation_xyzw).inv().apply([0,0,-1])
        pos,quat=self.body_arrays(frame);oldpos,oldquat=self.body_arrays(previous)
        return np.r_[q-self.default,qd,gravity,omega].astype(np.float32),privileged_encoding(pos,quat,oldpos,oldquat,dt)

    def act(self,state,reference):
        obs=self.actor_observation(state)
        encoded=[self.reference_observation(cur,prev) for prev,cur in zip(reference[:-1],reference[1:])]
        weights=np.ones((8,1),np.float32)
        for i in range(1,8):
            if reference[i+1].source_time_s<=reference[i].source_time_s+1e-10:weights[i]=0
        obs.update(reference_state=np.stack([v[0] for v in encoded]),reference_privileged=np.stack([v[1] for v in encoded]),weights=weights)
        action=self.session.run(['action'],obs)[0].reshape(29)
        if not np.isfinite(action).all():raise ValueError('nonfinite BFM action')
        # Native env.step calls _create_observation again before applying action.
        self.actor_observation(state)
        self.last=(action*self.cfg['action_rescale']).astype(np.float32)
        # Preserve the author's float32 operation order before PD conversion.
        target=(action*np.asarray(self.cfg['action_scales'],np.float32)*self.cfg['action_rescale']
                + self.default.astype(np.float32))
        def order(x):return tuple(float(v) for v in np.asarray(x)[self.inverse])
        return ControlCommand(order(target),(0.,)*29,order(self.cfg['kp']),order(self.cfg['kd']),(0.,)*29)

    def close(self):self.session=None


def create(root,config):return BFMZero(root,config)
