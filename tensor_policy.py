"""BFM-Zero frozen backward map and actor with a dynamic environment axis."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from motion_challenge.validation import sha256_file


class TensorBFM(torch.nn.Module):
    def __init__(self,directory,device='cpu',expected_source=None):
        super().__init__();directory=Path(directory)
        receipt=json.loads((directory/'conversion.json').read_text())
        if expected_source and receipt['source_sha256']!=expected_source:raise ValueError('Source checksum mismatch')
        if sha256_file(directory/'weights.safetensors')!=receipt['weights_sha256']:raise ValueError('Converted checksum mismatch')
        self.keys={}
        for index,(name,value) in enumerate(load_file(str(directory/'weights.safetensors'),device=device).items()):
            key='w'+str(index);self.register_buffer(key,value);self.keys[name]=key

    def weight(self,key):return getattr(self,self.keys[key])
    def linear(self,x,key):return F.linear(x,self.weight(key+'.weight'),self.weight(key+'.bias'))
    def layer_norm(self,x,key):
        w=self.weight(key+'.weight');return F.layer_norm(x,(w.numel(),),w,self.weight(key+'.bias'),1e-5)
    def normalizer(self,x,key):
        key='_obs_normalizer._normalizers.'+key+'._normalizer'
        return (x-self.weight(key+'.running_mean'))/torch.sqrt(self.weight(key+'.running_var')+1e-5)
    def block(self,x,key,activate=True):
        x=self.linear(self.layer_norm(x,key+'.0'),key+'.1')
        return x*torch.tanh(F.softplus(x,threshold=float('inf'))) if activate else x

    def encode(self,state,privileged):
        x=torch.cat((self.normalizer(state,'state'),self.normalizer(privileged,'privileged_state')),dim=-1)
        x=self.linear(x,'_backward_map.net.0')
        x=torch.tanh(self.layer_norm(x,'_backward_map.net.1'))
        return F.normalize(self.linear(x,'_backward_map.net.3'),dim=-1,eps=1e-12)*16.

    def actor(self,state,last_action,history,latent):
        s=torch.cat((self.normalizer(state,'state'),self.normalizer(last_action,'last_action'),self.normalizer(history,'history_actor')),dim=-1)
        z=torch.cat((s,latent),dim=-1)
        for i in range(2):z=self.block(z,f'_actor.embed_z.{i}.mlp')
        for i in range(2):s=self.block(s,f'_actor.embed_s.{i}.mlp')
        x=torch.cat((s,z),dim=-1)
        for i in range(6):x=x+self.block(x,f'_actor.policy.{i}.mlp')
        return torch.tanh(self.block(x,'_actor.policy.6.mlp',False))

    def forward(self,state,last_action,history_actor,reference_state,reference_privileged,weights):
        z=self.encode(reference_state,reference_privileged)
        z=(z*weights).sum(dim=1)/weights.sum(dim=1)
        latent=F.normalize(z,dim=-1,eps=1e-12)*16.
        return self.actor(state,last_action,history_actor,latent),latent
