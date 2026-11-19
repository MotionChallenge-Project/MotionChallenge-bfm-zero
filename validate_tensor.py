import sys,types,json,argparse
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
sys.path.insert(0,str(root));sys.modules.setdefault('onnxruntime',types.ModuleType('onnxruntime'))
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
from tensor_adapter import TensorBFMZero
from tensor_policy import TensorBFM
from adapter import BFMZero
from motion_challenge.resources import resource_root
from motion_challenge.simulation.physics import World
from motion_challenge.data.reference import Motion
from motion_challenge.simulation.tensor_reference import TensorMotionBank,TensorReference
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket
meta=json.loads((a.prepared/'dataset.json').read_text());poses=np.load(a.prepared/'poses.npy',mmap_mode='r');runtime=json.loads((root/'runtime.mjwarp.json').read_text())
worlds=[World(resource_root()/'robots/g1-29dof-mjwarp-v1.json') for _ in range(7)];names=worlds[0].joint_names
for clip,w in enumerate(worlds):
 x=poses[meta['starts'][clip]:meta['starts'][clip]+meta['lengths'][clip]];w.reset(Motion(names,tuple(np.arange(len(x))/meta['fps']),x[:,7:],x[:,:3],x[:,3:7]))
bank=TensorMotionBank(np.array(poses),meta['starts'],meta['lengths'],np.full(len(meta['names']),meta['fps']),meta['names'],a.device)
refs=TensorReference(worlds[0],bank,147,a.device);adapter=TensorBFMZero(root,runtime,names,7,a.device,a.weights);oracle=TensorBFM(a.weights,'cpu',runtime['weights_sha256']).eval()
old=[]
for _ in range(7):
 o=BFMZero.__new__(BFMZero);o.cfg=json.loads((root/'policy.json').read_text());o.default=np.asarray(o.cfg['default_joint_pos']);old.append(o)
rng=np.random.default_rng(1931);maxima=np.zeros(6);network=0.
for step in range(15):
 times=np.arange(7)[:,None]*.03+step*.02+np.array(runtime['reference_offsets_s'])
 if step==14:times=np.array([(meta['lengths'][i]-1)/meta['fps'] for i in range(7)])[:,None]+np.array(runtime['reference_offsets_s'])
 reference=refs.motion(torch.arange(7,device=a.device),torch.tensor(times,device=a.device))
 q=rng.normal(0,.2,(7,29));qd=rng.normal(0,.4,(7,29));quat=rng.normal(size=(7,4));quat/=np.linalg.norm(quat,axis=1,keepdims=True);omega=rng.normal(0,.1,(7,3))
 state=DeployableStateBatch(names,torch.full((7,),step*.02,device=a.device),*[torch.tensor(x,device=a.device) for x in (q,qd,quat,omega)])
 if step==8:adapter.reset(torch.tensor([1,5],device=a.device))
 actual=adapter.actor_observation(state)+adapter.reference_observations(reference);expected=[[] for _ in actual];packets=[]
 for row,(world,o) in enumerate(zip(worlds,old)):
  packet=DeployableStatePacket(step*.02,names,tuple(q[row]),tuple(qd[row]),tuple(quat[row]),tuple(omega[row]));packets.append(packet);frames=world.references(times[row])
  if step==0 or (step==8 and row in (1,5)):o.reset(packet,frames,0)
  obs=o.actor_observation(packet);encoded=[o.reference_observation(cur,prev) for prev,cur in zip(frames[:-1],frames[1:])];weights=np.ones((8,1),np.float32)
  for i in range(1,8):
   if frames[i+1].source_time_s<=frames[i].source_time_s+1e-10:weights[i]=0
  values=[obs['state'][0],obs['last_action'][0],obs['history_actor'][0],np.stack([v[0] for v in encoded]),np.stack([v[1] for v in encoded]),weights]
  for j,x in enumerate(values):expected[j].append(x)
 for j in range(6):
  e=np.stack(expected[j]);x=actual[j].cpu().numpy();maxima[j]=max(maxima[j],float(np.max(abs(x-e))));np.testing.assert_allclose(x,e,atol=5e-6,rtol=5e-6)
 with torch.inference_mode():
  result=adapter.actor(*actual)[0].cpu();wanted=torch.cat([oracle(*(torch.tensor(np.asarray(expected[j][i]))[None] for j in range(6)))[0] for i in range(7)])
 network=max(network,float((result-wanted).abs().max()));np.testing.assert_allclose(result.numpy(),wanted.numpy(),atol=3e-5,rtol=3e-5)
 adapter.actor_observation(state);adapter.last.copy_(wanted.to(a.device)*adapter.cfg['action_rescale'])
 for row,o in enumerate(old):o.actor_observation(packets[row]);o.last=(wanted[row].numpy()*o.cfg['action_rescale']).astype(np.float32)
report={'status':'passed','steps':15,'batch':7,'partial_reset_slots':[1,5],'includes_clamped_endpoints':True,'observation_max_abs_errors':maxima.tolist(),'network_max_abs_error':network,'reference_fk_oracle':'CPU MuJoCo kinematics and SciPy quaternion finite differences'}
(root/'validation/tensor-observation-parity.json').write_text(json.dumps(report,indent=2)+'\n');runtime['tensor_entrypoint']='tensor_adapter:TensorBFMZero';runtime['tensor_conversion']='convert_weights:prepare';(root/'runtime.mjwarp.json').write_text(json.dumps(runtime,indent=2)+'\n');print(json.dumps(report),flush=True)
