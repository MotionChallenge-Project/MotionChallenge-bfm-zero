"""Compare converted GPU inference to the unmodified files in ONNX Runtime."""
import argparse,json,subprocess,tempfile
from pathlib import Path
import numpy as np,torch
from motion_challenge.validation import sha256_file
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--encoder',type=Path);p.add_argument('--weights',type=Path,required=True);p.add_argument('--ort-python',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();root=Path(__file__).resolve().parent;runtime=json.loads((root/'runtime.mjwarp.json').read_text());assert sha256_file(a.source)==runtime['weights_sha256'];torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(1);torch.manual_seed(9114);count=7
if a.encoder:
 from tensor_policy import TensorUFO
 assert sha256_file(a.encoder)==runtime['artifacts']['encoder']['sha256'];net=TensorUFO(a.weights,a.device,runtime['weights_sha256']);values={'actor_obs':torch.randn(count,721)*.2,'state':torch.randn(count,64)*.2,'privileged_state':torch.randn(count,463)*.2}
 with torch.inference_mode():expected={'action':net(values['actor_obs'].to(a.device)).cpu().numpy(),'latent':net.encode(values['state'].to(a.device),values['privileged_state'].to(a.device)).cpu().numpy()}
else:
 from tensor_policy import TensorBFM
 net=TensorBFM(a.weights,a.device,runtime['weights_sha256']);values={'state':torch.randn(count,64)*.2,'last_action':torch.randn(count,29)*.2,'history_actor':torch.randn(count,372)*.2,'reference_state':torch.randn(count,8,64)*.2,'reference_privileged':torch.randn(count,8,463)*.2,'weights':torch.ones(count,8,1)}
 with torch.inference_mode():out=net(*(x.to(a.device) for x in values.values()));expected={'action':out[0].cpu().numpy(),'latent':out[1].cpu().numpy()}
code='''import sys,numpy as np,onnxruntime as ort
source,encoder,inputs,output=sys.argv[1:];v=np.load(inputs);options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
actor=ort.InferenceSession(source,options,providers=['CPUExecutionProvider']);rows=[];latents=[]
if encoder!='none':
 enc=ort.InferenceSession(encoder,options,providers=['CPUExecutionProvider'])
 for i in range(7):
  rows.append(actor.run(None,{'actor_obs':v['actor_obs'][i:i+1]})[0]);latents.append(enc.run(None,{'state':v['state'][i:i+1],'privileged_state':v['privileged_state'][i:i+1]})[0])
else:
 for i in range(7):
  feed={k:v[k][i] if k.startswith('reference_') or k=='weights' else v[k][i:i+1] for k in v.files};out=actor.run(None,feed);rows.append(out[0]);latents.append(out[1])
np.savez(output,action=np.concatenate(rows),latent=np.concatenate(latents),version=np.array(ort.__version__))
'''
with tempfile.TemporaryDirectory() as temporary:
 d=Path(temporary);np.savez(d/'inputs.npz',**{k:v.numpy() for k,v in values.items()});subprocess.run([a.ort_python,'-c',code,str(a.source),str(a.encoder) if a.encoder else 'none',str(d/'inputs.npz'),str(d/'outputs.npz')],check=True);actual=np.load(d/'outputs.npz');errors={}
 for key,value in expected.items():errors[key]=float(np.max(abs(value-actual[key])));np.testing.assert_allclose(value,actual[key],atol=3e-5,rtol=3e-5)
 report={'status':'passed','onnxruntime_version':str(actual['version']),'scalar_inputs':count,'gpu_batch':count,'max_abs_errors':errors,'source_sha256':sha256_file(a.source),'encoder_sha256':sha256_file(a.encoder) if a.encoder else None}
(root/'validation/actual-onnxruntime-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
