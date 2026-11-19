
"""Extract pinned BFM-Zero ONNX tensors and validate environment batching."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from tensor_policy import TensorBFM
from motion_challenge.validation import sha256_file


def prepare(root,runtime,source,output,artifacts,device):
    import onnx
    from onnx import numpy_helper
    from onnx.reference import ReferenceEvaluator
    root,source,output=Path(root),Path(source),Path(output)
    if sha256_file(source)!=runtime['weights_sha256']:raise ValueError('Source checkpoint mismatch')
    network_hash=sha256_file(root/'tensor_policy.py')
    if (output/'conversion.json').exists() and (output/'parity.json').exists():
        receipt=json.loads((output/'conversion.json').read_text())
        if receipt.get('source_sha256')==runtime['weights_sha256'] and receipt.get('network_sha256')==network_hash and sha256_file(output/'weights.safetensors')==receipt['weights_sha256']:
            return output
    model=onnx.load(source);output.mkdir(parents=True,exist_ok=True)
    weights={t.name:torch.from_numpy(numpy_helper.to_array(t).copy()) for t in model.graph.initializer}
    save_file(weights,str(output/'weights.safetensors'))
    receipt={'source_sha256':runtime['weights_sha256'],'weights_sha256':sha256_file(output/'weights.safetensors'),'network_sha256':network_hash,'environment_batch':'dynamic','reference_targets':8}
    (output/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    oracle=ReferenceEvaluator(model);gpu=TensorBFM(output,device,runtime['weights_sha256'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(1)
    rng=np.random.default_rng(2137);rows=[];maximum=0.
    for n in (1,7,40,256):
        feed={k:rng.normal(size=(n,*shape)).astype(np.float32) for k,shape in [('state',(64,)),('last_action',(29,)),('history_actor',(372,)),('reference_state',(8,64)),('reference_privileged',(8,463))]}
        feed['weights']=rng.integers(0,2,size=(n,8,1)).astype(np.float32);feed['weights'][:,0]=1
        expected=[]
        for i in range(n):
            single={k:(v[i] if k in ['reference_state','reference_privileged','weights'] else v[i:i+1]) for k,v in feed.items()}
            expected.append(oracle.run(['action','latent'],single))
        target=[np.concatenate([x[j] for x in expected]) for j in range(2)]
        with torch.inference_mode():actual=gpu(**{k:torch.tensor(v,device=device) for k,v in feed.items()})
        errors=[]
        for x,y in zip(actual,target):
            x=x.cpu().numpy();np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-5)
            errors.append(float(np.max(abs(x-y))))
        maximum=max(maximum,*errors);rows.append({'batch':n,'action_max_abs_error':errors[0],'latent_max_abs_error':errors[1]});print(rows[-1],flush=True)
    report={'status':'passed','source_sha256':runtime['weights_sha256'],'weights_sha256':receipt['weights_sha256'],'max_abs_error':maximum,'cases':rows,'reference':'Original ONNX per environment, including endpoint target masks'}
    (output/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    return output


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();root=Path(__file__).resolve().parent
    prepare(root,json.loads((root/'runtime.mjwarp.json').read_text()),a.source,a.output,{},a.device)

