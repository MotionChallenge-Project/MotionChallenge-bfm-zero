"""Export the author actor, backward map and normalizer without training."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from motion_challenge.entrypoint import artifact
from motion_challenge.validation import sha256_file


def load_sources(source):
    manifest = json.loads(Path(__file__).with_name('sources.json').read_text())
    source = Path(source)
    for name, item in manifest['source_files'].items():
        if name.endswith('.py'):
            artifact(source/name, item['sha256'], item['url'])
    return manifest


def build_inference(source, weights):
    """Instantiate only the inference modules from verified, unmodified source."""
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    sys.path.insert(0, str(Path(source).resolve()))
    from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModelConfig
    from bfm_zero_inference_code.misc.gym_spaces import json_to_space

    raw = json.loads((weights/'config.json').read_text())
    raw['device'] = 'cpu'
    cfg = FBcprAuxModelConfig.model_validate(raw)
    kw = json.loads((weights/'init_kwargs.json').read_text())
    space = json_to_space(kw['obs_space'])
    assert cfg.seq_length == 8 and cfg.archi.norm_z and not cfg.amp

    class Inference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = cfg.archi.actor.build(space, cfg.archi.z_dim, kw['action_dim'])
            self._backward_map = cfg.archi.b.build(space, cfg.archi.z_dim)
            self._obs_normalizer = cfg.obs_normalizer.build(space)

        def forward(self, state, last_action, history_actor, reference_state, reference_privileged, weights):
            ref = self._obs_normalizer({'state': reference_state, 'privileged_state': reference_privileged})
            z = (self._backward_map(ref)*weights).sum(dim=0, keepdim=True)/weights.sum()
            z = (cfg.archi.z_dim**.5)*F.normalize(z, dim=-1)
            obs = self._obs_normalizer({'state': state, 'last_action': last_action, 'history_actor': history_actor})
            return self._actor(obs, z, cfg.actor_std).mean.float(), z

    model = Inference()
    keys = set(model.state_dict())
    with safe_open(weights/'model.safetensors', framework='pt', device='cpu') as tensors:
        selected = {key: tensors.get_tensor(key) for key in sorted(keys)}
    assert len(selected) == 62
    model.load_state_dict(selected, strict=True)
    model.eval().requires_grad_(False)
    return model, cfg


def prepare(destination, source=None, weights=None):
    import torch

    cache = Path(os.environ.get('BFM_ZERO_ARTIFACT_CACHE', '~/.cache/MotionChallenge/bfm-zero')).expanduser()
    source = Path(source) if source else cache/'source'
    weights = Path(weights) if weights else cache/'weights'
    manifest = load_sources(source)
    for name, item in manifest['artifacts'].items():
        artifact(weights/name, item['sha256'], item['url'])
    torch.set_num_threads(1)
    model, _ = build_inference(source, weights)
    names = ['state', 'last_action', 'history_actor', 'reference_state', 'reference_privileged', 'weights']
    torch.manual_seed(42)
    inputs = (torch.randn(1, 64), torch.randn(1, 29), torch.randn(1, 372),
              torch.randn(8, 64), torch.randn(8, 463), torch.ones(8, 1))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        output = Path(temporary)/'policy.onnx'
        torch.onnx.export(model, inputs, str(output), input_names=names,
                          output_names=['action', 'latent'], opset_version=17, dynamo=False)
        expected = json.loads(Path(__file__).parents[1].joinpath('runtime.json').read_text())['weights_sha256']
        actual = sha256_file(output)
        if actual != expected:
            raise ValueError(f'Export SHA256 mismatch: {actual} != {expected}')
        output.replace(destination)
    print(json.dumps({'phase': 'checkpoint_exported', 'file': str(destination),
                      'bytes': destination.stat().st_size, 'sha256': expected}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.source, args.weights)
