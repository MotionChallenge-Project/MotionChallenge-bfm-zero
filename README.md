# Motion Challenge · bfm-zero

[bfm-zero](https://github.com/LeCAR-Lab/BFM-Zero) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-bfm-zero.git
cd motion-challenge-bfm-zero
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

**Development adapter — author-inference review pending.**
[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.

Run all frames after tracking failure: `uv run evaluate.py --termination none`.
This records full-motion errors under a separate protocol; success still requires no failure.

## Checkpoint source

[LeCAR-Lab/BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero/blob/df9c4a9633aa155c727419e82c906010b1e1845f/README.md) · [model/checkpoint/model/model.safetensors → policy.onnx](https://huggingface.co/LeCAR-Lab/BFM-Zero/blob/62b4206d68e026de5e5dc7efb1529bccfb95164c/model/checkpoint/model/model.safetensors). Motion Challenge conversion; publication incomplete.

Motion Challenge exported the actor/backward map from the original safetensors checkpoint. The public conversion chain is incomplete and the configured release download currently returns 404; results are withheld.
