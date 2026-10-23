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
[Provenance](provenance.json) · [Review requirements](motion-challenge/docs/ADAPTERS.md) ·
[Website](https://motion-challenge-project.epiccabin1025.chatgpt.site)
