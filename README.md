# BFM-Zero

[Author project](https://lecar-lab.github.io/BFM-Zero/) · [Author checkpoint](https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/62b4206d68e026de5e5dc7efb1529bccfb95164c/model/checkpoint/model) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/motion-challenge-bfm-zero.git
cd motion-challenge-bfm-zero
uv sync
uv run evaluate.py
```

Produces success, completion and six MPBPE values in a verified `result.json`.
MuJoCo 3.17 CPU batch; `--jobs 8` sets parallelism. Results and traces stay outside Git.

The first run downloads the author's 3.38 GB checkpoint and pinned source, then
exports only the actor, backward map and normalizer to a SHA256-checked ONNX file.
Later runs reuse the cache. Use `--weights PATH --data PATH --offline` for cached artifacts.

[Inference and export validation](validation/README.md) · [Native parity report](validation/native-parity.json)
