# Inference and export validation

`conversion/prepare.py` downloads the original safetensors, configuration and
unmodified author source listed in `conversion/sources.json`. It selects the 62
inference tensors, loads them strictly into the author's network classes, and
exports CPU float32 ONNX with the locked PyTorch version. No retraining or
quantization is used. The export SHA256 is checked before evaluation.

The exported file is 128,469,542 bytes, SHA256
`cdc9d6454d72b00fabfd1c2ab26ad463a5638cde4e3ac82946975d1e459d9939`.
A fresh export on macOS matched the earlier Linux export byte for byte.

The adapter follows `FBModel.tracking_inference`: an eight-frame uniform latent
mean, shortened at the reference endpoint. The notebook also demonstrates a
separately labelled discounted variant; that is not selected here. The original
per-term four-frame histories and duplicate insertion before physics are retained.
Reference angular finite differences follow the author's implementation, including
its frame convention. A fixed head frame is reconstructed from the torso solely
for reference encoding; benchmark geometry is unchanged.

## Reproduce

```bash
git clone https://github.com/LeCAR-Lab/BFM-Zero.git /tmp/bfm-author
git -C /tmp/bfm-author checkout df9c4a9633aa155c727419e82c906010b1e1845f
# The default evaluation first populates ~/.cache/MotionChallenge/bfm-zero/weights.
uv run -m conversion.prepare --source /tmp/bfm-author \
  --weights ~/.cache/MotionChallenge/bfm-zero/weights --output /tmp/bfm-policy.onnx
uv run validate_inference.py --source /tmp/bfm-author \
  --original ~/.cache/MotionChallenge/bfm-zero/weights --onnx /tmp/bfm-policy.onnx \
  --data ~/.cache/MotionChallenge/lafan --output /tmp/bfm-native-parity.json
```

`BFM_ZERO_ARTIFACT_CACHE` can redirect the source/weight cache. The 160-case
comparison includes four resets, 32 endpoint cases and shuffled joint names.
Observation/history and identical-action PD mapping matched exactly; reference
encoding differed by at most 9.54e-7. Original PyTorch versus ONNX action error
was at most 1.29e-4, corresponding to 2.82e-4 rad (0.017 degrees) at motor targets.
This is numerical agreement, not bitwise network equivalence. The initial 2e-5
action gate failed; isolating identical inputs confirmed backend roundoff. The
published report exposes all tolerances, including a 1e-3 rad motor-target bound.

Scores require a separate full-suite run and independent repeatability check;
this inference report alone is not a benchmark result.
