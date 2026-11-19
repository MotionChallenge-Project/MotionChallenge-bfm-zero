# Validation

The checks below use the pinned model artifacts and the strict observation contract in [admission.mjwarp.json](admission.mjwarp.json).

| Check | Evidence |
| --- | --- |
| Original network versus converted batch network | [Passed](validation/mjwarp-network-parity.json) |
| Observation construction, history and partial resets | [Passed](validation/tensor-observation-parity.json) |
| Full 4096-slot forward and independent seven-slot comparison | [Passed](validation/full-batch-capacity.json) |
| Original ONNX Runtime comparison | [Passed](validation/actual-onnxruntime-parity.json) |
| Official author release and original artifact hashes | [Verified](validation/checkpoint-origin.json) |
| Complete public native checkpoint, fresh-cache ONNX export and tensor conversion | [Passed](validation/native-full-bootstrap.json) |
| Full LAFAN rollout and independent CPU score verification | [Passed](validation/lafan-full.json) |
| Full SEED rollout and independent CPU score verification | [Passed](validation/seed-full.json) |
| Independent model directory, empty conversion/data cache, one-command evaluation | [Passed](validation/clean-reproduction.json) |

The full-batch check executes four complete 4096-slot forwards, resets slots 1 and 4095, and compares seven selected slots with a separately initialized adapter. Its timing measures adapter inference, not complete simulation throughput.

Full dataset reports bind the runtime, protocol, checkpoint, prepared dataset and result hashes. Scores are independently recomputed on CPU from every saved trace. The empty-cache report records the exact tested identity and the scope of the pipeline check.

The complete 3,384,968,272-byte public native checkpoint is verified against its pinned SHA256. A fresh cache runs the model-owned native exporter with ONNX 1.21.0, PyTorch 2.14.1, Pydantic 2.12.5 and Gymnasium 1.2.2. The resulting ONNX matches the original checkpoint hash, and the converted safetensors exactly match the weights used for evaluation. The report includes scalar/batch inference parity and offline cache reuse.
