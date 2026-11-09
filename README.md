# Transfer

Transfer generates synchronized three-view videos from depth videos, reference
images, and a shared text prompt. This release includes the official **20K
checkpoint**, model architecture, and inference code.

## QuickStart

Run the following commands from this directory.

### Installation

Use Python 3.11 and a CUDA GPU with BF16 support.

```bash
git clone https://github.com/aigc-apps/VideoX-Fun.git third_party/VideoX-Fun
git -C third_party/VideoX-Fun checkout 4b7b6402a1e0f0406bd6801fb66c0a00bd922621
pip install -r third_party/VideoX-Fun/requirements.txt -r requirements.txt
```

### Weights

Download the official [Transfer 20K checkpoint](https://huggingface.co/eddyLfan/SINDRI-Transfer)
(cross-view modules and LoRA, approximately 401 MB):

```bash
hf download eddyLfan/SINDRI-Transfer checkpoints/Transfer.pt configs/release.json --local-dir .
```

Place the base model in `models/alibaba-pai/Wan2.2-Fun-5B-Control/`.
Its required revision and file checksums are listed in `configs/base_model.json`.

### Inference

```bash
python infer.py --request examples/request.json --output-dir outputs/example --device cuda:0
# Check inputs without a GPU or model weights:
python infer.py --request examples/request.json --output-dir outputs/check --check-only
```

For custom inputs, edit the request JSON; paths are relative to that file.
Depth must be RGB-encoded grayscale relative disparity, with white indicating
near surfaces. Each run requires a fresh output directory and writes three MP4s
plus JSON records.

## Examples

A real Songling dual-arm robot placing a green pepper onto a blue plate,
240 frames at 24 FPS (10 seconds).

```text
examples/
  input/           Depth, reference images, prompt, and original RGB for comparison
  output/          Three generated videos from the official 20K evaluation
  request.json     Inference request
  comparison.mp4   Original RGB / depth / reference / output, synchronized across views
  source.json      Source identity and checksums
```

## Notes

- Architecture: `sindri_transfer/architecture.py` (`TransferModel`).
- Defaults: 640×352, 24 FPS, 50 steps, CFG 6, 121-frame windows with 25-frame overlap.
- Original RGB is for comparison only. Bundled outputs come from an existing
  official 20K evaluation; re-encoded inputs may change exact generated pixels.
- CPU input checks and architecture equivalence have passed; end-to-end GPU
  inference through this entry remains unverified.
