# Comfyui-SynthidBypass

Bundled custom node pack for the SynthID-Bypass v2 workflow.

This folder is laid out as a standard ComfyUI custom node package so it can be copied directly into `ComfyUI/custom_nodes/Comfyui-SynthidBypass`.

## Included nodes

### `Synthid-Bypass-Facedetailer`

A model-swap face detailer built on the Impact Pack SEGS detail flow, with direct `MODEL`, `CLIP`, `VAE`, `POSITIVE`, and `NEGATIVE` inputs plus simplified adaptive denoise controls and per-frame batch support (`fixed_seed_for_batch`).

Core inputs:

- `image` (`IMAGE`)
- `segs` (`SEGS`)
- `model` (`MODEL`)
- `clip` (`CLIP`)
- `vae` (`VAE`)
- `positive` / `negative` (`CONDITIONING`)
- standard sampling controls such as `steps`, `cfg`, `sampler_name`, `scheduler`, and `denoise`

Adaptive denoise settings:

- `adaptive_mode`: `largest_face` or `per_face`
- `adaptive_ratio`: target face ratio for base denoise
- `adaptive_denoise_min` / `adaptive_denoise_max`: clamp range

Formula:

`applied_denoise = clamp(base_denoise * (face_ratio / adaptive_ratio), min, max)`

Outputs:

- `segs` (`SEGS`): enhanced segments
- `cnet_images` (`IMAGE[]`): control previews
- `denoise_values` (`FLOAT[]`): applied denoise per segment
- `denoise_report` (`STRING`): per-segment denoise log

### `Synthid-Bypass-AdaptiveDenoise`

A lightweight helper node that computes a denoise value from image resolution only.

Inputs:

- `image` (`IMAGE`)
- `adaptive_level` (`INT`)
- `denoise_min` (`FLOAT`)
- `denoise_max` (`FLOAT`)

Outputs:

- `denoise` (`FLOAT`)
- `width` / `height` (`INT`)
- `pixel_ratio` (`FLOAT`)
- `report` (`STRING`)

### `Synthid-Bypass-GPUAccelerator`

Detects active CUDA hardware (`nvidia` via PyTorch) or applies a manual preset (`rtx_4000_pro`, `rtx_4500_pro`, `rtx_6000_pro`) and configures inference optimizations (`OFF`, `STANDARD`, `MAXIMUM`). Prints an explicit OOM safety warning inside `gpu_report` if `MAXIMUM` mode is enabled below 48 GB VRAM.

Outputs:

- `model` (`MODEL`): optionally passed-through / compiled model
- `recommended_batch_size` (`INT`): tuned chunk size (`12` for 4000 Pro, `24` for 4500 Pro, `128` for 6000 Pro)
- `max_resolution` (`INT`): profile resolution cap (`1024`, `1536`, `2048`)
- `gpu_report` (`STRING`): detected hardware stats, active optimizations, and warnings

### `Synthid-Bypass-VideoGlobalRedraw`

Runs Qwen-Image `vae.encode` + optional `QwenImageDiffsynthControlnet` + `KSampler` + `vae.decode` strictly frame-by-frame (`T=1`) over a `[B, H, W, C]` video chunk, avoiding Wan/Qwen 3D VAE temporal squashing across batch frames and preventing VRAM spikes.

### `Synthid-Bypass-VideoFrameProcessor`

Processes a video `IMAGE` batch `[B, H, W, C]` frame-by-frame through Ultralytics face detection, optional SAM refinement, `SEGSDetailerModelSwap` adaptive face cleanup, and SEGS paste-back. Supports `reference_frames_opt` to detect/repair faces from source frames while compositing onto globally redrawn video frames.

### `Synthid-Bypass-TemporalStabilizer`

Reduces inter-frame diffusion flicker using Optical Flow (`optical_flow_farneback` via OpenCV or `optical_flow_raft` via TorchVision) or `frame_blend`, aligning neighboring already-processed frames guided by `reference_frames_opt` without re-injecting raw source watermarks.

## Installation

1. Install with one of these layouts:
   - Copy this folder into `ComfyUI/custom_nodes/Comfyui-SynthidBypass`.
   - Or clone the full Synthid-Bypass repo directly into `ComfyUI/custom_nodes/`; the repo root contains a compatibility loader that exposes the same nodes.
2. Make sure the dependencies below are installed.
3. Restart ComfyUI.

This package currently has no extra Python dependencies of its own beyond what ComfyUI and the required custom node packs already provide.

## Dependencies

- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) (for `UltralyticsDetectorProvider`)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (for video loading/saving in the video workflow)
- [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) if you want the exact sampler/scheduler combination used in the v2 release workflow

The v2 workflow also relies on external node packs listed in the main repository README.

## Troubleshooting

- Search for `Synthid-Bypass-AdaptiveDenoise` after restarting ComfyUI. If it is missing, the package did not load.
- `Install Missing Custom Nodes` in ComfyUI Manager does not currently install this bundled node pack from the workflow repo. Use one of the manual install layouts above.
- Avoid nested folders like `.../Comfyui-SynthidBypass/Comfyui-SynthidBypass/...`.
- If `SEGSDetailerModelSwap` errors on load, verify that `ComfyUI-Impact-Pack` is installed and enabled.

## Typical usage

1. Build face masks or SEGS with Impact Pack tooling.
2. Run `Synthid-Bypass-AdaptiveDenoise` to choose a resolution-aware denoise level.
3. Feed face SEGS into `Synthid-Bypass-Facedetailer`.
4. Paste the updated SEGS back with `SEGSPaste`.

## Files

- `__init__.py`
- `segs_detailer_modelswap.py`
- `gpu_accelerator.py`
- `video_frame_iterator.py`
- `requirements.txt`
- `README.md`
