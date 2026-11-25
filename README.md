# SynthID-Bypass

**Disclaimer:** This project is intended for educational and AI safety research purposes only. The tools and techniques described herein should not be used for malicious purposes, to circumvent copyright, or to misrepresent the origin of digital content. This proof of concept is presented as-is and without warranty.

---

## 1. Overview

This repository contains a proof-of-concept exploration into the robustness of Google's SynthID [2], a digital watermarking technology integrated into Nano Banana Pro and related Google image systems to help identify synthetic media.

The current release is **V2**, a single ComfyUI workflow kept directly at the repository root. V2 replaces the earlier split between general, portrait, and GGUF workflow variants. The previous release is preserved in the [`V1/`](V1/) archive.

At a high level, the bypass still follows the same core hypothesis as the original release: if the watermark lives in low-level image noise, then a carefully constrained diffusion reconstruction can keep the composition while discarding the watermark-carrying pixels.

## 2. What's New in V2

- **One workflow instead of multiple branches.** V2 is centered on [`Synthid-Bypass-v2.0.json`](Synthid-Bypass-v2.0.json), which consolidates the main path and the face restoration path into one graph.
- **Resolution-aware denoise.** A bundled custom node computes the base denoise from image resolution so the workflow scales more predictably across different input sizes.
- **Custom face detail pass.** V2 ships with a custom node pack, [`custom_nodes/Comfyui-SynthidBypass/`](custom_nodes/Comfyui-SynthidBypass/), that adds a model-swap face detailer with adaptive denoise controls, keeping the renoised faces virtually 1:1 with the originals.
- **Qwen-based base reconstruction.** The main redraw pass uses Qwen Image GGUF models, a Qwen Image Lightning LoRA, and a Qwen DiffSynth Canny controlnet for stronger structure retention.
- **Stricter face segmentation.** V2 combines Ultralytics face detection, MediaPipe face mesh, SAM refinement, SEGS conversion, and a dedicated paste-back stage to preserve facial structure while still shifting pixels enough to break detection.

## 3. Repository Layout

| Path | Purpose |
| --- | --- |
| [`Synthid-Bypass-v2.0.json`](Synthid-Bypass-v2.0.json) | Current image workflow release. |
| [`Synthid-Bypass-Video-v2.0.json`](Synthid-Bypass-Video-v2.0.json) | Video processing workflow with GPU Accelerator, temporal stabilization, and per-frame face cleanup. |
| [`setup_remote.sh`](setup_remote.sh) | Remote GPU auto-detect setup script (`4000pro`, `4500pro`, `6000pro`, or `auto`). |
| [`comparison/before/`](comparison/before/) | Place new v2 input images here. |
| [`comparison/after/`](comparison/after/) | Place new v2 processed outputs here. |
| [`workflow_screenshots/`](workflow_screenshots/) | Place fresh screenshots of the v2 graph here. |
| [`V1/`](V1/) | Archived v1 workflows, screenshots, and comparison assets. |
| [`synthid_analysis/`](synthid_analysis/) | Analysis images showing the watermark pattern and discovery process. |
| [`custom_nodes/Comfyui-SynthidBypass/`](custom_nodes/Comfyui-SynthidBypass/) | Bundled custom node pack used by v2. |

## 4. V2 Workflow

The v2 release lives in [`Synthid-Bypass-v2.0.json`](Synthid-Bypass-v2.0.json). Internally, it combines two model paths:

- A **Qwen Image** path for the global redraw, using `qwen-image-2512-Q4_K_M.gguf`, a Qwen image VAE, a Qwen VL GGUF text encoder, a Qwen Image Lightning LoRA, and a Qwen DiffSynth Canny controlnet.
- A **Z-Image Turbo** path for targeted face cleanup, using `z_image_turbo-Q4_K_M.gguf`, `ae.safetensors`, a GGUF Qwen 3.4B text encoder, and a custom face detailer node.

### V2 pipeline summary

1. **Input normalization and adaptive denoise**  
   The workflow loads the source image, rescales where needed, and computes a denoise target with the `Synthid-Bypass-AdaptiveDenoise` node based on input resolution.

2. **Global redraw pass**  
   A single low-denoise `KSampler` redraws the image using the Qwen model path while a Canny edge map is fed into `QwenImageDiffsynthControlnet` to lock the composition.

3. **Strict face segmentation**  
   V2 detects faces with Ultralytics, converts face regions into SEGS masks, and also uses MediaPipe face mesh plus a SAM-assisted fallback path to get cleaner face boundaries.

4. **Adaptive face reconstruction**  
   The bundled `Synthid-Bypass-Facedetailer` node swaps to the Z-Image Turbo model path for face-only cleanup. It supports adaptive denoise scaling so larger detected faces can receive stronger detail repair without overprocessing small faces.

5. **Composite and save**  
   The updated SEGS regions are pasted back onto the main image and the final output is written with `SaveImage`.

### V2 comparison examples

Current v2 proof images live under [`comparison/before/`](comparison/before/) and [`comparison/after/`](comparison/after/).

#### Example 12

| Original | Processed |
| --- | --- |
| <img src="comparison/before/12.png" width="400"> | <img src="comparison/after/12.png" width="400"> |

#### Example 10

| Original | Processed |
| --- | --- |
| <img src="comparison/before/10.png" width="400"> | <img src="comparison/after/10.png" width="400"> |

Additional workflow screenshots can be added under [`workflow_screenshots/`](workflow_screenshots/).

## 5. Analysis: Visualizing the SynthID Pattern

The original insight that led to this bypass came from a simple question: what does SynthID actually look like?

### Visualizing the invisible

To reveal the watermark, I took a blank black image and asked Nano Banana Pro to remake it exactly:

<img src="synthid_analysis/nano_banana_remake_process.png" width="600">

I then increased the exposure, brilliance, highlights, and shadows while lowering contrast. That amplified the subtle pixel-level differences enough to make the watermark pattern visible.

| Watermark Pattern 1 | Watermark Pattern 2 |
| --- | --- |
| <img src="synthid_analysis/watermark_pattern_01.jpeg" width="400"> | <img src="synthid_analysis/watermark_pattern_02.jpeg" width="400"> |

An important observation was that the watermark pattern is not deterministic. Regenerating the same prompt produces a different pattern each time, which suggests the watermark is not fixed per prompt.

### From visualization to bypass

This led to the core hypothesis behind both v1 and v2: if the watermark is just a layer of image noise, it should be possible to remove it by re-processing the image through a diffusion model while keeping the source composition as tightly constrained as possible.

The key ingredients are:

1. **Low-denoise regeneration** to repaint the image while replacing low-level noise.
2. **Structural guidance** with edge-based control so the redraw follows the source composition.
3. **Targeted face repair** so high-salience human details survive the redraw.

That combination remains the foundation of the project, even though v2 now packages it as a single workflow with a more deliberate face-specific path.

## 6. V1 Archive

The original multi-workflow release has been preserved under [`V1/`](V1/). That archive includes:

- Archived workflows in [`V1/workflows/`](V1/workflows/)
- Archived before/after comparison images in [`V1/comparison/`](V1/comparison/)
- Archived workflow screenshots in [`V1/workflow_screenshots/`](V1/workflow_screenshots/)

This keeps the historical proof set available without mixing legacy assets into the root v2 release.

## 7. Setup & Usage

To run v2, you need a working ComfyUI install plus the required models and custom nodes.

### Installation flow

1. Install a recent version of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
2. Install the required external custom node repos listed below.
3. Install the bundled Synthid custom nodes with one of these two layouts:
   - **Manual copy:** copy [`custom_nodes/Comfyui-SynthidBypass/`](custom_nodes/Comfyui-SynthidBypass/) into `ComfyUI/custom_nodes/Comfyui-SynthidBypass/`.
   - **Direct repo clone:** clone this whole repo directly into `ComfyUI/custom_nodes/`. The repo root now includes a compatibility loader so ComfyUI can register the bundled Synthid nodes even if you do not extract the nested folder separately.
4. Download the required models into the exact ComfyUI model directories listed in the resources section.
5. Drag [`Synthid-Bypass-v2.0.json`](Synthid-Bypass-v2.0.json) onto your ComfyUI canvas.
6. Set the `Load Image` node to your source image and run the graph.

### Notes

- V2 is **GGUF-first**. The workflow expects GGUF loaders for both model and CLIP paths.
- The current v2 face detail path uses the `res_2s` sampler and `bong_tangent` scheduler, which come from the `RES4LYF` custom node pack.
- Higher-resolution images may require a stronger adaptive level and/or a slightly higher denoise ceiling.
- ComfyUI Manager's `Install Missing Custom Nodes` flow does **not** currently install the bundled Synthid node pack from this repo. Use one of the two install layouts above for `Synthid-Bypass-Facedetailer` and `Synthid-Bypass-AdaptiveDenoise`.

### Troubleshooting

- After restarting ComfyUI, search for `Synthid-Bypass-AdaptiveDenoise`. If it is missing, the package did not load.
- Supported folder shapes are either `ComfyUI/custom_nodes/<repo-clone>/__init__.py` or `ComfyUI/custom_nodes/Comfyui-SynthidBypass/__init__.py`.
- Do not nest the folder twice. Broken layouts usually look like `ComfyUI/custom_nodes/<repo-clone>/<repo-clone>/...` or `ComfyUI/custom_nodes/Comfyui-SynthidBypass/Comfyui-SynthidBypass/...`.
- `SEGSDetailerModelSwap` depends on [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack). If that pack failed to load, the Synthid face detailer will also fail.

## 8. Resources

### Required v2 custom nodes

| Custom node | Purpose in v2 | Link |
| --- | --- | --- |
| Bundled `Comfyui-SynthidBypass` | Custom `Synthid-Bypass-Facedetailer`, `Synthid-Bypass-AdaptiveDenoise`, `Synthid-Bypass-GPUAccelerator`, `Synthid-Bypass-VideoGlobalRedraw`, `Synthid-Bypass-VideoFrameProcessor`, and `Synthid-Bypass-TemporalStabilizer` nodes | [`custom_nodes/Comfyui-SynthidBypass/`](custom_nodes/Comfyui-SynthidBypass/) |
| ComfyUI-GGUF | GGUF UNet and CLIP loaders | [GitHub](https://github.com/city96/ComfyUI-GGUF) |
| ComfyUI Impact Pack | Detection, SEGS tooling, paste-back, and related face utilities | [GitHub](https://github.com/ltdrdata/ComfyUI-Impact-Pack) |
| ComfyUI Impact Subpack | Provides `UltralyticsDetectorProvider` (`bbox/yolov8n-face.pt`) | [GitHub](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) |
| ComfyUI-VideoHelperSuite | Video load/combine (`VHS_LoadVideo`, `VHS_BatchManager`, `VHS_VideoInfoLoaded`, `VHS_VideoCombine`) | [GitHub](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| rgthree-comfy | Power LoRA loader, image comparer, and `Display Any (rgthree)` nodes | [GitHub](https://github.com/rgthree/rgthree-comfy) |
| comfyui_controlnet_aux | Canny and MediaPipe preprocessors | [GitHub](https://github.com/Fannovel16/comfyui_controlnet_aux) |
| ComfyUI-KJNodes | Image resize helpers used in v2 | [GitHub](https://github.com/kijai/ComfyUI-KJNodes) |
| ComfyUI-Inpaint-CropAndStitch | Crop/stitch support for face-region processing | [GitHub](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) |
| RES4LYF | `res_2s` sampler and `bong_tangent` scheduler used in the face detail path | [GitHub](https://github.com/ClownsharkBatwing/RES4LYF) |
| ComfyUI-Frame-Interpolation *(optional)* | Optional RIFE frame interpolation | [GitHub](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) |

### Legacy v1 custom nodes

These are retained for the archived v1 workflows in [`V1/workflows/`](V1/workflows/):

- [ComfyUI-DyPE](https://github.com/wildminder/ComfyUI-DyPE)
- [Masquerade Nodes](https://github.com/BadCafeCode/masquerade-nodes-comfyui)
- [SeedVR2 VideoUpscaler Nodes](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

### V2 model download links

Place each file in the matching ComfyUI directory:

| Model file | Directory | Download link |
| --- | --- | --- |
| `qwen-image-2512-Q4_K_M.gguf` | `models/diffusion_models/` | [Hugging Face](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf) |
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `models/clip/` | [Hugging Face](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf) |
| `qwen_image_vae.safetensors` | `models/vae/` | [Hugging Face](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| `qwen_image_canny_diffsynth_controlnet.safetensors` | `models/model_patches/` | [Hugging Face](https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets/resolve/main/split_files/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors) |
| `Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors` | `models/loras/` | [Hugging Face](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors) |
| `z_image_turbo-Q4_K_M.gguf` | `models/diffusion_models/` | [Hugging Face](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K_M.gguf) |
| `Qwen_3_4b-imatrix-IQ4_XS.gguf` | `models/clip/` | [Hugging Face](https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-imatrix-IQ4_XS.gguf) |
| `ae.safetensors` | `models/vae/` | [Hugging Face](https://huggingface.co/Comfy-Org/Z-Image-ComfyUI/resolve/main/split_files/vae/ae.safetensors) |
| `yolov8n-face.pt` | `models/ultralytics/bbox/` | [Hugging Face](https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt) |
| `sam_vit_b_01ec64.pth` | `models/sams/` | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

## 9. Limitations

- This remains a research workflow, not a one-click consumer product.
- V2 still requires a capable GPU, ComfyUI familiarity, and several external node packs.
- Very high-resolution images may need stronger adaptive settings or adjusted denoise ranges.
- Like v1, the process can introduce artifacts, especially in fine textures or text-heavy regions.

## 10. Related Work & References

This project builds on existing research into diffusion-based watermarking and diffusion-based reconstruction attacks. The core idea of using controlled diffusion to remove low-level watermark signals is aligned with the broader literature on the fragility of pixel-space watermarks [1].

1. Hu, Y., et al. (2024). *Stable signature is unstable: Removing image watermark from diffusion models*. arXiv preprint arXiv:2405.07145. https://arxiv.org/abs/2405.07145
2. Google DeepMind. (2023). *SynthID*. https://deepmind.google/models/synthid/

## 11. Ethical Use

This project is released in the spirit of open and responsible AI safety research. The goal is to help researchers understand the limits of current watermarking systems so more robust approaches can be built.

If you develop a defense that defeats these workflows, please open an issue or pull request and share the results.

## 12. Video Processing (V2 Video Workflow)

[`Synthid-Bypass-Video-v2.0.json`](Synthid-Bypass-Video-v2.0.json) extends V2 to process full video files while keeping original resolution, original FPS, and original audio in MP4 (H.264) output.

### Video pipeline overview

1. **`Synthid-Bypass-GPUAccelerator`** auto-detects the active GPU (or applies a manual preset: `rtx_4000_pro`, `rtx_4500_pro`, `rtx_6000_pro`), sets CUDA optimizations (`OFF` / `STANDARD` / `MAXIMUM`), and outputs `recommended_batch_size`, `max_resolution`, and a live `gpu_report`.
2. **`VHS_BatchManager` + `VHS_LoadVideo`** stream the entire video (`frame_load_cap = 0`) in GPU-tuned sub-batches (`recommended_batch_size` frames per chunk) while extracting original audio and `video_info` (`VHS_VideoInfoLoaded` -> original `fps`).
3. **Qwen Image 2512 Global Redraw (`Synthid-Bypass-VideoGlobalRedraw`)** runs Canny ControlNet (`QwenImageDiffsynthControlnet`) + `VAEEncode`/`KSampler`/`VAEDecode` strictly one frame at a time (`T=1`) across the chunk so the 3D Wan/Qwen VAE never mixes frames temporally and VRAM stays constant, using a fixed seed and resolution-aware denoise from `Synthid-Bypass-AdaptiveDenoise`.
4. **`Synthid-Bypass-TemporalStabilizer`** aligns neighboring processed frames via Optical Flow (`optical_flow_farneback` or `optical_flow_raft`) guided by reference motion without re-injecting raw source watermarks.
5. **`Synthid-Bypass-VideoFrameProcessor`** runs per-frame face detection (`UltralyticsDetectorProvider` + optional `SAMLoader`), Z-Image Turbo adaptive face cleanup, and SEGS paste-back across all frames in the batch.
6. **`VHS_VideoCombine`** encodes the processed frames with the original audio track and frame rate into MP4.

### GPU Acceleration Profiles & Warning

| GPU Target | VRAM | Chunk Batch Size | Default Model Format | Recommended Mode |
| --- | --- | --- | --- | --- |
| **RTX 4000 Pro** | 20 GB | 12 frames | `GGUF Q4_K_M` | `STANDARD` |
| **RTX 4500 Pro** | 24 GB | 24 frames | `GGUF Q4_K_M` | `STANDARD` |
| **RTX 6000 Pro** | 96 GB | 128 frames | `GGUF Q4_K_M` or optional `bf16` | `MAXIMUM` |

> [!WARNING]
> Only enable **`MAXIMUM`** acceleration mode in `Synthid-Bypass-GPUAccelerator` on GPUs with **48 GB+ VRAM** (such as **RTX 6000 Pro 96 GB**). On smaller GPUs (20–24 GB), `MAXIMUM` memory pinning / unquantized `bf16` models can cause out-of-memory (OOM) crashes. Use `STANDARD` on RTX 4000 Pro / 4500 Pro.

### Automated Cloud / Remote Setup (`setup_remote.sh`)

Run [`setup_remote.sh`](setup_remote.sh) on a fresh Linux GPU instance (RunPod, Vast.ai, Lambda, etc.) to install ComfyUI, all custom nodes, required models, and workflows:

```bash
# Auto-detect GPU via nvidia-smi
bash setup_remote.sh

# Or force a specific hardware profile manually:
bash setup_remote.sh 4000pro   # RTX 4000 Pro (20GB VRAM)
bash setup_remote.sh 4500pro   # RTX 4500 Pro (24GB VRAM)
bash setup_remote.sh 6000pro   # RTX 6000 Pro (96GB VRAM, also downloads bf16 weights)
```

### Optional `bf16` model downloads (RTX 6000 Pro 96 GB)

When using the `6000pro` preset, `setup_remote.sh` downloads both the standard GGUF Q4 weights (for immediate out-of-the-box workflow execution) and unquantized `bf16` weights so you can optionally swap `UnetLoaderGGUF`/`CLIPLoaderGGUF` to `UNETLoader`/`CLIPLoader`:

| Model file | Directory | Download link |
| --- | --- | --- |
| `qwen_image_2512_bf16.safetensors` | `models/diffusion_models/` | [Hugging Face](https://huggingface.co/Comfy-Org/Qwen-Image/resolve/main/split_files/diffusion_models/qwen_image_2512_bf16.safetensors) |
| `qwen_2.5_vl_7b.safetensors` | `models/text_encoders/` | [Hugging Face](https://huggingface.co/Comfy-Org/Qwen-Image/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors) |
| `z_image_turbo_bf16.safetensors` | `models/diffusion_models/` | [Hugging Face](https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) |
| `qwen_3_4b.safetensors` | `models/text_encoders/` | [Hugging Face](https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
