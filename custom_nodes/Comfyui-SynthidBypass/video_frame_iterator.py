import inspect
import logging
import torch
import torch.nn.functional as F
import nodes
from nodes import MAX_RESOLUTION
import comfy

from .segs_detailer_modelswap import SEGSDetailerModelSwap, core, utils

logger = logging.getLogger("SynthidBypass.Video")

try:
    from impact import detectors as _impact_detectors
except Exception:
    _impact_detectors = None

try:
    from impact import impact_pack as _impact_pack_nodes
except Exception:
    _impact_pack_nodes = None

try:
    from impact import segs_nodes as _impact_segs_nodes
except Exception:
    _impact_segs_nodes = None


def _get_simple_detector_cls():
    """Resolve Impact Pack SimpleDetectorForEach across Impact Pack versions."""
    mapped = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("ImpactSimpleDetectorSEGS")
    if mapped is not None:
        return mapped
    for mod in (_impact_detectors, _impact_pack_nodes):
        if mod is not None and hasattr(mod, "SimpleDetectorForEach"):
            return mod.SimpleDetectorForEach
    return None


def _get_segs_paste_cls():
    """Resolve Impact Pack SEGSPaste across Impact Pack versions."""
    mapped = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("SEGSPaste")
    if mapped is not None:
        return mapped
    for mod in (_impact_segs_nodes, _impact_pack_nodes):
        if mod is not None and hasattr(mod, "SEGSPaste"):
            return mod.SEGSPaste
    return None


def _ensure_same_resolution(source_batch, target_hw):
    """Resize BHWC tensor to (target_h, target_w) only when dimensions differ."""
    target_h, target_w = target_hw
    if source_batch.shape[1] == target_h and source_batch.shape[2] == target_w:
        return source_batch
    nchw = source_batch.permute(0, 3, 1, 2)
    resized = F.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _unwrap_node_output(value):
    """Extract first value from ComfyUI V1 tuple or V3 NodeOutput objects."""
    if hasattr(value, "result"):
        return value.result[0]
    if hasattr(value, "args"):
        return value.args[0]
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    return value


def _detect_frame_segs(
    bbox_detector,
    image_frame,
    bbox_threshold,
    bbox_dilation,
    crop_factor,
    drop_size,
    sub_threshold,
    sub_dilation,
    sub_bbox_expansion,
    sam_mask_hint_threshold,
    post_dilation,
    sam_model_opt=None,
    segm_detector_opt=None,
):
    """Detect face SEGS for a single [1, H, W, C] frame using Impact Pack."""
    detector_cls = _get_simple_detector_cls()
    if detector_cls is not None:
        detect_fn = getattr(detector_cls, "detect", None)
        if callable(detect_fn):
            sig = inspect.signature(detect_fn)
            if "post_dilation" in sig.parameters:
                result = detect_fn(
                    bbox_detector,
                    image_frame,
                    bbox_threshold,
                    bbox_dilation,
                    crop_factor,
                    drop_size,
                    sub_threshold,
                    sub_dilation,
                    sub_bbox_expansion,
                    sam_mask_hint_threshold,
                    post_dilation,
                    sam_model_opt=sam_model_opt,
                    segm_detector_opt=segm_detector_opt,
                )
            else:
                result = detect_fn(
                    bbox_detector,
                    image_frame,
                    bbox_threshold,
                    bbox_dilation,
                    crop_factor,
                    drop_size,
                    sub_threshold,
                    sub_dilation,
                    sub_bbox_expansion,
                    sam_mask_hint_threshold,
                    sam_model_opt=sam_model_opt,
                    segm_detector_opt=segm_detector_opt,
                )
            return _unwrap_node_output(result)

        doit_fn = getattr(detector_cls, "doit", None)
        if callable(doit_fn):
            fn = doit_fn if inspect.isfunction(doit_fn) else detector_cls().doit
            sig = inspect.signature(fn)
            kwargs = {
                "bbox_detector": bbox_detector,
                "image": image_frame,
                "bbox_threshold": bbox_threshold,
                "bbox_dilation": bbox_dilation,
                "crop_factor": crop_factor,
                "drop_size": drop_size,
                "sub_threshold": sub_threshold,
                "sub_dilation": sub_dilation,
                "sub_bbox_expansion": sub_bbox_expansion,
                "sam_mask_hint_threshold": sam_mask_hint_threshold,
                "post_dilation": post_dilation,
                "sam_model_opt": sam_model_opt,
                "segm_detector_opt": segm_detector_opt,
            }
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return _unwrap_node_output(fn(**filtered))

    # Fallback path via bbox_detector.detect + optional SAM mask refinement.
    segs = bbox_detector.detect(
        image_frame,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        drop_size,
    )
    if sam_model_opt is not None and len(segs[1]) > 0 and hasattr(core, "make_sam_mask"):
        sam_mask = core.make_sam_mask(
            sam_model_opt,
            segs,
            image_frame,
            "center-1",
            sub_dilation,
            sub_threshold,
            sub_bbox_expansion,
            sam_mask_hint_threshold,
            False,
        )
        if hasattr(core, "segs_bitwise_and_mask"):
            segs = core.segs_bitwise_and_mask(segs, sam_mask)
    return segs


def _paste_frame_segs(base_frame, segs, feather, alpha):
    """Paste refined SEGS back onto a single [1, H, W, C] frame."""
    segs_paste_cls = _get_segs_paste_cls()
    if segs_paste_cls is not None:
        if hasattr(segs_paste_cls, "execute"):
            return _unwrap_node_output(segs_paste_cls.execute(base_frame, segs, feather, alpha))
        if hasattr(segs_paste_cls, "doit"):
            return _unwrap_node_output(segs_paste_cls.doit(base_frame, segs, feather, alpha))

    # Tensor fallback when Impact Pack's SEGSPaste class is unavailable.
    segs = core.segs_scale_match(segs, base_frame.shape)
    out = base_frame.clone()
    alpha_scale = max(0.0, min(1.0, float(alpha) / 255.0))
    for seg in segs[1]:
        if seg.cropped_image is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in seg.crop_region]
        crop_t = utils.to_tensor(seg.cropped_image)
        if crop_t.ndim == 3:
            crop_t = crop_t.unsqueeze(0)
        mask_t = utils.to_tensor(seg.cropped_mask)
        while mask_t.ndim < 4:
            mask_t = mask_t.unsqueeze(0) if mask_t.ndim < 3 else mask_t.unsqueeze(-1)
        mask_t = mask_t[..., :1].clamp(0.0, 1.0) * alpha_scale
        out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1.0 - mask_t) + crop_t * mask_t
    return out


class SynthidBypassVideoFrameProcessor:
    """
    Processes a video batch [B, H, W, C] frame-by-frame through face detection,
    optional SAM refinement, Z-Image Turbo face reconstruction, and SEGS paste-back.
    Supports separate reference_frames_opt so faces can be sourced from the original
    frames and composited onto globally redrawn frames (matching v2.0 behavior).
    """

    @classmethod
    def INPUT_TYPES(cls):
        schedulers = core.get_schedulers() if core is not None else ["normal"]
        return {
            "required": {
                "frames": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "guide_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 0.10, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "bbox_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "drop_size": ("INT", {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "paste_feather": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "paste_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            },
            "optional": {
                "reference_frames_opt": ("IMAGE",),
                "sam_model_opt": ("SAM_MODEL",),
                "segm_detector_opt": ("SEGM_DETECTOR",),
                "sub_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sub_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "post_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "adaptive_denoise": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "adaptive_mode": (("largest_face", "per_face"), {"default": "largest_face"}),
                "adaptive_ratio": ("FLOAT", {"default": 0.03, "min": 0.0001, "max": 1.0, "step": 0.0005}),
                "adaptive_denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "adaptive_denoise_max": ("FLOAT", {"default": 0.28, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "fixed_seed_for_batch": ("BOOLEAN", {"default": True, "label_on": "fixed (anti-flicker)", "label_off": "increment per frame"}),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_frames", "video_report")
    FUNCTION = "process_video"
    CATEGORY = "SynthidBypass/Video"
    DESCRIPTION = (
        "Itera um batch de video frame-a-frame: detecta faces (Ultralytics + SAM opcional), "
        "aplica o face detailer Z-Image Turbo com seed fixa/adaptativa e faz o paste-back."
    )

    def process_video(
        self,
        frames,
        model,
        clip,
        vae,
        positive,
        negative,
        bbox_detector,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        drop_size,
        paste_feather,
        paste_alpha,
        reference_frames_opt=None,
        sam_model_opt=None,
        segm_detector_opt=None,
        sub_threshold=0.93,
        sub_dilation=0,
        sub_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        post_dilation=0,
        noise_mask=True,
        force_inpaint=True,
        refiner_ratio=0.2,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=20,
        adaptive_denoise=True,
        adaptive_mode="largest_face",
        adaptive_ratio=0.03,
        adaptive_denoise_min=0.05,
        adaptive_denoise_max=0.28,
        fixed_seed_for_batch=True,
        scheduler_func_opt=None,
    ):
        SEGSDetailerModelSwap._require_impact_pack()

        total_frames = int(frames.shape[0])
        if total_frames == 0:
            return (frames, "No frames received.")

        ref_batch = None
        if reference_frames_opt is not None and reference_frames_opt.shape[0] > 0:
            ref_batch = _ensure_same_resolution(
                reference_frames_opt,
                (int(frames.shape[1]), int(frames.shape[2])),
            )

        processed_frames = []
        report_lines = []
        faces_total = 0

        for idx in range(total_frames):
            target_frame = frames[idx:idx + 1]
            if ref_batch is not None:
                ref_idx = min(idx, int(ref_batch.shape[0]) - 1)
                detail_source_frame = ref_batch[ref_idx:ref_idx + 1]
            else:
                detail_source_frame = target_frame

            segs = _detect_frame_segs(
                bbox_detector=bbox_detector,
                image_frame=detail_source_frame,
                bbox_threshold=bbox_threshold,
                bbox_dilation=bbox_dilation,
                crop_factor=crop_factor,
                drop_size=drop_size,
                sub_threshold=sub_threshold,
                sub_dilation=sub_dilation,
                sub_bbox_expansion=sub_bbox_expansion,
                sam_mask_hint_threshold=sam_mask_hint_threshold,
                post_dilation=post_dilation,
                sam_model_opt=sam_model_opt,
                segm_detector_opt=segm_detector_opt,
            )

            num_faces = len(segs[1]) if segs and len(segs) > 1 else 0
            faces_total += num_faces

            if num_faces == 0:
                processed_frames.append(target_frame)
                report_lines.append(f"frame={idx + 1}/{total_frames} faces=0 (skipped)")
                continue

            frame_seed = seed if fixed_seed_for_batch else (seed + idx)
            enhanced_segs, _, _, detail_report = SEGSDetailerModelSwap.do_detail(
                detail_source_frame,
                segs,
                model,
                clip,
                vae,
                positive,
                negative,
                guide_size,
                guide_size_for,
                max_size,
                frame_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                noise_mask,
                force_inpaint,
                refiner_ratio,
                1,
                cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                adaptive_denoise=adaptive_denoise,
                adaptive_mode=adaptive_mode,
                adaptive_ratio=adaptive_ratio,
                adaptive_denoise_min=adaptive_denoise_min,
                adaptive_denoise_max=adaptive_denoise_max,
                scheduler_func_opt=scheduler_func_opt,
            )

            pasted = _paste_frame_segs(target_frame, enhanced_segs, paste_feather, paste_alpha)
            processed_frames.append(pasted)
            compact_detail = detail_report.replace("\n", " | ")
            report_lines.append(f"frame={idx + 1}/{total_frames} faces={num_faces} {compact_detail}")

        header = (
            f"Processed {total_frames} frames, total face segments={faces_total}, "
            f"fixed_seed={fixed_seed_for_batch}"
        )
        return (torch.cat(processed_frames, dim=0), header + "\n" + "\n".join(report_lines))


class SynthidBypassTemporalStabilizer:
    """
    Self-contained temporal consistency stabilizer for processed video batches.
    Uses optical flow computed on reference_frames_opt (when provided) or processed_frames
    to warp neighboring processed frames into alignment, reducing diffusion flicker
    without re-injecting raw source watermark pixels.
    """

    _RAFT_MODEL = None
    _RAFT_DEVICE = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_frames": ("IMAGE",),
                "mode": (
                    ("optical_flow_farneback", "optical_flow_raft", "frame_blend", "off"),
                    {"default": "optical_flow_farneback"},
                ),
                "strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "reference_frames_opt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("stabilized_frames", "stabilization_report")
    FUNCTION = "stabilize"
    CATEGORY = "SynthidBypass/Video"
    DESCRIPTION = (
        "Reduz flickering temporal entre frames apos a reconstrucao por difusao. "
        "Usa optical flow dos frames de referencia para alinhar apenas frames ja processados."
    )

    @classmethod
    def _get_raft_model(cls, device):
        if cls._RAFT_MODEL is not None and cls._RAFT_DEVICE == device:
            return cls._RAFT_MODEL
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
        cls._RAFT_MODEL = model
        cls._RAFT_DEVICE = device
        return model

    @staticmethod
    def _warp_with_flow(neighbor_bhwc, flow_yx_pixels):
        """Warp [1, H, W, C] neighbor frame using (dx, dy) pixel flow tensor [1, H, W, 2]."""
        _, h, w, _ = neighbor_bhwc.shape
        device = neighbor_bhwc.device
        dtype = neighbor_bhwc.dtype

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        sample_coords = base_grid + flow_yx_pixels.to(device=device, dtype=torch.float32)
        norm_x = 2.0 * sample_coords[..., 0] / max(1, w - 1) - 1.0
        norm_y = 2.0 * sample_coords[..., 1] / max(1, h - 1) - 1.0
        norm_grid = torch.stack((norm_x, norm_y), dim=-1)

        warped = F.grid_sample(
            neighbor_bhwc.permute(0, 3, 1, 2).float(),
            norm_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.permute(0, 2, 3, 1).to(dtype=dtype)

    @staticmethod
    def _compute_farneback_flow(src_curr, src_neighbor):
        import cv2
        import numpy as np

        curr_gray = cv2.cvtColor(
            (src_curr[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        neigh_gray = cv2.cvtColor(
            (src_neighbor[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        flow = cv2.calcOpticalFlowFarneback(
            curr_gray, neigh_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return torch.from_numpy(flow).unsqueeze(0)

    @classmethod
    def _compute_raft_flow(cls, src_curr, src_neighbor):
        device = src_curr.device
        model = cls._get_raft_model(device)
        _, h, w, _ = src_curr.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        img1 = src_curr.permute(0, 3, 1, 2).float() * 2.0 - 1.0
        img2 = src_neighbor.permute(0, 3, 1, 2).float() * 2.0 - 1.0
        if pad_h or pad_w:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.no_grad():
            flow = model(img1, img2)[-1][:, :, :h, :w]
        return flow.permute(0, 2, 3, 1)

    def stabilize(self, processed_frames, mode, strength, reference_frames_opt=None):
        total = int(processed_frames.shape[0])
        strength_f = max(0.0, min(1.0, float(strength)))
        if mode == "off" or strength_f <= 0.0 or total < 2:
            return (processed_frames, f"Temporal stabilization skipped (mode={mode}, frames={total}).")

        motion_guide = processed_frames
        if reference_frames_opt is not None and reference_frames_opt.shape[0] == total:
            motion_guide = _ensure_same_resolution(
                reference_frames_opt,
                (int(processed_frames.shape[1]), int(processed_frames.shape[2])),
            )

        active_mode = mode
        if mode == "optical_flow_raft":
            try:
                self._get_raft_model(processed_frames.device)
            except Exception as exc:
                logger.warning("RAFT unavailable (%s); falling back to optical_flow_farneback.", exc)
                active_mode = "optical_flow_farneback"

        if active_mode == "optical_flow_farneback":
            try:
                import cv2  # noqa: F401
            except ImportError:
                logger.warning("OpenCV unavailable; falling back to frame_blend.")
                active_mode = "frame_blend"

        stabilized = []
        neighbor_weight = strength_f * 0.5

        for idx in range(total):
            curr = processed_frames[idx:idx + 1]
            if idx == 0 or idx == total - 1:
                stabilized.append(curr)
                continue

            prev_frame = processed_frames[idx - 1:idx]
            next_frame = processed_frames[idx + 1:idx + 2]

            if active_mode in ("optical_flow_raft", "optical_flow_farneback"):
                guide_curr = motion_guide[idx:idx + 1]
                guide_prev = motion_guide[idx - 1:idx]
                guide_next = motion_guide[idx + 1:idx + 2]

                flow_fn = (
                    self._compute_raft_flow
                    if active_mode == "optical_flow_raft"
                    else self._compute_farneback_flow
                )
                prev_warped = self._warp_with_flow(prev_frame, flow_fn(guide_curr, guide_prev))
                next_warped = self._warp_with_flow(next_frame, flow_fn(guide_curr, guide_next))
                blended = (
                    (1.0 - strength_f) * curr
                    + neighbor_weight * prev_warped
                    + neighbor_weight * next_warped
                )
            else:
                blended = (
                    (1.0 - strength_f) * curr
                    + neighbor_weight * prev_frame
                    + neighbor_weight * next_frame
                )

            stabilized.append(blended.clamp(0.0, 1.0))

        report = (
            f"Temporal stabilization applied: requested={mode}, active={active_mode}, "
            f"strength={strength_f:.2f}, frames={total}"
        )
        return (torch.cat(stabilized, dim=0), report)


class SynthidBypassVideoGlobalRedraw:
    """
    Executes the global Qwen-Image reconstruction strictly one frame at a time (T=1)
    across a video chunk [B, H, W, C]. This prevents 3D Wan/Qwen VAE temporal
    squashing across batch frames, prevents DiffSynth ControlNet 5D view crashes
    when B > 1, and keeps DiT attention VRAM constant regardless of chunk size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005}),
            },
            "optional": {
                "canny_frames_opt": ("IMAGE",),
                "model_patch_opt": ("MODEL_PATCH",),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "controlnet_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "controlnet_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "fixed_seed_for_batch": ("BOOLEAN", {"default": True, "label_on": "fixed (anti-flicker)", "label_off": "increment per frame"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("redrawn_frames", "redraw_report")
    FUNCTION = "redraw_video"
    CATEGORY = "SynthidBypass/Video"
    DESCRIPTION = (
        "Executa VAEEncode + QwenImageDiffsynthControlnet + KSampler + VAEDecode "
        "estritamente frame a frame (T=1), evitando compressao temporal da VAE Wan/Qwen "
        "e estouro de VRAM em batches de video."
    )

    @staticmethod
    def _get_qwen_controlnet_node():
        mapped = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("QwenImageDiffsynthControlnet")
        if mapped is not None:
            return mapped
        from comfy_extras.nodes_model_patch import QwenImageDiffsynthControlnet

        return QwenImageDiffsynthControlnet

    def redraw_video(
        self,
        frames,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        canny_frames_opt=None,
        model_patch_opt=None,
        controlnet_strength=1.0,
        controlnet_start_percent=0.0,
        controlnet_end_percent=1.0,
        fixed_seed_for_batch=True,
    ):
        total_frames = int(frames.shape[0])
        if total_frames == 0:
            return (frames, "No frames received.")

        use_controlnet = (
            canny_frames_opt is not None
            and model_patch_opt is not None
            and float(controlnet_strength) != 0.0
        )
        canny_batch = None
        cnet_node = None
        if use_controlnet:
            canny_batch = _ensure_same_resolution(
                canny_frames_opt,
                (int(frames.shape[1]), int(frames.shape[2])),
            )
            cnet_cls = self._get_qwen_controlnet_node()
            cnet_node = cnet_cls()

        intermediate_dev = comfy.model_management.intermediate_device()
        out_frames = []

        for idx in range(total_frames):
            single_frame = frames[idx:idx + 1]
            frame_latent = vae.encode(single_frame[:, :, :, :3])

            frame_model = model
            if use_controlnet and cnet_node is not None and canny_batch is not None:
                cnet_idx = min(idx, int(canny_batch.shape[0]) - 1)
                single_canny = canny_batch[cnet_idx:cnet_idx + 1]
                frame_model = cnet_node.diffsynth_controlnet(
                    model=model,
                    model_patch=model_patch_opt,
                    vae=vae,
                    image=single_canny,
                    strength=float(controlnet_strength),
                    start_percent=float(controlnet_start_percent),
                    end_percent=float(controlnet_end_percent),
                )[0]

            frame_seed = seed if fixed_seed_for_batch else (seed + idx)
            sampled = nodes.common_ksampler(
                frame_model,
                frame_seed,
                int(steps),
                float(cfg),
                sampler_name,
                scheduler,
                positive,
                negative,
                {"samples": frame_latent},
                denoise=float(denoise),
            )[0]

            decoded = vae.decode(sampled["samples"])
            if decoded.ndim == 5:
                decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            out_frames.append(decoded.to(device=intermediate_dev))

            del frame_model, sampled, frame_latent

        report = (
            f"Global redraw completed frame-by-frame: frames={total_frames}, "
            f"controlnet={'enabled' if use_controlnet else 'disabled'}, "
            f"denoise={float(denoise):.4f}, fixed_seed={fixed_seed_for_batch}"
        )
        return (torch.cat(out_frames, dim=0), report)


VIDEO_NODE_CLASS_MAPPINGS = {
    "Synthid-Bypass-VideoGlobalRedraw": SynthidBypassVideoGlobalRedraw,
    "Synthid-Bypass-VideoFrameProcessor": SynthidBypassVideoFrameProcessor,
    "SynthidBypassVideoFrameProcessor": SynthidBypassVideoFrameProcessor,
    "Synthid-Bypass-TemporalStabilizer": SynthidBypassTemporalStabilizer,
}

VIDEO_NODE_DISPLAY_NAME_MAPPINGS = {
    "Synthid-Bypass-VideoGlobalRedraw": "Synthid-Bypass-VideoGlobalRedraw",
    "Synthid-Bypass-VideoFrameProcessor": "Synthid-Bypass-VideoFrameProcessor",
    "SynthidBypassVideoFrameProcessor": "Synthid-Bypass-VideoFrameProcessor (Alias)",
    "Synthid-Bypass-TemporalStabilizer": "Synthid-Bypass-TemporalStabilizer",
}

