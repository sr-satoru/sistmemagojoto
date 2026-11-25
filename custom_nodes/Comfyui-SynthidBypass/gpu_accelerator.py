import logging
import torch

logger = logging.getLogger("SynthidBypass.GPUAccelerator")

# GPU profiles with VRAM thresholds and tuned batch/precision presets.
GPU_PROFILES = {
    "rtx_4000_pro": {
        "name": "RTX 4000 Pro",
        "vram_gb": 20.0,
        "batch_size": 12,
        "model_precision": "Q4_K_M (GGUF)",
        "torch_compile": True,
        "pin_memory": False,
        "cuda_graphs": True,
        "xformers_flash": True,
        "vae_tiling": True,
        "max_resolution": 1024,
    },
    "rtx_4500_pro": {
        "name": "RTX 4500 Pro",
        "vram_gb": 24.0,
        "batch_size": 24,
        "model_precision": "Q4_K_M (GGUF)",
        "torch_compile": True,
        "pin_memory": False,
        "cuda_graphs": True,
        "xformers_flash": True,
        "vae_tiling": True,
        "max_resolution": 1536,
    },
    "rtx_6000_pro": {
        "name": "RTX 6000 Pro",
        "vram_gb": 96.0,
        "batch_size": 128,
        "model_precision": "fp16/bf16 (sem quantizacao)",
        "torch_compile": True,
        "pin_memory": True,
        "cuda_graphs": True,
        "xformers_flash": True,
        "vae_tiling": False,
        "max_resolution": 2048,
    },
}

# Minimum VRAM (GB) before warning on MAXIMUM acceleration mode.
FULL_ACCEL_VRAM_THRESHOLD_GB = 48.0


class SynthidBypassGPUAccelerator:
    """
    Detects active GPU hardware, reports VRAM/capabilities, and applies
    CUDA/PyTorch optimizations with safety warnings when MAXIMUM mode is
    enabled on GPUs below 48 GB VRAM (e.g. outside RTX 6000 Pro 96 GB).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "acceleration_mode": (
                    ("OFF", "STANDARD", "MAXIMUM"),
                    {"default": "STANDARD"},
                ),
                "gpu_profile_override": (
                    ("auto_detect", "rtx_4000_pro", "rtx_4500_pro", "rtx_6000_pro"),
                    {"default": "auto_detect"},
                ),
                "force_fp16_models": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "fp16/bf16 (high VRAM, best quality)",
                        "label_off": "GGUF quantizado (low VRAM)",
                    },
                ),
            },
            "optional": {
                "model": ("MODEL",),
                "torch_compile_unet": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "enabled (MAXIMUM only, slow 1st run)",
                        "label_off": "disabled",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "INT", "INT", "STRING")
    RETURN_NAMES = ("model", "recommended_batch_size", "max_resolution", "gpu_report")
    FUNCTION = "accelerate"
    CATEGORY = "SynthidBypass/Video"
    DESCRIPTION = (
        "Detecta a GPU e aplica aceleracao de inferencia. "
        "AVISO: O modo MAXIMUM (pin VRAM + quantizacao/precisao maxima em VRAM) "
        "so deve ser ativado em GPUs com 48GB+ VRAM como a RTX 6000 Pro (96GB). "
        "Em GPUs menores (4000/4500 Pro), use STANDARD para evitar OOM e crash."
    )

    @staticmethod
    def _detect_gpu():
        """Detect active CUDA device properties safely."""
        if not torch.cuda.is_available():
            return {
                "name": "CPU (sem GPU CUDA)",
                "vram_gb": 0.0,
                "compute": "N/A",
                "multiprocessors": 0,
            }

        props = torch.cuda.get_device_properties(0)
        total_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        vram_gb = round(float(total_bytes) / (1024 ** 3), 1)
        return {
            "name": str(props.name),
            "vram_gb": vram_gb,
            "compute": f"{props.major}.{props.minor}",
            "multiprocessors": int(props.multi_processor_count),
        }

    @staticmethod
    def _match_profile(gpu_info, override):
        """Resolve GPU profile either by explicit override, name, or VRAM tier."""
        if override != "auto_detect" and override in GPU_PROFILES:
            return override, dict(GPU_PROFILES[override])

        name_lower = gpu_info.get("name", "").lower()
        vram = float(gpu_info.get("vram_gb", 0.0))

        if "6000" in name_lower or vram >= 48.0:
            profile = dict(GPU_PROFILES["rtx_6000_pro"])
            if "h100" in name_lower:
                profile["name"] = "NVIDIA H100"
            elif "6000" not in name_lower and vram < 80.0:
                profile["batch_size"] = 64
            return "rtx_6000_pro", profile

        if "5090" in name_lower or vram >= 30.0:
            return "rtx_4500_pro", {
                **GPU_PROFILES["rtx_4500_pro"],
                "name": gpu_info["name"],
                "vram_gb": vram,
                "batch_size": 48,
                "max_resolution": 1536,
            }

        if "4500" in name_lower or vram >= 22.0:
            return "rtx_4500_pro", dict(GPU_PROFILES["rtx_4500_pro"])

        if "4000" in name_lower or vram >= 17.0:
            return "rtx_4000_pro", dict(GPU_PROFILES["rtx_4000_pro"])

        return "rtx_4000_pro", {
            **GPU_PROFILES["rtx_4000_pro"],
            "batch_size": 8 if vram >= 12.0 else 4,
            "torch_compile": False,
            "name": f"Generic ({vram:.1f}GB)",
        }

    @staticmethod
    def _try_compile_model(model):
        """Attempt non-destructive torch.compile on a cloned ComfyUI model patcher."""
        if model is None or not hasattr(torch, "compile"):
            return model, False, "model None ou torch.compile indisponivel"

        try:
            compiled_model = model.clone() if hasattr(model, "clone") else model
            if hasattr(compiled_model, "model") and hasattr(compiled_model.model, "diffusion_model"):
                diff_model = compiled_model.model.diffusion_model
                compiled_model.model.diffusion_model = torch.compile(
                    diff_model,
                    mode="reduce-overhead",
                    dynamic=True,
                )
                return compiled_model, True, "torch.compile(dynamic=True) aplicado no diffusion_model"
            return compiled_model, False, "objeto MODEL sem diffusion_model compilavel"
        except Exception as exc:
            return model, False, f"torch.compile ignorado: {exc}"

    def accelerate(
        self,
        acceleration_mode,
        gpu_profile_override,
        force_fp16_models,
        model=None,
        torch_compile_unet=False,
    ):
        gpu_info = self._detect_gpu()
        _, profile = self._match_profile(gpu_info, gpu_profile_override)

        if force_fp16_models:
            profile["model_precision"] = "fp16/bf16 (forcado pelo usuario)"

        report_lines = [
            "=" * 60,
            "  SYNTHID-BYPASS GPU ACCELERATOR",
            "=" * 60,
            f"  GPU Detectada:     {gpu_info['name']}",
            f"  VRAM Total:        {gpu_info['vram_gb']} GB",
            f"  Compute Cap:       {gpu_info.get('compute', 'N/A')}",
            f"  SMs:               {gpu_info.get('multiprocessors', 'N/A')}",
            f"  Perfil Aplicado:   {profile['name']}",
            f"  Modo Aceleracao:   {acceleration_mode}",
            "-" * 60,
        ]

        warnings = []
        optimizations_applied = []

        # Safety validation before enabling MAXIMUM VRAM pinning / fp16.
        if acceleration_mode == "MAXIMUM" and gpu_info["vram_gb"] < FULL_ACCEL_VRAM_THRESHOLD_GB:
            warnings.append(
                f"ATENCAO: Modo MAXIMUM ativado em GPU com apenas {gpu_info['vram_gb']}GB VRAM!\n"
                f"    Use MAXIMUM somente em RTX 6000 Pro (96GB) ou GPU com 48GB+ VRAM.\n"
                f"    RISCO: Out-of-Memory (OOM) e crash durante o processamento de video.\n"
                f"    Recomendacao: troque acceleration_mode para STANDARD."
            )
            logger.warning(
                "SynthidBypassGPUAccelerator: MAXIMUM mode enabled on %s (%.1f GB VRAM < %.1f GB threshold).",
                gpu_info["name"],
                gpu_info["vram_gb"],
                FULL_ACCEL_VRAM_THRESHOLD_GB,
            )

        if force_fp16_models and gpu_info["vram_gb"] < 32.0:
            warnings.append(
                f"ATENCAO: fp16/bf16 forcado em GPU com {gpu_info['vram_gb']}GB VRAM!\n"
                f"    Modelos completos fp16/bf16 (~40GB) excedem esta VRAM; prefira GGUF Q4_K_M."
            )

        out_model = model

        if acceleration_mode in ("STANDARD", "MAXIMUM"):
            if torch.cuda.is_available():
                if profile.get("torch_compile"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                    optimizations_applied.append("TF32 matmul + cuDNN benchmark ativados")

                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    try:
                        torch.backends.cuda.enable_flash_sdp(True)
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                        optimizations_applied.append("PyTorch SDPA Flash/MemEfficient Attention ativado")
                    except Exception:
                        pass

            if profile.get("xformers_flash"):
                try:
                    import xformers  # noqa: F401
                    optimizations_applied.append("xformers flash attention disponivel")
                except ImportError:
                    optimizations_applied.append("xformers nao instalado (usando PyTorch SDPA)")

            if profile.get("cuda_graphs"):
                optimizations_applied.append("CUDA Graphs hint habilitado")

        if acceleration_mode == "MAXIMUM":
            if profile.get("pin_memory") and gpu_info["vram_gb"] >= FULL_ACCEL_VRAM_THRESHOLD_GB:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.95, 0)
                        optimizations_applied.append("VRAM pinned (95% alocado para manter modelos residentes)")
                    except Exception as exc:
                        optimizations_applied.append(f"VRAM fraction mantida padrao ({exc})")
            else:
                optimizations_applied.append(
                    "VRAM pin completo nao recomendado para esta GPU (abaixo de 48GB)"
                )

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    warmup = torch.empty((256, 256), device="cuda", dtype=torch.float16)
                    del warmup
                    optimizations_applied.append("CUDA warmup executado")
                except Exception as exc:
                    optimizations_applied.append(f"CUDA warmup ignorado ({exc})")

            if torch_compile_unet and out_model is not None:
                out_model, _, compile_msg = self._try_compile_model(out_model)
                optimizations_applied.append(compile_msg)

        if acceleration_mode == "OFF":
            optimizations_applied.append("Modo OFF: nenhuma otimizacao extra aplicada")

        report_lines.append("  Otimizacoes:")
        for opt in optimizations_applied:
            report_lines.append(f"    - {opt}")

        if warnings:
            report_lines.append("")
            report_lines.append("  AVISOS:")
            for warning in warnings:
                for line in warning.split("\n"):
                    report_lines.append(f"    {line}")

        report_lines.extend([
            "-" * 60,
            f"  Batch Size Recomendado:  {profile['batch_size']} frames",
            f"  Precisao dos Modelos:    {profile['model_precision']}",
            f"  Resolucao Maxima:        {profile['max_resolution']}px",
            f"  VAE Tiling Recomendado:  {'Sim' if profile.get('vae_tiling') else 'Nao'}",
            "=" * 60,
        ])

        report = "\n".join(report_lines)
        logger.info("\n%s", report)

        return (
            out_model,
            int(profile["batch_size"]),
            int(profile["max_resolution"]),
            report,
        )


NODE_CLASS_MAPPINGS = {
    "Synthid-Bypass-GPUAccelerator": SynthidBypassGPUAccelerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Synthid-Bypass-GPUAccelerator": "Synthid-Bypass-GPUAccelerator",
}

