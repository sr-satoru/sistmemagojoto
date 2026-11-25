#!/usr/bin/env bash
set -euo pipefail

# ==============================================================
#  Synthid-Bypass Video - Remote Setup with GPU Auto-Detection
# ==============================================================
# Usage:
#   bash setup_remote.sh             -> auto-detect GPU via nvidia-smi
#   bash setup_remote.sh 4000pro     -> RTX 4000 Pro preset (20GB VRAM)
#   bash setup_remote.sh 4500pro     -> RTX 4500 Pro preset (24GB VRAM)
#   bash setup_remote.sh 6000pro     -> RTX 6000 Pro preset (96GB VRAM + bf16)
#
# Optional env vars:
#   COMFYUI_DIR=/path/to/ComfyUI     -> reuse or install at custom location
#   PYTHON_BIN=python3               -> override Python executable
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANUAL_PROFILE="${1:-auto}"
COMFYUI_DIR="${COMFYUI_DIR:-$PWD/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN}  Synthid-Bypass Video - Setup Remoto             ${NC}"
echo -e "${CYAN}==================================================${NC}"

usage() {
    echo "Uso: bash setup_remote.sh [auto|4000pro|4500pro|6000pro]"
    echo "Variaveis opcionais: COMFYUI_DIR=/caminho/ComfyUI PYTHON_BIN=python3"
}

if [[ "$MANUAL_PROFILE" == "-h" || "$MANUAL_PROFILE" == "--help" ]]; then
    usage
    exit 0
fi

detect_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}AVISO: nvidia-smi nao encontrado. Hardware nao detectado via CUDA.${NC}"
        GPU_NAME="Unknown (no nvidia-smi)"
        GPU_VRAM_MB="0"
        GPU_VRAM_GB="0.0"
        GPU_DRIVER="N/A"
        GPU_CUDA="N/A"
        GPU_COUNT="0"
        return
    fi

    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | xargs)"
    GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
    GPU_VRAM_GB="$($PYTHON_BIN -c "print(round(float('${GPU_VRAM_MB:-0}') / 1024.0, 1))")"
    GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1 | xargs)"
    GPU_CUDA="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | xargs || echo "N/A")"
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l | xargs)"

    echo -e "${GREEN}GPU Detectada:${NC}    $GPU_NAME"
    echo -e "${GREEN}VRAM:${NC}             ${GPU_VRAM_GB} GB (${GPU_VRAM_MB} MB)"
    echo -e "${GREEN}Driver:${NC}           $GPU_DRIVER"
    echo -e "${GREEN}Compute Cap:${NC}      $GPU_CUDA"
    echo -e "${GREEN}Num GPUs:${NC}         $GPU_COUNT"
    echo ""
}

determine_profile() {
    local normalized
    normalized="$(echo "$MANUAL_PROFILE" | tr '[:upper:]' '[:lower:]' | tr -d '_-')"

    if [[ "$normalized" != "auto" ]]; then
        case "$normalized" in
            4000pro|rtx4000pro|4000) GPU_PROFILE="4000pro" ;;
            4500pro|rtx4500pro|4500) GPU_PROFILE="4500pro" ;;
            6000pro|rtx6000pro|6000) GPU_PROFILE="6000pro" ;;
            *)
                echo -e "${RED}Perfil desconhecido: $MANUAL_PROFILE${NC}"
                echo "Perfis validos: auto, 4000pro, 4500pro, 6000pro"
                exit 1
                ;;
        esac
        echo -e "${YELLOW}Perfil forcado manualmente:${NC} $GPU_PROFILE"
    else
        local vram_int name_lc
        vram_int="${GPU_VRAM_MB%.*}"
        vram_int="${vram_int:-0}"
        name_lc="$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')"

        if [[ "$name_lc" == *"6000"* || "$name_lc" == *"h100"* || "$vram_int" -ge 48000 ]]; then
            GPU_PROFILE="6000pro"
        elif [[ "$name_lc" == *"4500"* || "$name_lc" == *"5090"* || "$vram_int" -ge 22000 ]]; then
            GPU_PROFILE="4500pro"
        elif [[ "$name_lc" == *"4000"* || "$vram_int" -ge 17000 ]]; then
            GPU_PROFILE="4000pro"
        else
            GPU_PROFILE="4000pro"
            echo -e "${YELLOW}AVISO: GPU com ${GPU_VRAM_GB}GB VRAM — usando perfil conservador (4000pro).${NC}"
        fi
        echo -e "${GREEN}Perfil auto-detectado:${NC} $GPU_PROFILE"
    fi

    case "$GPU_PROFILE" in
        6000pro)
            BATCH_SIZE=128
            MODEL_TYPE="bf16+gguf"
            ACCEL_MODE="MAXIMUM"
            VAE_TILING="false"
            MAX_RES=2048
            DOWNLOAD_BF16="true"
            echo -e "${GREEN}RTX 6000 Pro (96GB) -> batch=$BATCH_SIZE, acel=$ACCEL_MODE, baixa GGUF + bf16 completo${NC}"
            ;;
        4500pro)
            BATCH_SIZE=24
            MODEL_TYPE="gguf_q4"
            ACCEL_MODE="STANDARD"
            VAE_TILING="true"
            MAX_RES=1536
            DOWNLOAD_BF16="false"
            echo -e "${GREEN}RTX 4500 Pro (24GB) -> batch=$BATCH_SIZE, acel=$ACCEL_MODE, GGUF Q4_K_M${NC}"
            ;;
        4000pro)
            BATCH_SIZE=12
            MODEL_TYPE="gguf_q4"
            ACCEL_MODE="STANDARD"
            VAE_TILING="true"
            MAX_RES=1024
            DOWNLOAD_BF16="false"
            echo -e "${GREEN}RTX 4000 Pro (20GB) -> batch=$BATCH_SIZE, acel=$ACCEL_MODE, GGUF Q4_K_M${NC}"
            ;;
    esac
    echo ""
}

ensure_system_tools() {
    echo -e "${CYAN}[1/5] Verificando ferramentas base (git, wget, ffmpeg)...${NC}"
    if ! command -v ffmpeg >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
            apt-get update -qq && apt-get install -y -qq ffmpeg wget git >/dev/null
        else
            echo -e "${YELLOW}AVISO: ffmpeg nao encontrado. Instale ffmpeg para VHS_VideoCombine exportar MP4 com audio.${NC}"
        fi
    fi
}

install_comfyui() {
    echo -e "${CYAN}[2/5] Instalando/Verificando ComfyUI em $COMFYUI_DIR...${NC}"
    if [[ ! -d "$COMFYUI_DIR" ]]; then
        git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    else
        echo "  ComfyUI ja existe em $COMFYUI_DIR, mantendo pasta existente."
    fi

    cd "$COMFYUI_DIR"
    "$PYTHON_BIN" -m pip install -r requirements.txt
    "$PYTHON_BIN" -m pip install -q opencv-python-headless gguf

    if ! "$PYTHON_BIN" -c "import xformers" >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip install -q xformers || \
            echo -e "${YELLOW}Aviso: xformers opcional falhou; GPU Accelerator usara PyTorch SDPA Flash Attention.${NC}"
    fi
    echo -e "${GREEN}ComfyUI pronto.${NC}"
}

install_custom_nodes() {
    echo -e "${CYAN}[3/5] Instalando custom nodes...${NC}"
    mkdir -p "$COMFYUI_DIR/custom_nodes"
    cd "$COMFYUI_DIR/custom_nodes"

    REPOS=(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
        "https://github.com/city96/ComfyUI-GGUF.git"
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
        "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git"
        "https://github.com/rgthree/rgthree-comfy.git"
        "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
        "https://github.com/kijai/ComfyUI-KJNodes.git"
        "https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git"
        "https://github.com/ClownsharkBatwing/RES4LYF.git"
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"
    )

    for repo in "${REPOS[@]}"; do
        dir_name="$(basename "$repo" .git)"
        if [[ ! -d "$dir_name" ]]; then
            echo "  Clonando $dir_name..."
            git clone --depth 1 "$repo" "$dir_name"
        else
            echo "  $dir_name ja existe, pulando."
        fi
    done

    for dir in */; do
        if [[ -f "${dir}requirements.txt" ]]; then
            echo "  Instalando deps de ${dir}..."
            "$PYTHON_BIN" -m pip install -r "${dir}requirements.txt" || \
                echo -e "${YELLOW}  Aviso: algumas deps opcionais em ${dir} falharam.${NC}"
        fi
        if [[ -f "${dir}install.py" ]]; then
            echo "  Executando install.py de ${dir}..."
            "$PYTHON_BIN" "${dir}install.py" || true
        fi
    done

    rm -rf "$COMFYUI_DIR/custom_nodes/Comfyui-SynthidBypass"
    cp -r "$SCRIPT_DIR/custom_nodes/Comfyui-SynthidBypass" "$COMFYUI_DIR/custom_nodes/Comfyui-SynthidBypass"
    echo -e "${GREEN}Comfyui-SynthidBypass instalado.${NC}"
}

fetch_file() {
    local url="$1"
    local dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  [OK] $(basename "$dest") ja existe."
        return
    fi
    echo "  Baixando $(basename "$dest") -> $dest"
    mkdir -p "$(dirname "$dest")"
    wget -c --progress=bar:force:noscroll -O "$dest" "$url" || {
        echo -e "${YELLOW}  AVISO: falha ao baixar $url${NC}"
        rm -f "$dest"
    }
}

download_models() {
    echo -e "${CYAN}[4/5] Baixando modelos ($MODEL_TYPE)...${NC}"
    cd "$COMFYUI_DIR"

    mkdir -p models/{diffusion_models,clip,text_encoders,vae,model_patches,diffusion_model_patches,loras,ultralytics/bbox,sams}

    # Base GGUF Q4 models (garante funcionamento imediato do workflow em qualquer GPU)
    fetch_file \
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf" \
        "models/diffusion_models/qwen-image-2512-Q4_K_M.gguf"
    fetch_file \
        "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" \
        "models/clip/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    fetch_file \
        "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K_M.gguf" \
        "models/diffusion_models/z_image_turbo-Q4_K_M.gguf"
    fetch_file \
        "https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-imatrix-IQ4_XS.gguf" \
        "models/clip/Qwen_3_4b-imatrix-IQ4_XS.gguf"

    # Full bf16 models when running RTX 6000 Pro (96GB)
    if [[ "$DOWNLOAD_BF16" == "true" ]]; then
        echo -e "${YELLOW}  Baixando pesos bf16 completos (RTX 6000 Pro 96GB)...${NC}"
        fetch_file \
            "https://huggingface.co/Comfy-Org/Qwen-Image/resolve/main/split_files/diffusion_models/qwen_image_2512_bf16.safetensors" \
            "models/diffusion_models/qwen_image_2512_bf16.safetensors"
        fetch_file \
            "https://huggingface.co/Comfy-Org/Qwen-Image/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" \
            "models/text_encoders/qwen_2.5_vl_7b.safetensors"
        fetch_file \
            "https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
            "models/diffusion_models/z_image_turbo_bf16.safetensors"
        fetch_file \
            "https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
            "models/text_encoders/qwen_3_4b.safetensors"
    fi

    # Shared VAEs, ControlNet, LoRA, face detectors
    fetch_file \
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
        "models/vae/qwen_image_vae.safetensors"
    fetch_file \
        "https://huggingface.co/Comfy-Org/Z-Image-ComfyUI/resolve/main/split_files/vae/ae.safetensors" \
        "models/vae/ae.safetensors"
    fetch_file \
        "https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets/resolve/main/split_files/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors" \
        "models/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors"

    # Symlink into diffusion_model_patches for older/alternative ComfyUI forks
    ln -sf ../model_patches/qwen_image_canny_diffsynth_controlnet.safetensors \
        "models/diffusion_model_patches/qwen_image_canny_diffsynth_controlnet.safetensors"

    fetch_file \
        "https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors" \
        "models/loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
    fetch_file \
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
        "models/ultralytics/bbox/yolov8n-face.pt"
    fetch_file \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
        "models/sams/sam_vit_b_01ec64.pth"

    echo -e "${GREEN}Modelos baixados.${NC}"
}

copy_workflows_and_config() {
    echo -e "${CYAN}[5/5] Copiando workflows e gerando config...${NC}"
    cd "$COMFYUI_DIR"

    mkdir -p user/default/workflows
    for wf in Synthid-Bypass-v2.0.json Synthid-Bypass-Video-v2.0.json; do
        if [[ -f "$SCRIPT_DIR/$wf" ]]; then
            cp "$SCRIPT_DIR/$wf" "$COMFYUI_DIR/$wf"
            cp "$SCRIPT_DIR/$wf" "$COMFYUI_DIR/user/default/workflows/$wf"
        fi
    done

    cat > "$COMFYUI_DIR/synthid_gpu_config.json" <<EOF
{
    "gpu_name": "$GPU_NAME",
    "gpu_vram_gb": $GPU_VRAM_GB,
    "gpu_profile": "$GPU_PROFILE",
    "recommended_acceleration_mode": "$ACCEL_MODE",
    "batch_size": $BATCH_SIZE,
    "model_type": "$MODEL_TYPE",
    "vae_tiling": $VAE_TILING,
    "max_resolution": $MAX_RES,
    "install_date": "$(date -Iseconds)"
}
EOF
    echo -e "${GREEN}Config salva em $COMFYUI_DIR/synthid_gpu_config.json.${NC}"
}

detect_gpu
determine_profile
ensure_system_tools
install_comfyui
install_custom_nodes
download_models
copy_workflows_and_config

echo ""
echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN}  Setup completo!                                 ${NC}"
echo -e "${CYAN}==================================================${NC}"
echo -e "  ComfyUI Dir:  ${GREEN}$COMFYUI_DIR${NC}"
echo -e "  GPU:          ${GREEN}$GPU_NAME (${GPU_VRAM_GB} GB)${NC}"
echo -e "  Perfil:       ${GREEN}$GPU_PROFILE${NC}"
echo -e "  Aceleracao:   ${GREEN}$ACCEL_MODE${NC}"
echo -e "  Batch Size:   ${GREEN}$BATCH_SIZE frames${NC}"
echo -e "  Modelos:      ${GREEN}$MODEL_TYPE${NC}"
echo ""
echo -e "  Iniciar ComfyUI:"
echo -e "    ${YELLOW}cd $COMFYUI_DIR && $PYTHON_BIN main.py --listen 0.0.0.0 --port 8188${NC}"
echo ""
echo -e "  Workflow de video:"
echo -e "    ${YELLOW}Synthid-Bypass-Video-v2.0.json${NC}"
