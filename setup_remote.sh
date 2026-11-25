#!/usr/bin/env bash
set -euo pipefail

# ==============================================================
#  Synthid-Bypass - Remote Setup with GPU Auto-Detection
#  v2.1 — parallel downloads (aria2c, 16 conns/file)
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
#   HF_TOKEN=hf_xxx                  -> Hugging Face token (gated models)
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANUAL_PROFILE="${1:-auto}"
COMFYUI_DIR="${COMFYUI_DIR:-$PWD/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_TOKEN="${HF_TOKEN:-}"

# Paralelismo de downloads — RunPod suporta 16 conexoes simultaneas por arquivo
ARIA2_CONNS=16          # conexoes por arquivo (-x / -s)
ARIA2_MAX_PARALLEL=5    # arquivos grandes em paralelo ao mesmo tempo

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}=================================================="
echo -e "  Synthid-Bypass - Setup Remoto v2.1"
echo -e "==================================================${NC}"

usage() {
    echo "Uso: bash setup_remote.sh [auto|4000pro|4500pro|6000pro]"
    echo "Variaveis opcionais: COMFYUI_DIR=/caminho/ComfyUI  PYTHON_BIN=python3  HF_TOKEN=hf_xxx"
}

if [[ "${MANUAL_PROFILE}" == "-h" || "${MANUAL_PROFILE}" == "--help" ]]; then
    usage; exit 0
fi

# ------------------------------------------------------------------
# Utilitario de download: aria2c com 16 conns, fallback para wget
# ------------------------------------------------------------------
DOWNLOADER="wget"

setup_downloader() {
    if command -v aria2c >/dev/null 2>&1; then
        DOWNLOADER="aria2c"
        echo -e "${GREEN}  Downloader: aria2c (${ARIA2_CONNS} conexoes/arquivo)${NC}"
    else
        echo -e "${YELLOW}  aria2c nao encontrado — tentando instalar...${NC}"
        if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
            apt-get install -y -qq aria2 >/dev/null && DOWNLOADER="aria2c" && \
                echo -e "${GREEN}  aria2c instalado.${NC}" || \
                echo -e "${YELLOW}  Falhou — usando wget como fallback.${NC}"
        else
            echo -e "${YELLOW}  Sem permissao root para instalar aria2 — usando wget.${NC}"
        fi
    fi
}

# fetch_file <url> <destino>
# - Pula se o arquivo ja existe e tem tamanho > 0
# - Usa aria2c com 16 conexoes ou wget com retries
fetch_file() {
    local url="$1"
    local dest="$2"
    local fname
    fname="$(basename "$dest")"

    if [[ -s "$dest" ]]; then
        echo -e "  ${GREEN}[OK]${NC} $fname ja existe, pulando."
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    echo -e "  ${CYAN}↓${NC} $fname"

    local hf_header=""
    if [[ -n "$HF_TOKEN" && "$url" == *"huggingface.co"* ]]; then
        hf_header="Authorization: Bearer ${HF_TOKEN}"
    fi

    if [[ "$DOWNLOADER" == "aria2c" ]]; then
        local aria_args=(
            --continue=true
            --max-connection-per-server="${ARIA2_CONNS}"
            --split="${ARIA2_CONNS}"
            --min-split-size=5M
            --file-allocation=falloc
            --console-log-level=warn
            --summary-interval=0
            --out="$(basename "$dest")"
            --dir="$(dirname "$dest")"
        )
        [[ -n "$hf_header" ]] && aria_args+=(--header="$hf_header")
        aria2c "${aria_args[@]}" "$url" || {
            echo -e "${YELLOW}  AVISO: aria2c falhou em $fname — tentando wget...${NC}"
            rm -f "$dest"
            local wget_args=(-c --progress=bar:force:noscroll -O "$dest")
            [[ -n "$hf_header" ]] && wget_args+=(--header="$hf_header")
            wget "${wget_args[@]}" "$url" || { echo -e "${RED}  ERRO: falha ao baixar $fname${NC}"; rm -f "$dest"; }
        }
    else
        local wget_args=(-c --tries=5 --progress=bar:force:noscroll -O "$dest")
        [[ -n "$hf_header" ]] && wget_args+=(--header="$hf_header")
        wget "${wget_args[@]}" "$url" || {
            echo -e "${RED}  ERRO: falha ao baixar $fname${NC}"
            rm -f "$dest"
        }
    fi
}

# fetch_parallel <url1> <dest1> [<url2> <dest2> ...]
# Dispara todos os downloads em paralelo e aguarda todos terminarem.
fetch_parallel() {
    local pids=()
    local pairs=("$@")

    local i=0
    while [[ $i -lt ${#pairs[@]} ]]; do
        local url="${pairs[$i]}"
        local dest="${pairs[$((i+1))]}"
        fetch_file "$url" "$dest" &
        pids+=($!)
        (( i += 2 ))
    done

    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || (( failed++ )) || true
    done

    if [[ $failed -gt 0 ]]; then
        echo -e "${YELLOW}  AVISO: $failed download(s) falharam. Verifique os arquivos ausentes.${NC}"
    fi
}

# ------------------------------------------------------------------
# GPU Detection
# ------------------------------------------------------------------
detect_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}AVISO: nvidia-smi nao encontrado.${NC}"
        GPU_NAME="Unknown"; GPU_VRAM_MB="0"; GPU_VRAM_GB="0.0"
        GPU_DRIVER="N/A"; GPU_CUDA="N/A"; GPU_COUNT="0"
        return
    fi

    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | xargs)"
    GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
    GPU_VRAM_GB="$("$PYTHON_BIN" -c "print(round(float('${GPU_VRAM_MB:-0}') / 1024.0, 1))")"
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

# ------------------------------------------------------------------
# Profile Selection
# ------------------------------------------------------------------
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
                exit 1 ;;
        esac
        echo -e "${YELLOW}Perfil forcado manualmente:${NC} $GPU_PROFILE"
    else
        local vram_int name_lc
        vram_int="${GPU_VRAM_MB%.*}"; vram_int="${vram_int:-0}"
        name_lc="$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')"

        if   [[ "$name_lc" == *"6000"* || "$name_lc" == *"h100"* || "$vram_int" -ge 48000 ]]; then
            GPU_PROFILE="6000pro"
        elif [[ "$name_lc" == *"4500"* || "$name_lc" == *"5090"* || "$vram_int" -ge 22000 ]]; then
            GPU_PROFILE="4500pro"
        elif [[ "$name_lc" == *"4000"* || "$vram_int" -ge 17000 ]]; then
            GPU_PROFILE="4000pro"
        else
            GPU_PROFILE="4000pro"
            echo -e "${YELLOW}AVISO: GPU com ${GPU_VRAM_GB}GB VRAM — perfil conservador (4000pro).${NC}"
        fi
        echo -e "${GREEN}Perfil auto-detectado:${NC} $GPU_PROFILE"
    fi

    case "$GPU_PROFILE" in
        6000pro)
            BATCH_SIZE=128; MODEL_TYPE="bf16+gguf"; ACCEL_MODE="MAXIMUM"
            VAE_TILING="false"; MAX_RES=2048; DOWNLOAD_BF16="true"
            echo -e "${GREEN}RTX 6000 Pro (96GB) -> batch=$BATCH_SIZE, acel=$ACCEL_MODE, GGUF + bf16${NC}" ;;
        4500pro)
            BATCH_SIZE=24; MODEL_TYPE="gguf_q4"; ACCEL_MODE="STANDARD"
            VAE_TILING="true"; MAX_RES=1536; DOWNLOAD_BF16="false"
            echo -e "${GREEN}RTX 4500 Pro (24GB) -> batch=$BATCH_SIZE, acel=$ACCEL_MODE, GGUF Q4_K_M${NC}" ;;
        4000pro)
            BATCH_SIZE=12; MODEL_TYPE="gguf_q4"; ACCEL_MODE="STANDARD"
            VAE_TILING="true"; MAX_RES=1024; DOWNLOAD_BF16="false"
            echo -e "${GREEN}RTX 4000 Pro (20GB) -> batch=$BATCH_SIZE, acel=$ACCEL_MODE, GGUF Q4_K_M${NC}" ;;
    esac
    echo ""
}

# ------------------------------------------------------------------
# Step 1 — System tools
# ------------------------------------------------------------------
ensure_system_tools() {
    echo -e "${CYAN}[1/5] Verificando ferramentas base (git, wget, ffmpeg, aria2)...${NC}"
    if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
        local pkgs=()
        command -v ffmpeg >/dev/null 2>&1   || pkgs+=(ffmpeg)
        command -v wget   >/dev/null 2>&1   || pkgs+=(wget)
        command -v git    >/dev/null 2>&1   || pkgs+=(git)
        command -v aria2c >/dev/null 2>&1   || pkgs+=(aria2)
        if [[ ${#pkgs[@]} -gt 0 ]]; then
            echo "  Instalando: ${pkgs[*]}"
            apt-get update -qq && apt-get install -y -qq "${pkgs[@]}" >/dev/null
        fi
    else
        command -v ffmpeg >/dev/null 2>&1 || \
            echo -e "${YELLOW}  AVISO: ffmpeg nao encontrado. MP4 com audio pode falhar.${NC}"
    fi
    setup_downloader
}

# ------------------------------------------------------------------
# Step 2 — ComfyUI
# ------------------------------------------------------------------
install_comfyui() {
    echo -e "${CYAN}[2/5] Instalando/Verificando ComfyUI em $COMFYUI_DIR...${NC}"
    if [[ ! -d "$COMFYUI_DIR" ]]; then
        git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    else
        echo "  ComfyUI ja existe em $COMFYUI_DIR, mantendo."
    fi

    cd "$COMFYUI_DIR"
    # pip installs em um unico invocation (mais rapido que varios chamados)
    "$PYTHON_BIN" -m pip install -q -r requirements.txt
    "$PYTHON_BIN" -m pip install -q opencv-python-headless gguf

    "$PYTHON_BIN" -c "import xformers" >/dev/null 2>&1 || \
        "$PYTHON_BIN" -m pip install -q xformers || \
        echo -e "${YELLOW}  xformers opcional nao instalado; PyTorch SDPA sera usado.${NC}"

    echo -e "${GREEN}ComfyUI pronto.${NC}"
}

# ------------------------------------------------------------------
# Step 3 — Custom nodes (clones em paralelo)
# ------------------------------------------------------------------
install_custom_nodes() {
    echo -e "${CYAN}[3/5] Instalando custom nodes (clones paralelos)...${NC}"
    mkdir -p "$COMFYUI_DIR/custom_nodes"
    cd "$COMFYUI_DIR/custom_nodes"

    local REPOS=(
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

    # Clona todos em paralelo (--depth 1 = apenas ultimo commit, muito mais rapido)
    local clone_pids=()
    for repo in "${REPOS[@]}"; do
        local dir_name
        dir_name="$(basename "$repo" .git)"
        if [[ ! -d "$dir_name" ]]; then
            echo "  Clonando $dir_name..."
            git clone --depth 1 --quiet "$repo" "$dir_name" &
            clone_pids+=($!)
        else
            echo "  $dir_name ja existe, pulando."
        fi
    done

    # Aguarda todos os clones
    for pid in "${clone_pids[@]}"; do
        wait "$pid" || echo -e "${YELLOW}  AVISO: um clone falhou (pid $pid).${NC}"
    done

    # Instala requirements de todos os nodes de uma vez (concatena tudo)
    local all_reqs=""
    for dir in */; do
        [[ -f "${dir}requirements.txt" ]] && all_reqs+=" ${dir}requirements.txt"
    done
    if [[ -n "$all_reqs" ]]; then
        echo "  Instalando dependencias de todos os custom nodes..."
        # shellcheck disable=SC2086
        "$PYTHON_BIN" -m pip install -q $(cat $all_reqs | sort -u | grep -v "^#" | tr '\n' ' ') || \
            echo -e "${YELLOW}  Algumas deps opcionais falharam (normal).${NC}"
    fi

    # install.py individuais (geralmente rapidos, nao paralelizar pois modificam o env)
    for dir in */; do
        [[ -f "${dir}install.py" ]] && "$PYTHON_BIN" "${dir}install.py" || true
    done

    # Copia o node proprio do projeto
    rm -rf "$COMFYUI_DIR/custom_nodes/Comfyui-SynthidBypass"
    cp -r "$SCRIPT_DIR/custom_nodes/Comfyui-SynthidBypass" \
           "$COMFYUI_DIR/custom_nodes/Comfyui-SynthidBypass"
    echo -e "${GREEN}Comfyui-SynthidBypass instalado.${NC}"
}

# ------------------------------------------------------------------
# Step 4 — Modelos (todos em paralelo com aria2c 16 conns)
# ------------------------------------------------------------------
download_models() {
    echo -e "${CYAN}[4/5] Baixando modelos em paralelo (${ARIA2_CONNS} conns/arquivo)...${NC}"
    cd "$COMFYUI_DIR"
    mkdir -p models/{diffusion_models,clip,text_encoders,vae,model_patches,diffusion_model_patches,loras,ultralytics/bbox,sams}

    # ------- Modelos GGUF Q4 (todos os perfis) -------
    local base_models=(
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf"
            "models/diffusion_models/qwen-image-2512-Q4_K_M.gguf"

        "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
            "models/clip/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"

        "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K_M.gguf"
            "models/diffusion_models/z_image_turbo-Q4_K_M.gguf"

        "https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-imatrix-IQ4_XS.gguf"
            "models/clip/Qwen_3_4b-imatrix-IQ4_XS.gguf"
    )

    # ------- Modelos compartilhados (VAE, ControlNet, LoRA, detectores) -------
    local shared_models=(
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"
            "models/vae/qwen_image_vae.safetensors"

        "https://huggingface.co/Comfy-Org/Z-Image-ComfyUI/resolve/main/split_files/vae/ae.safetensors"
            "models/vae/ae.safetensors"

        "https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets/resolve/main/split_files/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors"
            "models/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors"

        "https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
            "models/loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"

        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
            "models/ultralytics/bbox/yolov8n-face.pt"

        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            "models/sams/sam_vit_b_01ec64.pth"
    )

    echo -e "  ${CYAN}Baixando modelos GGUF Q4 e compartilhados em paralelo...${NC}"
    fetch_parallel "${base_models[@]}" "${shared_models[@]}"

    # Symlink do ControlNet para caminho alternativo
    ln -sf ../model_patches/qwen_image_canny_diffsynth_controlnet.safetensors \
        models/diffusion_model_patches/qwen_image_canny_diffsynth_controlnet.safetensors 2>/dev/null || true

    # ------- Pesos bf16 completos (apenas RTX 6000 Pro 96 GB) -------
    if [[ "${DOWNLOAD_BF16}" == "true" ]]; then
        echo -e "  ${YELLOW}Baixando pesos bf16 completos (RTX 6000 Pro 96GB) em paralelo...${NC}"
        local bf16_models=(
            "https://huggingface.co/Comfy-Org/Qwen-Image/resolve/main/split_files/diffusion_models/qwen_image_2512_bf16.safetensors"
                "models/diffusion_models/qwen_image_2512_bf16.safetensors"

            "https://huggingface.co/Comfy-Org/Qwen-Image/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
                "models/text_encoders/qwen_2.5_vl_7b.safetensors"

            "https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
                "models/diffusion_models/z_image_turbo_bf16.safetensors"

            "https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
                "models/text_encoders/qwen_3_4b.safetensors"
        )
        fetch_parallel "${bf16_models[@]}"
    fi

    echo -e "${GREEN}Modelos baixados.${NC}"
}

# ------------------------------------------------------------------
# Step 5 — Workflows e config
# ------------------------------------------------------------------
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
    "downloader": "$DOWNLOADER",
    "aria2_conns_per_file": $ARIA2_CONNS,
    "install_date": "$(date -Iseconds)"
}
EOF
    echo -e "${GREEN}Config salva em $COMFYUI_DIR/synthid_gpu_config.json.${NC}"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
detect_gpu
determine_profile
ensure_system_tools
install_comfyui
install_custom_nodes
download_models
copy_workflows_and_config

echo ""
echo -e "${CYAN}${BOLD}=================================================="
echo -e "  Setup completo!"
echo -e "==================================================${NC}"
echo -e "  ComfyUI Dir:   ${GREEN}$COMFYUI_DIR${NC}"
echo -e "  GPU:           ${GREEN}$GPU_NAME (${GPU_VRAM_GB} GB)${NC}"
echo -e "  Perfil:        ${GREEN}$GPU_PROFILE${NC}"
echo -e "  Aceleracao:    ${GREEN}$ACCEL_MODE${NC}"
echo -e "  Batch Size:    ${GREEN}$BATCH_SIZE frames${NC}"
echo -e "  Modelos:       ${GREEN}$MODEL_TYPE${NC}"
echo -e "  Downloader:    ${GREEN}$DOWNLOADER (${ARIA2_CONNS} conns/arquivo)${NC}"
echo ""
echo -e "  Iniciar ComfyUI:"
echo -e "    ${YELLOW}cd $COMFYUI_DIR && $PYTHON_BIN main.py --listen 0.0.0.0 --port 8188${NC}"
echo ""
echo -e "  Workflow de imagem:  ${YELLOW}Synthid-Bypass-v2.0.json${NC}"
echo -e "  Workflow de video:   ${YELLOW}Synthid-Bypass-Video-v2.0.json${NC}"
