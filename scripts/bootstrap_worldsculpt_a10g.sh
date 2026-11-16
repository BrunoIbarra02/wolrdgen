#!/usr/bin/env bash
set -euo pipefail

# Run on the EC2 instance through SSH or SSM. No token is persisted by this script.
# Optional variables: SCENEGEN_ROOT, HF_TOKEN.

SCENEGEN_ROOT="${SCENEGEN_ROOT:-/opt/scenegen}"
TRELLIS_DIR="$SCENEGEN_ROOT/TRELLIS.2"
WORLDSCULPT_DIR="$SCENEGEN_ROOT/WorldSculpt"
MINIFORGE_DIR="$SCENEGEN_ROOT/miniforge3"
SWAP_GIB="${SWAP_GIB:-24}"

command -v nvidia-smi >/dev/null
command -v git >/dev/null
nvidia-smi

TORCH_PROBE=python3
if [[ -x /opt/pytorch/bin/python ]]; then
  TORCH_PROBE=/opt/pytorch/bin/python
fi
GPU_ARCH="$($TORCH_PROBE -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")"
if [[ "$GPU_ARCH" != "8.6" ]]; then
  echo "Expected A10G compute capability 8.6, got $GPU_ARCH" >&2
  exit 1
fi

sudo install -d -o "$(id -u)" -g "$(id -g)" "$SCENEGEN_ROOT"
sudo apt-get update
sudo apt-get install -y build-essential curl git git-lfs ffmpeg gcc-13 g++-13 libgl1 libglib2.0-0 libjpeg-dev ninja-build

if [[ "$SWAP_GIB" -gt 0 ]] && ! swapon --show=NAME --noheadings | grep -q '^/swapfile$'; then
  sudo fallocate -l "${SWAP_GIB}G" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

if [[ ! -d "$TRELLIS_DIR/.git" ]]; then
  git clone --branch main --recursive https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
fi
if [[ ! -d "$WORLDSCULPT_DIR/.git" ]]; then
  git clone --branch main https://github.com/AlayaLab/WorldSculpt.git "$WORLDSCULPT_DIR"
fi

if ! command -v conda >/dev/null && [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
  installer="$SCENEGEN_ROOT/Miniforge3-Linux-x86_64.sh"
  curl --fail --location --retry 3 \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    --output "$installer"
  bash "$installer" -b -p "$MINIFORGE_DIR"
  rm -f "$installer"
fi
export PATH="$MINIFORGE_DIR/bin:$PATH"

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx worldsculpt; then
  installed_cuda="$(conda run -n worldsculpt nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1 || true)"
  if [[ -n "$installed_cuda" && "$installed_cuda" != 12.4* ]]; then
    echo "Replacing incompatible CUDA $installed_cuda environment with CUDA 12.4"
    conda env remove -y -n worldsculpt
  fi
fi
if ! conda env list | awk '{print $1}' | grep -qx worldsculpt; then
  conda create -y -n worldsculpt python=3.10 pip
fi
# NVIDIA's rolling channel now lets the cuda-toolkit 12.4 metapackage resolve
# to CUDA 13.x components. The immutable 12.4.1 label avoids that mismatch.
conda install -y -n worldsculpt --override-channels \
  -c nvidia/label/cuda-12.4.1 -c conda-forge \
  cuda-toolkit=12.4.1 cuda-version=12.4
# Ubuntu 26's glibc headers are newer than CUDA 12.4. Conda's GCC 13
# toolchain provides a compatible sysroot without changing the host system.
conda install -y -n worldsculpt -c conda-forge gcc_linux-64=13 gxx_linux-64=13
# CUDA's conda deactivation hook reads unset backup variables; keep nounset off
# after activation so a non-interactive SSH shell can exit cleanly.
set +u
conda activate worldsculpt
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu ${LDFLAGS:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST=8.6
nvcc --version

python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

cd "$TRELLIS_DIR"
# Installing the exact upstream wheel directly avoids flash-attn's source
# package trying to rename it across /tmp and pip's cache (EXDEV on DLAMI).
python -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
# TRELLIS clones all source extensions here and does not reuse a partial tree.
rm -rf /tmp/extensions
MAX_JOBS=4 . ./setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

cd "$WORLDSCULPT_DIR"
NATTEN_CUDA_ARCH=8.6 NATTEN_N_WORKERS=4 python -m pip install natten==0.21.0 --no-build-isolation
python -m pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
python -m pip install peft pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh \
  transformers==4.57.1 zstandard kornia timm diffusers accelerate gradio plyfile \
  matplotlib scikit-image scikit-learn fpsample iopath pycocotools ftfy "setuptools<81"
python -m pip install "huggingface_hub[cli]"

hf download AlayaLab/WorldSculpt --local-dir ./pretrained
hf download TencentARC/Pixal3D --local-dir ./pretrained/Pixal3D

python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
})
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 6)
PY

echo "Bootstrap complete at $WORLDSCULPT_DIR"
