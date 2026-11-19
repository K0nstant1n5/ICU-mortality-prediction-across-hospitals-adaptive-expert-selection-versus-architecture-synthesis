#!/usr/bin/env bash
# Creates the Python virtual environment (.venv) with all dependencies, and optionally an R environment
# (.r-env, via micromamba, no root needed) for the ricu / YAIB-cohorts extraction step.
#
#   ./setup_env.sh                    # Python env, CUDA 12.1 PyTorch wheels
#   ./setup_env.sh --cuda cpu         # CPU-only PyTorch (laptops, CI)
#   ./setup_env.sh --with-r           # also install R 4.2 + ricu + YAIB-cohorts deps
#   ./setup_env.sh --with-mamba       # also try the fast Mamba CUDA kernels (needs nvcc on the build node)
#
# Re-running is safe: existing environments are reused and only missing packages are installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA="cu121"; WITH_R=0; WITH_MAMBA=0
TORCH_VERSION="2.4.1"
YAIB_COHORTS_COMMIT="74ac699f5d4117613f9c48bdc4e0852099c42e19"   # pinned (Nov 2025)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA="$2"; shift 2 ;;
    --with-r) WITH_R=1; shift ;;
    --with-mamba) WITH_MAMBA=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

# ---- Python -------------------------------------------------------------------------------------------
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for p in python3.11 python3.12 python3.10 python3; do
    if command -v "$p" >/dev/null && "$p" -c 'import sys; exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)'; then
      PYTHON="$p"; break
    fi
  done
fi
[[ -n "$PYTHON" ]] || { echo "Need Python 3.10-3.12 (on the cluster try: module load python/3.11)"; exit 1; }
echo ">> Using $($PYTHON --version) at $(command -v $PYTHON)"

if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if ! python -c "import torch" 2>/dev/null; then
  pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA" \
    || { echo ">> PyTorch index unreachable; falling back to the PyPI wheel (CUDA 12 build, works on CPU too)";
         pip install "torch==$TORCH_VERSION"; }
fi
pip install -e "$ROOT[dev,tracking,eda]"

if [[ "$WITH_MAMBA" == 1 ]]; then
  echo ">> Installing Mamba kernels (falls back to the pure-PyTorch scan if this fails)"
  # Prebuilt wheels first (no nvcc needed; match torch 2.4 / CUDA 12 / cxx11abi=False), then source build.
  PYTAG="cp$(python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
  WHL="cu122torch2.4cxx11abiFALSE-${PYTAG}-${PYTAG}-linux_x86_64.whl"
  { pip install einops ninja "transformers==4.44.2" \
    && pip install --no-deps \
      "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+${WHL}" \
      "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+${WHL}"; } \
  || pip install --no-build-isolation "causal-conv1d==1.4.0" "mamba-ssm==2.2.2" \
  || echo "WARNING: mamba-ssm did not install; the pure-PyTorch Mamba block will be used."
fi

# ---- YAIB-cohorts (R scripts used for extraction) -------------------------------------------------------
mkdir -p "$ROOT/third_party"
if [[ ! -d "$ROOT/third_party/YAIB-cohorts" ]]; then
  git clone https://github.com/rvandewater/YAIB-cohorts "$ROOT/third_party/YAIB-cohorts"
fi
git -C "$ROOT/third_party/YAIB-cohorts" fetch --quiet origin "$YAIB_COHORTS_COMMIT" || true
git -C "$ROOT/third_party/YAIB-cohorts" checkout --quiet "$YAIB_COHORTS_COMMIT"

# ---- R (optional) ---------------------------------------------------------------------------------------
if [[ "$WITH_R" == 1 ]]; then
  MM="$ROOT/.r-env/bin/micromamba"
  if [[ ! -x "$MM" ]]; then
    mkdir -p "$ROOT/.r-env/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT/.r-env" bin/micromamba
  fi
  export MAMBA_ROOT_PREFIX="$ROOT/.r-env/root"
  if [[ ! -d "$ROOT/.r-env/env" ]]; then
    "$MM" create -y -p "$ROOT/.r-env/env" -c conda-forge \
      "r-base=4.2" r-arrow r-data.table r-yaml r-units r-rlang r-vctrs r-remotes r-fst r-curl r-openssl
  fi
  "$MM" run -p "$ROOT/.r-env/env" Rscript "$ROOT/r/install_r_packages.R"
fi

# ---- Check ----------------------------------------------------------------------------------------------
python - <<'PY'
import torch, lightning, lightgbm, sklearn, hydra
from icu_aes_nas.core.blocks.ssm import KERNEL_AVAILABLE
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | "
      f"lightning {lightning.__version__} | lightgbm {lightgbm.__version__}")
print("Mamba kernels:", "yes" if KERNEL_AVAILABLE else "no (pure-PyTorch fallback)")
PY
echo ">> Done. Activate with: source $ROOT/.venv/bin/activate"
