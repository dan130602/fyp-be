#!/bin/bash
# Build the vLLM environment in SCRATCH on the login node (CPU only; the login node reaches PyPI). Nothing is tuned: the verified meralion-3-asr wheel pulls the vllm and torch it pins.
#   bash $CODE/tools/origami/setup_vllm_venv.sh <path to the verified meralion_3_asr-0.1.0-py3-none-any.whl>
set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
set -u
source /common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code/tools/origami/common.sh
WHEEL=${1:?path to the wheel}
EXPECTED=$(python3 -c "import json,sys; print(json.load(open('$CODE/configs/third_party/meralion-3-asr-0.1.0.json'))['artifact']['sha256'])")
ACTUAL=$(sha256sum "$WHEEL" | cut -d' ' -f1)
echo "wheel sha256 $ACTUAL (manifest $EXPECTED)"
test "$ACTUAL" = "$EXPECTED"
test ! -e "$VENV_VLLM" || { echo "$VENV_VLLM already exists: refusing to modify it"; exit 3; }
python3 -m venv "$VENV_VLLM"
"$VENV_VLLM/bin/python" -m pip install --upgrade pip
cp "$WHEEL" "$BASE/"
"$VENV_VLLM/bin/python" -m pip install "$BASE/$(basename "$WHEEL")" "numpy>=2" "soundfile>=0.12"
"$VENV_VLLM/bin/python" -m pip freeze > "$BASE/vllm-pip-freeze.txt"
"$VENV_VLLM/bin/python" -m pip check > "$BASE/vllm-pip-check.txt" 2>&1 || { cat "$BASE/vllm-pip-check.txt"; echo "pip check FAILED: the environment is inconsistent, nothing after this runs"; exit 4; }
cat "$BASE/vllm-pip-check.txt"
cd "$CODE"
"$VENV_VLLM/bin/python" -m meralion_worker verify-environment --model-dir "$MODEL" | tee "$BASE/vllm-verify-environment.json"
# CPU checks: what the package requests and its constructor defaults equal the frozen table, before any GPU time is used
"$VENV_VLLM/bin/python" - <<'EOF' | tee "$BASE/vllm-frozen-check.txt"
from meralion_worker.decoders.frozen import FROZEN_DECODING, check_frozen
from meralion_worker.decoders.vllm_decoder import read_package_info
info = read_package_info()
frozen = FROZEN_DECODING["vllm"]
assert info["requested_sampling"] == frozen["requested_sampling"], (info["requested_sampling"], frozen["requested_sampling"])
assert info["constructor_defaults"] == frozen["constructor_defaults"], (info["constructor_defaults"], frozen["constructor_defaults"])
print("requested:", info["requested_sampling"]); print("constructor defaults:", info["constructor_defaults"]); print("FROZEN TABLE MATCHES")
import importlib.metadata as m
print("vllm", m.version("vllm"), "torch", m.version("torch"))
EOF
echo "vLLM environment ready: $VENV_VLLM"
