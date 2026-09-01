#!/usr/bin/env bash
set -euo pipefail

EVIDENCE_ROOT=/workspace/evidence
RUNNER_PATH=/workspace/run_layout_divergent_blackwell.py
RUNNER_SOURCE_SHA=4160d8af580030c305b09f85158a43aa24cea3a2
RUNNER_SHA256=564bf794fafbd14f0d1db9f32299cd051d73d6d3c7055142b36ebd77f4cb802e
RUNNER_URL="https://raw.githubusercontent.com/nya-a-cat/tilelang/${RUNNER_SOURCE_SHA}/benchmark/research/run_layout_divergent_blackwell.py"
CONTAINER_IMAGE="vastai/pytorch@sha256:6ee5f68a3c11bd89e9364771bf6b929d5f266c4382fb3628d751b5e89241d462"
STARTED_UNIX="$(date +%s)"

mkdir -p "${EVIDENCE_ROOT}"
exec > >(tee -a "${EVIDENCE_ROOT}/vast-onstart.log") 2>&1

finalize() {
  exit_code=$?
  trap - EXIT
  finished_unix="$(date +%s)"
  status=failed
  if [[ ${exit_code} -eq 0 ]]; then
    status=complete
  fi
  export TILELANG_ONSTART_EXIT_CODE="${exit_code}"
  export TILELANG_ONSTART_STATUS="${status}"
  export TILELANG_ONSTART_STARTED_UNIX="${STARTED_UNIX}"
  export TILELANG_ONSTART_FINISHED_UNIX="${finished_unix}"
  /venv/main/bin/python - <<'PY' || true
import json
import os
from pathlib import Path

root = Path("/workspace/evidence")
payload = {
    "schema": "tilelang-vast-onstart-lifecycle-v1",
    "status": os.environ["TILELANG_ONSTART_STATUS"],
    "exit_code": int(os.environ["TILELANG_ONSTART_EXIT_CODE"]),
    "started_unix": int(os.environ["TILELANG_ONSTART_STARTED_UNIX"]),
    "finished_unix": int(os.environ["TILELANG_ONSTART_FINISHED_UNIX"]),
    "runner_source_sha": os.environ["TILELANG_RUNNER_SOURCE_SHA"],
    "runner_sha256": os.environ["TILELANG_RUNNER_SHA256"],
    "container_image": os.environ["TILELANG_CONTAINER_IMAGE"],
    "expected_gpu_name": os.environ["TILELANG_EXPECTED_GPU_NAME"],
    "vast_offer_id": os.environ.get("TILELANG_VAST_OFFER_ID"),
}
temporary = root / "vast-lifecycle.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(root / "vast-lifecycle.json")
(root / "VAST_RUN_DONE").write_text(payload["status"] + "\n", encoding="utf-8")
PY
  echo "TILELANG_VAST_RUN_DONE status=${status} exit_code=${exit_code}"
  sync || true
  exit "${exit_code}"
}
trap finalize EXIT

export TILELANG_EVIDENCE_ROOT="${EVIDENCE_ROOT}"
export TILELANG_WORK_ROOT=/workspace/tilelang-work
export TILELANG_EXPECTED_GPU_NAME="RTX 5060 Ti"
export TILELANG_RUNNER_SOURCE_SHA="${RUNNER_SOURCE_SHA}"
export TILELANG_RUNNER_SHA256="${RUNNER_SHA256}"
export TILELANG_CONTAINER_IMAGE="${CONTAINER_IMAGE}"
export TILELANG_CONTAINER_IMAGE_DIGEST="sha256:6ee5f68a3c11bd89e9364771bf6b929d5f266c4382fb3628d751b5e89241d462"

echo "tilelang Vast Blackwell run started at ${STARTED_UNIX}"
echo "runner source ${RUNNER_SOURCE_SHA} sha256 ${RUNNER_SHA256}"
echo "container ${CONTAINER_IMAGE}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader

/venv/main/bin/python - "${RUNNER_URL}" "${RUNNER_PATH}" "${RUNNER_SHA256}" <<'PY'
import hashlib
from pathlib import Path
import sys
import urllib.request

url, destination_text, expected = sys.argv[1:]
destination = Path(destination_text)
urllib.request.urlretrieve(url, destination)
actual = hashlib.sha256(destination.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"runner SHA-256 mismatch: expected {expected}, got {actual}")
print(f"downloaded checked runner {destination} sha256={actual}")
PY

/venv/main/bin/python "${RUNNER_PATH}"
