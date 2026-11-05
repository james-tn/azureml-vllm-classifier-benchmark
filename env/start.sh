#!/bin/bash
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

# Weights baked into the image set MODEL_PATH; otherwise Azure ML mounts the registered model under
# $AZUREML_MODEL_DIR.
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH=$(dirname "$(find "${AZUREML_MODEL_DIR:?}" -name config.json | head -1)")
fi
echo "Serving classifier from ${MODEL_PATH} with args: ${VLLM_ARGS}"

# Prefill-only classification: pooling runner + seq-cls head; /classify returns softmax probabilities.
# ScoreAlias also serves /classify at /score, since the Azure ML gateway may forward /score unchanged.
exec vllm serve "${MODEL_PATH}" \
  --served-model-name classifier \
  --runner pooling --convert classify \
  --host 0.0.0.0 --port 8000 \
  --middleware score_alias.ScoreAlias \
  ${VLLM_ARGS}
