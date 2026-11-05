# Serving image without weights, for MODEL_IN_IMAGE=false: Azure ML builds this file in the cloud and
# mounts the registered model at $AZUREML_MODEL_DIR (requires reachable workspace storage).
FROM vllm/vllm-openai:v0.32.0-x86_64-cu129

# The cu129 image ships a CUDA-13 torchcodec build (needs libnvrtc.so.13) that crashes vLLM on import.
RUN pip uninstall -y torchcodec

ENV VLLM_ARGS="--max-model-len 8192"
COPY score_alias.py /opt/score_alias.py
ENV PYTHONPATH=/opt
COPY start.sh /start.sh
ENTRYPOINT ["bash", "/start.sh"]
