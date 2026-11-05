"""Settings shared by deploy.py and test_endpoint.py.

Values come from environment variables, optionally loaded from ./config.env (KEY=VALUE lines;
see config.env.example). Real environment variables take precedence over config.env.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_env_file = ROOT / "config.env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.split(" #", 1)[0].strip().strip('"').strip("'")
            if _v:
                os.environ.setdefault(_k.strip(), _v)


def _required(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing setting {name}: set it in config.env or the environment.")
    return value


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


SUBSCRIPTION = _required("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = _required("AZURE_RESOURCE_GROUP")
WORKSPACE = _required("AZUREML_WORKSPACE")
ACR_NAME = os.environ.get("ACR_NAME", "")  # empty -> the workspace's own container registry

ENDPOINT = os.environ.get("ENDPOINT_NAME", "vllm-cls-poc")
DEPLOYMENT = os.environ.get("DEPLOYMENT_NAME", "h100")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "Standard_NC40ads_H100_v5")
INSTANCE_COUNT = int(os.environ.get("INSTANCE_COUNT", "1"))
VLLM_ARGS = os.environ.get("VLLM_ARGS", "--max-model-len 8192 --max-num-batched-tokens 32768")

VLLM_BASE_TAG = os.environ.get("VLLM_BASE_TAG", "v0.32.0-x86_64-cu129")
IMAGE_TAG = os.environ.get("IMAGE_TAG", "v1")
# Dockerfile (inside env/) used for the cloud image build: Dockerfile.baked downloads the stand-in
# checkpoint during the build; Dockerfile.byo copies your own checkpoint from env/checkpoint/.
DOCKERFILE = os.environ.get("DOCKERFILE", "Dockerfile.baked")
# true: weights live in the image (no workspace-storage access needed).
# false: register model/ as an Azure ML model and let Azure ML build env/Dockerfile.
MODEL_IN_IMAGE = _flag("MODEL_IN_IMAGE", "true")

ENVIRONMENT_NAME = "vllm-classify"
MODEL_NAME = "classifier-qwen35-0p8b"
SERVED_MODEL_NAME = "classifier"  # must match --served-model-name in env/start.sh
BENCH_ACI_NAME = os.environ.get("BENCH_ACI_NAME", f"{ENDPOINT}-bench")


def ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    # Select the CLI account by subscription so the right signed-in identity is used.
    cred = AzureCliCredential(subscription=SUBSCRIPTION, process_timeout=60)
    return MLClient(cred, SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
