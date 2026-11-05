# vLLM sequence-classifier on Azure ML managed online endpoints: deploy and benchmark

This repo deploys a **fine-tuned small LLM classifier** (a decoder backbone with a sequence-classification head) on an
**Azure ML managed online endpoint** and benchmarks throughput and latency.

The server is vLLM in classify mode (`--runner pooling --convert classify`). It does one prefill pass per document with
no token generation, and `/classify` returns softmax probabilities over the labels.

The default model is a stand-in: Qwen3.5-0.8B (text-only) with a 7-class head. It uses the same architecture and
on-disk format as a checkpoint fine-tuned with Hugging Face `AutoModelForSequenceClassification`, so you can swap in
your own checkpoint (see [Bring your own checkpoint](#bring-your-own-checkpoint)).

Everything is built in Azure: images are built by ACR Tasks, and the benchmark client runs on Azure Container
Instances in the endpoint's region. No local Docker or GPU is needed.

## Results (1× Standard_NC40ads_H100_v5, westus2)

vLLM v0.32, bf16, `--max-model-len 8192 --max-num-batched-tokens 32768`. Synthetic documents, unique per request. The
client is in the same region and calls the endpoint's `/score` URL through the Azure ML gateway (TLS + key auth).
Each test point runs for 30 s with zero errors. Raw data is in [results/](results/).

| Tokens/doc | Peak docs/s | Docs/day (sustained) | p50 / p99 at the knee | Single-request p50 |
|---:|---:|---:|---:|---:|
| 512  | ~500 | ~43M  | 65 / 76 ms (concurrency 32)   | 19 ms |
| 1024 | ~245 | ~21M  | 132 / 141 ms (concurrency 32) | 38 ms |
| 2048 | ~117 | ~10M  | 269 / 301 ms (concurrency 32)* | 42 ms* |
| 4096 | ~61  | ~5.2M | 135 / 140 ms (concurrency 8)  | 44 ms |

\* The 2048-token row comes from the run with default args (`--max-model-len 8192` only), which is 3–10% slower overall.

- **The GPU is the limit** at about 250K prompt tokens/s. Past the knee, extra concurrency only adds queueing.
- **Batching 16 docs per request** gives the same docs/s. Use it to stay under the endpoint's default limit of 500 requests/s.
- **Sizing example:** 2M docs/day is about 23 docs/s on average. One H100 has about 10× headroom at 1K tokens/doc and
  about 2.6× at 4K. Run 2 or more instances for HA.
- **Cost:** about $6.98/hr per instance at the Linux pay-as-you-go list price in westus2 (about $5.1K/month).
  Reserved instances lower this to about $5.24/hr (1-year) or about $3.84/hr (3-year). Check current pricing for
  your region.

## Prerequisites

- An Azure ML workspace in a region with H100 capacity, with quota for `Standard NCadsH100v5 Family` vCPUs.
  Managed online endpoints reserve 20% extra quota, so 1 instance needs `ceil(1.2) × 40 = 80` vCPUs.
- The workspace's container registry, or another ACR set as `ACR_NAME`. The endpoint pulls with the workspace identity.
- Azure CLI (logged in: `az login`) and Python 3.10+.
- For `deploy.py bench` only: the ACR admin user must be enabled, because ACI pulls the benchmark image with it
  (`az acr update -n <acr> --admin-enabled true`). Alternatively, run `bench/bench.py` from any VM in the region.

```powershell
git clone https://github.com/james-tn/azureml-vllm-classifier-benchmark
cd azureml-vllm-classifier-benchmark
python -m venv .venv; .venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy config.env.example config.env                      # then edit subscription, resource group, workspace
```

## Deploy, benchmark, tear down

```powershell
python deploy.py all          # base image import, image build, environment, endpoint, deployment (~45-60 min)
python test_endpoint.py --azure   # probabilities + accuracy on 7 sample docs, plus a small load test from your machine
python deploy.py bench-image  # build the benchmark client image (~3 min)
python deploy.py bench        # full sweep on ACI next to the endpoint (~40 min); results in bench/outputs/
python deploy.py bench --lengths 1024 --concurrency 1 32 --duration 20   # quick run
python deploy.py destroy      # delete endpoint, ACI and images; archive environments (--base also deletes the vLLM base image)
```

| Step | What it does |
|---|---|
| `base` | Imports `vllm/vllm-openai:<VLLM_BASE_TAG>` from Docker Hub into ACR (about 15 min, once). If Docker Hub rate-limits, run `az acr import` yourself with `--username/--password`. |
| `image` | `az acr build` of `env/$DOCKERFILE`. The default `Dockerfile.baked` downloads the stand-in base model at a pinned revision and builds the classifier checkpoint inside the image. |
| `env` | Registers Azure ML environment `vllm-classify` pointing at the image, with routes `/health` (probes) and `/classify` (scoring) on port 8000. |
| `endpoint` | Creates the managed online endpoint (key auth). |
| `deployment` | Creates the GPU deployment (`INSTANCE_TYPE` × `INSTANCE_COUNT`, `VLLM_ARGS`, `max_concurrent_requests_per_instance=1024`) and routes 100% of traffic to it. |

Settings are documented in [config.env.example](config.env.example). Environment variables override `config.env`.

## Calling the endpoint

The request body is vLLM's `/classify` API. `input` takes one string or a list (several docs per request). Format
each document with the same prompt template used in training.

```python
import requests
r = requests.post(SCORING_URI, headers={"Authorization": f"Bearer {KEY}"},
                  json={"model": "classifier", "input": ["<prompt with document 1>", "<prompt with document 2>"]})
for d in r.json()["data"]:
    print(d["label"], d["probs"])        # probs follow config.json id2label order and sum to 1
```

## Bring your own checkpoint

Your checkpoint must contain the **full model weights**, not only a LoRA adapter (merge adapters first). It needs:

- `config.json` with `architectures: ["<Arch>ForSequenceClassification"]`, `id2label`/`label2id` and `num_labels`;
- `*.safetensors`, including the classification head (`score.weight`);
- the tokenizer files.

1. Copy the checkpoint folder to `env/checkpoint/`. It is git-ignored.
2. In `config.env`, set `DOCKERFILE=Dockerfile.byo` and a new `IMAGE_TAG`.
3. Run `python deploy.py image`, then `python deploy.py env`, then `python deploy.py deployment` (or `all` on a fresh workspace).
4. Update `LABELS` and `PROMPT` in `test_endpoint.py` and `bench/bench.py` to match your training format.

To check serving parity, run `python hf_reference.py env/checkpoint` on any CUDA machine. It prints transformers
probabilities for the same sample docs as `test_endpoint.py`, so you can compare the two.

**Model asset instead of baked weights (`MODEL_IN_IMAGE=false`):** this keeps weights out of the image. `deploy.py`
registers `model/` as an Azure ML model and lets Azure ML build `env/Dockerfile`. The model is mounted at
`$AZUREML_MODEL_DIR`. This requires the workspace storage account to be reachable from where you run `deploy.py` and
from the Azure ML build. To produce `model/` for the stand-in: download the base model to `base/`, then run
`python env/build_checkpoint.py`.

## Notes

- `env/score_alias.py` also serves `/classify` at `/score`. The managed-endpoint gateway sometimes forwarded `/score`
  unchanged, which caused 404/424 errors.
- The vLLM cu129 image ships a CUDA-13 `torchcodec` that breaks imports, so the Dockerfiles uninstall it. A text
  classifier doesn't need it.
- `--max-num-batched-tokens 32768` was 3–10% faster than the default 8192 for prefill-only traffic.
- Documents longer than `--max-model-len` are rejected. Truncate or chunk them, or raise the limit.
- Default endpoint limits: 500 requests/s, 500 connections and 5 MBPS. You can request an increase.

## Repo layout

```
config.py / config.env.example   settings
deploy.py                        build, deploy, bench and destroy orchestration
test_endpoint.py                 functional + small load test
hf_reference.py                  transformers reference probabilities (parity check)
env/                             serving images (Dockerfile, Dockerfile.baked, Dockerfile.byo), start.sh,
                                 score_alias.py, build_checkpoint.py
bench/                           benchmark client (bench.py + Dockerfile)
results/                         raw benchmark results from the run above
```
