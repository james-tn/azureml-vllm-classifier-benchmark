"""Build, deploy, benchmark and tear down a vLLM sequence-classification server on an Azure ML
managed online endpoint. Images are built in Azure (ACR Tasks); nothing is built locally.

Usage: python deploy.py <step> [args]
  base         import the vLLM base image from Docker Hub into ACR (once; ~15 min)
  image        build the serving image in ACR from env/$DOCKERFILE (~15-25 min)
  env          register the Azure ML environment that points at the image
  model        register model/ as an Azure ML model (only when MODEL_IN_IMAGE=false)
  endpoint     create the managed online endpoint (key auth)
  deployment   create the GPU deployment and route 100% of traffic to it (~15-20 min)
  all          base, image, env, model, endpoint, deployment
  bench-image  build the benchmark client image in ACR
  bench [...]  run bench/bench.py on Azure Container Instances in the workspace region;
               extra args are passed through to bench.py (e.g. --lengths 1024 --concurrency 1 32)
  destroy      delete the endpoint, benchmark container and ACR images; archive environment/model
               versions (add --base to also delete the imported vLLM base image)
"""
import json
import os
import subprocess
import sys
import time

import config as cfg

WIN = sys.platform == "win32"
_ml = None
_acr = None


def ml():
    global _ml
    if _ml is None:
        _ml = cfg.ml_client()
    return _ml


def acr():
    """ACR_NAME, or the registry attached to the Azure ML workspace."""
    global _acr
    if not _acr:
        registry = cfg.ACR_NAME or (ml().workspaces.get(cfg.WORKSPACE).container_registry or "").split("/")[-1]
        if not registry:
            raise SystemExit("The workspace has no container registry yet; set ACR_NAME in config.env.")
        _acr = registry
    return _acr


def az(*args, capture=False, check=True):
    """Run an az CLI command against the configured subscription."""
    cmd = ["az", *args, "--subscription", cfg.SUBSCRIPTION]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")  # avoid cp1252 crashes in az on Windows
    r = subprocess.run(cmd, shell=WIN, cwd=cfg.ROOT, env=env, text=True, encoding="utf-8", errors="replace",
                       capture_output=capture)
    if check and r.returncode != 0:
        raise SystemExit(f"az {args[0]} {args[1]} failed ({r.returncode}): {(r.stderr or '').strip()[-2000:]}")
    return (r.stdout or "") if capture else r.returncode


def server_image():
    return f"{acr()}.azurecr.io/vllm-cls/server:{cfg.IMAGE_TAG}"


def bench_image():
    return f"{acr()}.azurecr.io/vllm-cls/bench:{cfg.IMAGE_TAG}"


def import_base_image():
    repo, tag = "vllm/vllm-openai", cfg.VLLM_BASE_TAG
    tags = az("acr", "repository", "show-tags", "-n", acr(), "--repository", repo, "-o", "tsv",
              capture=True, check=False)
    if tag in tags.split():
        print(f"base image {repo}:{tag} already in {acr()}")
        return
    print(f"importing docker.io/{repo}:{tag} into {acr()} (~15 min) ...", flush=True)
    az("acr", "import", "-n", acr(), "--source", f"docker.io/{repo}:{tag}", "--image", f"{repo}:{tag}")
    print("base image imported")


def acr_build(image, dockerfile, context, *extra):
    """Cloud build with ACR Tasks; blocks until the run finishes."""
    print(f"building {image} in ACR from {dockerfile} ...", flush=True)
    t0 = time.time()
    out = az("acr", "build", "-r", acr(), "--timeout", "7200", "--no-logs", "-o", "json",
             "-t", image.split(".azurecr.io/")[1], "-f", dockerfile, *extra, context, capture=True)
    run = json.loads(out[out.find("{"):]) if "{" in out else {}
    status = run.get("status")
    if status != "Succeeded":
        raise SystemExit(f"ACR build {run.get('runId')} ended with status {status}; "
                         f"see: az acr task logs -r {acr()} --run-id {run.get('runId')}")
    print(f"built {image} (run {run.get('runId')}, {(time.time() - t0) / 60:.1f} min)")


def build_server_image():
    base = f"{acr()}.azurecr.io/vllm/vllm-openai:{cfg.VLLM_BASE_TAG}"
    acr_build(server_image(), f"env/{cfg.DOCKERFILE}", "env", "--build-arg", f"BASE_IMAGE={base}")


def environment():
    from azure.ai.ml.entities import BuildContext, Environment

    routes = {
        "liveness_route": {"port": 8000, "path": "/health"},
        "readiness_route": {"port": 8000, "path": "/health"},
        "scoring_route": {"port": 8000, "path": "/classify"},
    }
    desc = "vLLM classify-mode server for a sequence-classification checkpoint"
    if cfg.MODEL_IN_IMAGE:
        return Environment(name=cfg.ENVIRONMENT_NAME, description=desc + " (weights in image)",
                           image=server_image(), inference_config=routes)
    # Azure ML builds env/Dockerfile in the cloud; needs network access to the workspace storage account.
    return Environment(name=cfg.ENVIRONMENT_NAME, description=desc, build=BuildContext(path=str(cfg.ROOT / "env")),
                       inference_config=routes)


def register_environment():
    env = ml().environments.create_or_update(environment())
    print("environment", env.name, env.version, env.image or "(image build runs in Azure ML)")


def register_model():
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    path = cfg.ROOT / "model"
    if not (path / "config.json").exists():
        raise SystemExit("model/config.json not found: build or copy a checkpoint into model/ first (see README).")
    m = ml().models.create_or_update(Model(name=cfg.MODEL_NAME, path=str(path), type=AssetTypes.CUSTOM_MODEL,
                                           description="Sequence-classification checkpoint (full weights)"))
    print("model", m.name, m.version)


def latest_active(assets):
    items = list(assets)
    if not items:
        return None
    return max(items, key=lambda a: int(a.version) if str(a.version).isdigit() else -1)


def create_endpoint():
    from azure.ai.ml.entities import ManagedOnlineEndpoint

    ep = ml().online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=cfg.ENDPOINT, auth_mode="key", description="vLLM classifier benchmark")
    ).result()
    print("endpoint", ep.name, ep.scoring_uri)


def create_deployment():
    from azure.ai.ml.entities import ManagedOnlineDeployment, OnlineRequestSettings, ProbeSettings

    env = latest_active(ml().environments.list(name=cfg.ENVIRONMENT_NAME))
    if env is None:
        raise SystemExit("No environment registered; run: python deploy.py env")
    model = None
    if not cfg.MODEL_IN_IMAGE:
        model = latest_active(ml().models.list(name=cfg.MODEL_NAME))
        if model is None:
            raise SystemExit("No model registered; run: python deploy.py model")
    print(f"deploying {cfg.INSTANCE_COUNT}x {cfg.INSTANCE_TYPE} with environment {env.name}:{env.version} "
          f"(~15-20 min) ...", flush=True)
    dep = ManagedOnlineDeployment(
        name=cfg.DEPLOYMENT,
        endpoint_name=cfg.ENDPOINT,
        model=f"azureml:{model.name}:{model.version}" if model else None,
        environment=f"azureml:{env.name}:{env.version}",
        instance_type=cfg.INSTANCE_TYPE,
        instance_count=cfg.INSTANCE_COUNT,
        environment_variables={"VLLM_ARGS": cfg.VLLM_ARGS},
        # The default of 1 concurrent request per instance would defeat vLLM's continuous batching.
        request_settings=OnlineRequestSettings(request_timeout_ms=90000, max_concurrent_requests_per_instance=1024),
        liveness_probe=ProbeSettings(initial_delay=600, period=10, timeout=5, failure_threshold=30),
        readiness_probe=ProbeSettings(initial_delay=120, period=10, timeout=5, failure_threshold=60),
    )
    ml().online_deployments.begin_create_or_update(dep).result()
    ep = ml().online_endpoints.get(cfg.ENDPOINT)
    ep.traffic = {cfg.DEPLOYMENT: 100}
    ml().online_endpoints.begin_create_or_update(ep).result()
    print("deployment ready; traffic", ml().online_endpoints.get(cfg.ENDPOINT).traffic)


def build_bench_image():
    acr_build(bench_image(), "bench/Dockerfile", "bench")


def run_bench(bench_args):
    """Run the benchmark client on ACI next to the endpoint, then collect its results."""
    ws = ml().workspaces.get(cfg.WORKSPACE)
    ep = ml().online_endpoints.get(cfg.ENDPOINT)
    key = ml().online_endpoints.get_keys(cfg.ENDPOINT).primary_key
    creds = json.loads(az("acr", "credential", "show", "-n", acr(), "-o", "json", capture=True, check=False) or "{}")
    if not creds.get("passwords"):
        raise SystemExit(f"ACR admin user is disabled; enable it (az acr update -n {acr()} --admin-enabled true) "
                         "or run bench/bench.py from any VM in the endpoint's region.")
    name = cfg.BENCH_ACI_NAME
    cmdline = " ".join(["python3", "bench.py", *bench_args])
    az("container", "delete", "-g", cfg.RESOURCE_GROUP, "-n", name, "-y", "-o", "none", capture=True, check=False)
    print(f"starting ACI {name} in {ws.location}: {cmdline}", flush=True)
    az("container", "create", "-g", cfg.RESOURCE_GROUP, "-n", name, "-l", ws.location, "--image", bench_image(),
       "--os-type", "Linux", "--cpu", "4", "--memory", "8", "--restart-policy", "Never",
       "--registry-login-server", f"{acr()}.azurecr.io", "--registry-username", creds["username"],
       "--registry-password", creds["passwords"][0]["value"],
       "--environment-variables", f"SCORING_URI={ep.scoring_uri}",
       "--secure-environment-variables", f"ENDPOINT_KEY={key}",
       "--command-line", cmdline, "-o", "none")
    t0, state = time.time(), ""
    while state != "Terminated":
        time.sleep(30)
        state = az("container", "show", "-g", cfg.RESOURCE_GROUP, "-n", name, "-o", "tsv",
                   "--query", "containers[0].instanceView.currentState.state", capture=True, check=False).strip()
        print(f"  {(time.time() - t0) / 60:5.1f} min  {state}", flush=True)
    logs = az("container", "logs", "-g", cfg.RESOURCE_GROUP, "-n", name, capture=True)
    out_dir = cfg.ROOT / "bench" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"bench_{stamp}_log.txt").write_text(logs, encoding="utf-8")
    results = [ln[len("RESULTS_JSON "):] for ln in logs.splitlines() if ln.startswith("RESULTS_JSON ")]
    print("\n".join(ln for ln in logs.splitlines() if not ln.startswith("RESULTS_JSON ")))
    if results:
        (out_dir / f"bench_{stamp}.json").write_text(json.dumps(json.loads(results[-1]), indent=2), encoding="utf-8")
        print(f"saved bench/outputs/bench_{stamp}.json")
    else:
        print("no RESULTS_JSON line in the logs; the container log above shows what happened")
    az("container", "delete", "-g", cfg.RESOURCE_GROUP, "-n", name, "-y", "-o", "none", check=False)


def destroy(args):
    from azure.core.exceptions import ResourceNotFoundError

    try:
        print(f"deleting endpoint {cfg.ENDPOINT} (~5-10 min) ...", flush=True)
        ml().online_endpoints.begin_delete(name=cfg.ENDPOINT).result()
        print("endpoint deleted")
    except ResourceNotFoundError:
        print("endpoint not found")
    az("container", "delete", "-g", cfg.RESOURCE_GROUP, "-n", cfg.BENCH_ACI_NAME, "-y", "-o", "none",
       capture=True, check=False)
    repos = ["vllm-cls/server", "vllm-cls/bench"] + (["vllm/vllm-openai"] if "--base" in args else [])
    for repo in repos:
        rc = az("acr", "repository", "delete", "-n", acr(), "--repository", repo, "-y", "-o", "none",
                check=False)
        print(f"ACR repository {repo}: {'deleted' if rc == 0 else 'not deleted (not found?)'}")
    for kind, name in (("environments", cfg.ENVIRONMENT_NAME), ("models", cfg.MODEL_NAME)):
        ops = getattr(ml(), kind)
        try:
            for asset in ops.list(name=name):
                ops.archive(name=name, version=asset.version)
                print(f"archived {kind[:-1]} {name}:{asset.version}")
        except ResourceNotFoundError:
            pass


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "help"
    args = sys.argv[2:]
    if step in ("help", "-h", "--help"):
        print(__doc__)
        return
    if step in ("base", "all") and cfg.MODEL_IN_IMAGE:
        import_base_image()
    if step in ("image", "all") and cfg.MODEL_IN_IMAGE:
        build_server_image()
    if step in ("env", "all"):
        register_environment()
    if step in ("model", "all") and not cfg.MODEL_IN_IMAGE:
        register_model()
    if step in ("endpoint", "all"):
        create_endpoint()
    if step in ("deployment", "all"):
        create_deployment()
    if step == "bench-image":
        build_bench_image()
    if step == "bench":
        run_bench(args)
    if step == "destroy":
        destroy(args)
    if step not in ("base", "image", "env", "model", "endpoint", "deployment", "all", "bench-image", "bench",
                    "destroy"):
        raise SystemExit(f"unknown step {step!r}\n{__doc__}")


if __name__ == "__main__":
    main()
