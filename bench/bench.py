"""Throughput/latency sweep for the vLLM /classify endpoint (one GPU instance).

Every request carries a unique synthetic document, so no caching can flatter the numbers.
Each level runs for a fixed duration with a closed loop of N concurrent clients.

  python bench.py --url <scoring_uri> --key <key> [--lengths 512 1024 2048 4096] [--concurrency 1 8 32 64 128 256]
Runs from any machine; run it in the endpoint's region (python deploy.py bench) for gateway-only latency.
Env vars SCORING_URI / ENDPOINT_KEY are used when --url/--key are omitted.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time

import aiohttp

MODEL = "classifier"  # --served-model-name in env/start.sh

LABELS = ["NDA", "Employment", "Lease", "Loan", "Purchase", "Invoice", "Other"]
PROMPT = (
    "<|im_start|>user\nClassify the document into exactly one category: "
    + ", ".join(LABELS)
    + ".\n\nDocument:\n{doc}\n\nAnswer with the category name only.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
CORPUS = (
    "agreement party parties shall hereby confidential information disclose receiving lender borrower principal "
    "interest rate payment monthly due default landlord tenant premises rent deposit term lease employee employer "
    "salary bonus position start date benefits at-will seller buyer purchase price vehicle title delivery invoice "
    "bill subtotal tax total remit net thirty days services hours quantity amount signature witness notary state "
    "law governing jurisdiction arbitration indemnify liability warranty termination notice effective date section "
    "clause schedule exhibit amendment assignment successor waiver severability entire counterpart electronic"
).split()


def make_doc(rng, n_words):
    return " ".join(rng.choice(CORPUS) for _ in range(n_words))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


async def post(sess, url, headers, docs):
    body = {"model": MODEL, "input": [PROMPT.format(doc=d) for d in docs]}
    t0 = time.perf_counter()
    async with sess.post(url, headers=headers, json=body) as r:
        text = await r.text()
        dt = time.perf_counter() - t0
        if r.status != 200:
            return r.status, dt, 0, text[:300]
        js = json.loads(text)
        return 200, dt, js.get("usage", {}).get("prompt_tokens", 0), js


async def calibrate(sess, url, headers, rng):
    """Return tokens-per-word for the synthetic corpus and the fixed prompt overhead."""
    s0, _, t0, i0 = await post(sess, url, headers, [make_doc(rng, 10)])
    s1, _, t1, i1 = await post(sess, url, headers, [make_doc(rng, 1010)])
    if s0 != 200 or s1 != 200:
        raise SystemExit(f"calibration failed: {s0} {i0} / {s1} {i1}")
    per_word = (t1 - t0) / 1000
    return per_word, t0 - 10 * per_word


async def run_level(sess, url, headers, conc, n_words, batch, duration, warmup, seed):
    rng = random.Random(seed)
    measure_from = time.perf_counter() + warmup
    stop_at = measure_from + duration
    lat, toks, errors, done_docs = [], 0, {}, 0
    t_first = t_last = None
    sample_err = None

    async def worker():
        nonlocal toks, done_docs, t_first, t_last, sample_err
        while time.perf_counter() < stop_at:
            docs = [make_doc(rng, n_words) for _ in range(batch)]
            start = time.perf_counter()
            try:
                status, dt, ntok, info = await post(sess, url, headers, docs)
            except Exception as e:  # timeouts, connection resets
                status, dt, ntok, info = type(e).__name__, time.perf_counter() - start, 0, str(e)[:300]
            if start < measure_from:
                continue
            if status == 200:
                lat.append(dt)
                toks += ntok
                done_docs += batch
                t_first = start if t_first is None else min(t_first, start)
                t_last = start + dt if t_last is None else max(t_last, start + dt)
            else:
                errors[str(status)] = errors.get(str(status), 0) + 1
                sample_err = sample_err or info

    await asyncio.gather(*(worker() for _ in range(conc)))
    wall = (t_last - t_first) if lat else duration
    ms = lambda v: round(v * 1000) if v is not None else None
    return {
        "concurrency": conc,
        "batch": batch,
        "requests": len(lat),
        "errors": errors,
        "sample_error": sample_err,
        "docs_per_s": round(done_docs / wall, 1),
        "tokens_per_s": int(toks / wall),
        "avg_tokens_per_doc": int(toks / max(done_docs, 1)),
        "p50_ms": ms(pct(lat, 50)),
        "p90_ms": ms(pct(lat, 90)),
        "p99_ms": ms(pct(lat, 99)),
        "mean_ms": ms(statistics.mean(lat)) if lat else None,
        "docs_per_day": int(done_docs / wall * 86400),
    }


def show(r, note=""):
    print(f"{r['concurrency']:5d} {r['batch']:5d} {r['avg_tokens_per_doc']:6d} {r['docs_per_s']:8.1f} "
          f"{r['tokens_per_s']:9d} {str(r['p50_ms']):>7} {str(r['p90_ms']):>7} {str(r['p99_ms']):>7} "
          f"{r['docs_per_day']:13,d} {r['errors'] or ''} {note}", flush=True)


async def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SCORING_URI"))
    ap.add_argument("--key", default=os.environ.get("ENDPOINT_KEY"))
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096], help="target tokens/doc")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    ap.add_argument("--batch", type=int, default=1, help="docs per request for the main sweep")
    ap.add_argument("--batch-probe", type=int, nargs="*", default=[16], help="extra runs with N docs/request")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--out", default="outputs/bench_results.json")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    MODEL = a.model
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    headers = {"Content-Type": "application/json"}
    if a.key:
        headers["Authorization"] = "Bearer " + a.key
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=120)
    results = {"url": a.url, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:
        per_word, overhead = await calibrate(sess, a.url, headers, random.Random(0))
        print(f"calibration: {per_word:.3f} tokens/word, prompt overhead {overhead:.0f} tokens", flush=True)
        results["calibration"] = {"tokens_per_word": per_word, "prompt_overhead": overhead}

        for L in a.lengths:
            n_words = max(1, int((L - overhead) / per_word))
            print(f"\n=== target ~{L} tokens/doc ===", flush=True)
            print(f"{'conc':>5} {'batch':>5} {'tok/doc':>6} {'docs/s':>8} {'tok/s':>9} {'p50ms':>7} {'p90ms':>7} "
                  f"{'p99ms':>7} {'docs/day':>13} errors", flush=True)
            best = None
            for c in a.concurrency:
                r = await run_level(sess, a.url, headers, c, n_words, a.batch, a.duration, a.warmup, L * 1000 + c)
                r["target_tokens"] = L
                results["runs"].append(r)
                show(r)
                if best is None or r["docs_per_s"] > best["docs_per_s"]:
                    best = r
                if sum(r["errors"].values()) > 0.05 * max(r["requests"], 1) or (r["p99_ms"] or 0) > 60000:
                    print(f"  stopping sweep: errors/latency limit; sample error: {r['sample_error']}", flush=True)
                    break
            for b in a.batch_probe or []:
                c = max(2, best["concurrency"] * 2 // b)
                r = await run_level(sess, a.url, headers, c, n_words, b, a.duration, a.warmup, L * 7 + b)
                r["target_tokens"] = L
                results["runs"].append(r)
                show(r, f"<- {b} docs/request")
            with open(a.out, "w") as f:  # checkpoint after each length
                json.dump(results, f, indent=2)
    print("\nsaved", a.out, flush=True)
    # Single-line copy for runners without file outputs (e.g. Azure Container Instances logs).
    print("RESULTS_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
