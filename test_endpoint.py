"""Test the vLLM /classify endpoint: probabilities, accuracy on sample docs, latency, throughput.

Local:  python test_endpoint.py --url http://localhost:8000/classify
Azure:  python test_endpoint.py --azure        (endpoint from config.env; scoring URI + key via azure-ai-ml)
Latency/throughput here include your network path; use deploy.py bench for in-region numbers.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

LABELS = ["NDA", "Employment", "Lease", "Loan", "Purchase", "Invoice", "Other"]
PROMPT = (
    "<|im_start|>user\nClassify the document into exactly one category: "
    + ", ".join(LABELS)
    + ".\n\nDocument:\n{doc}\n\nAnswer with the category name only.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

DOCS = [
    ("NDA", "MUTUAL NON-DISCLOSURE AGREEMENT. This Agreement is made between Acme Corp and Beta LLC. Each party may "
            "disclose Confidential Information to the other solely to evaluate a potential business relationship. The "
            "Receiving Party shall not disclose Confidential Information to any third party and shall return or destroy "
            "it upon request. Obligations survive for three (3) years."),
    ("Employment", "OFFER LETTER. We are pleased to offer you the position of Senior Software Engineer reporting to the "
                   "VP of Engineering. Your annual base salary will be $165,000, paid bi-weekly, plus eligibility for the "
                   "annual bonus plan and 10,000 RSUs. Employment is at-will. Please sign and return by May 1."),
    ("Lease", "RESIDENTIAL LEASE AGREEMENT. Landlord hereby leases to Tenant the premises located at 12 Oak Street, Unit "
              "4B, for a term of twelve months beginning June 1. Monthly rent is $2,400 due on the first of each month. "
              "Tenant shall pay a security deposit of $2,400. No pets without written consent of Landlord."),
    ("Loan", "PROMISSORY NOTE. For value received, Borrower promises to pay to the order of Lender the principal sum of "
             "$50,000 together with interest at a fixed rate of 6.5% per annum. Payments of $978.31 are due monthly for "
             "60 months. Upon an Event of Default the entire unpaid balance becomes immediately due."),
    ("Purchase", "VEHICLE PURCHASE AGREEMENT. Seller agrees to sell and Buyer agrees to buy a 2022 Toyota Camry, VIN "
                 "4T1B11HK5NU000000, for a total purchase price of $24,500. Buyer will pay a deposit of $2,000 at signing "
                 "and the balance at delivery. Title passes to Buyer upon full payment. Sold as-is."),
    ("Invoice", "INVOICE #INV-1042. Bill To: Contoso Ltd, 1 Microsoft Way. Date: 2026-10-15. Item: Consulting services, "
                "Qty 40 hrs @ $175.00 = $7,000.00. Subtotal $7,000.00, Tax (8.8%) $616.00, TOTAL DUE $7,616.00. "
                "Payment terms: Net 30. Please remit to account ending 4421."),
    ("Other", "MINUTES OF THE QUARTERLY ALL-HANDS MEETING. Attendees: full staff. The CEO reviewed Q3 highlights and "
              "the new office opening. HR announced the holiday party on December 12 and the updated PTO calendar. "
              "Q&A followed. Meeting adjourned at 4:05 pm."),
]


def make_session(pool):
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
    s.mount("http://", requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
    return s


def classify(sess, url, headers, docs):
    body = {"model": "classifier", "input": [PROMPT.format(doc=d) for d in docs]}
    t0 = time.perf_counter()
    r = sess.post(url, headers=headers, json=body, timeout=120)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def long_doc(n_words):
    base = " ".join(d for _, d in DOCS if _ == "Lease")
    words = (base + " ") * (n_words // len(base.split()) + 1)
    return " ".join(words.split()[:n_words])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--azure", action="store_true")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--doc-words", type=int, default=700, help="~1k tokens per doc incl. prompt")
    ap.add_argument("--out", default="test_results.json")
    a = ap.parse_args()

    headers = {"Content-Type": "application/json"}
    url = a.url
    if a.azure:
        import config as cfg

        ml = cfg.ml_client()
        url = ml.online_endpoints.get(cfg.ENDPOINT).scoring_uri
        headers["Authorization"] = "Bearer " + ml.online_endpoints.get_keys(cfg.ENDPOINT).primary_key
    if not url:
        raise SystemExit("pass --url or --azure")
    sess = make_session(max(a.concurrency) + 4)
    results = {"url": url}

    # 1) Functional: labels, probabilities, accuracy (one batched call)
    out, dt = classify(sess, url, headers, [d for _, d in DOCS])
    print(f"Batch of {len(DOCS)} docs in {dt*1000:.0f} ms; usage={out.get('usage')}")
    correct = 0
    rows = []
    for (gold, _), item in zip(DOCS, out["data"]):
        probs = item["probs"]
        pred = item.get("label") or LABELS[max(range(len(probs)), key=probs.__getitem__)]
        correct += pred == gold
        top = sorted(zip(LABELS, probs), key=lambda x: -x[1])[:3]
        print(f"  gold={gold:10} pred={pred:10} sum(p)={sum(probs):.4f} top3=" +
              ", ".join(f"{l}:{p:.3f}" for l, p in top))
        rows.append({"gold": gold, "pred": pred, "probs": dict(zip(LABELS, probs))})
    print(f"Accuracy {correct}/{len(DOCS)}")
    results["functional"] = {"accuracy": f"{correct}/{len(DOCS)}", "rows": rows, "raw_first": out["data"][0]}

    # 2) Load test with ~1k-token docs, one doc per request
    doc = long_doc(a.doc_words)
    probe, _ = classify(sess, url, headers, [doc])
    tokens = probe.get("usage", {}).get("prompt_tokens")
    print(f"\nLoad test: {tokens} prompt tokens/doc, {a.requests} requests per level")
    results["load"] = {"tokens_per_doc": tokens, "levels": []}
    for c in a.concurrency:
        n = a.requests if c > 1 else min(a.requests, 50)
        lat = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            for _, dt in ex.map(lambda _: classify(sess, url, headers, [doc]), range(n)):
                lat.append(dt)
        wall = time.perf_counter() - t0
        rps = n / wall
        lvl = {"concurrency": c, "n": n, "docs_per_s": round(rps, 1), "p50_ms": round(pct(lat, 50) * 1000),
               "p95_ms": round(pct(lat, 95) * 1000), "mean_ms": round(statistics.mean(lat) * 1000),
               "docs_per_day_at_this_rate": int(rps * 86400)}
        print(f"  conc={c:3} docs/s={rps:7.1f} p50={lvl['p50_ms']} ms p95={lvl['p95_ms']} ms "
              f"=> {lvl['docs_per_day_at_this_rate']:,} docs/day")
        results["load"]["levels"].append(lvl)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
