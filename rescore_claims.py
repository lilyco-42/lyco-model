"""Re-score every committed run record with the current checker and list the
rows whose verdict flips, so the widened claim check can be audited against
answers we already judged by hand.

Run: python rescore_claims.py   (writes results/rescore_claims.json)
"""
import json, io, os
import rag_loop as R

ARTIFACTS = ["rag_loop_live.json", "rag_loop_memory.json",
             "rag_loop_0p6b_v2.json", "rag_loop_1p7b_v2.json"]
K = ("independent", "memory", "memory_control")


def rescore(d, label):
    flips, passes, fails = [], 0, 0
    for k in K:
        pool = ""
        for r in d[k]:
            prior = [pool] if pool else ()
            v = R.verify(r["retrieval_query"], r["evidence"], r["response"],
                         searched=r["routed_search"],
                         anchors=R.relevance_terms(r["question"]),
                         prior_evidence=prior)
            if v["pass"]:
                passes += 1
            else:
                fails += 1
            if v["pass"] != r["verify"]["pass"]:
                flips.append({"artifact": label, "scenario": k,
                              "question": r["question"],
                              "old": [r["verify"]["pass"], r["verify"]["reason"]],
                              "new": [v["pass"], v["reason"]],
                              "invented_numbers": v.get("invented_numbers", []),
                              "unsupported_terms": v.get("unsupported_terms", []),
                              "response_head": r["response"][:180]})
            if k == "memory" and r["routed_search"] and r["evidence"]:
                pool = ("\n".join(r["evidence"]) + "\n" + pool)[:R.EVIDENCE_POOL_CHARS]
    return passes, fails, flips


out = {"artifacts": {}, "flips": []}
for name in ARTIFACTS:
    p = os.path.join(R.OUTPUT_DIR, name)
    if not os.path.exists(p):
        print("missing", name)
        continue
    d = json.load(io.open(p, encoding="utf-8"))
    passes, fails, flips = rescore(d, name)
    n = passes + fails
    out["artifacts"][name] = {"model": d.get("model", "chat_slm_qwen3_0p6b-Q4_K_M.gguf"), "rows": n,
                              "old_pass": sum(1 for k in K for r in d[k]
                                              if r["verify"]["pass"]),
                              "new_pass": passes, "new_fail": fails,
                              "flips": len(flips)}
    out["flips"].extend(flips)
    print(f"{name:26s} rows={n:2d}  old pass="
          f"{sum(1 for k in K for r in d[k] if r['verify']['pass']):2d} -> "
          f"new pass={passes:2d}  flips={len(flips)}")

print(f"\n=== {len(out['flips'])} flips: audit each ===")
for f in out["flips"]:
    print(f"- [{f['artifact']}] {f['scenario']} :: {f['question']}")
    print(f"    {f['old'][0]}/{f['old'][1]}  ->  {f['new'][0]}/{f['new'][1]}")
    print(f"    invented={f['invented_numbers']} unsup={f['unsupported_terms']}")
    print(f"    resp: {f['response_head']}")

with io.open(os.path.join(R.OUTPUT_DIR, "rescore_claims.json"), "w",
             encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\nsaved results/rescore_claims.json")
