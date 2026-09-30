"""Reproduce the cache-poisoning failure mode: feed the 429 error text as
'evidence' and see what the 0.6B summarizer does with it. Also confirms the
gate now rejects that string, so it can never be cached again."""
import json, io, os, sys
import rag_loop as R

POISON = ("【DeepWiki:ggml-org/llama.cpp】Error processing question: Client error "
          "'429 Too Many Requests' for url 'https://api.devin.ai/ada/query' "
          "For more information check: "
          "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429")
QUESTION = "量化有什么缺点？"

proc = R.ensure_server()
try:
    gated = R.looks_like_error(POISON)
    r = R.summarize(QUESTION, [POISON], searched=True)
    v = R.verify(QUESTION, [POISON], r["response"], searched=True,
                 anchors=R.relevance_terms(QUESTION))
    out = {
        "model": os.path.basename(R.CHAT_MODEL),
        "note": "the error text below is fed in manually to show what the model "
                "does with junk evidence; the current code never caches it",
        "gate_rejects_this_text": gated,
        "evidence_used": [POISON],
        "question": QUESTION,
        "response": r["response"],
        "verify": v,
    }
finally:
    if proc is not None:
        proc.terminate()

dst = os.path.join(R.OUTPUT_DIR, "cache_poison_demo.json")
with io.open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("gate_rejects:", gated)
print("response:", r["response"][:400])
print("saved", dst)
