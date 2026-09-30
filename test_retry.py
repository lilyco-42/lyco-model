"""Offline tests for the DeepWiki retry/backoff -- no network, no model.

Run: python test_retry.py
"""
import time
import rag_loop as R

LIMIT = ("Error processing question: Client error '429 Too Many Requests' "
         "for url 'https://api.devin.ai/ada/query'")
PERMANENT = "Error processing question: repo not found"
ANSWER = "Q4_K_M is a 4-bit K-quant block format, perplexity +0.1754."


def make_call(bodies):
    """A fake transport: returns the given bodies in order and counts calls."""
    state = {"calls": 0}

    def call():
        body = bodies[min(state["calls"], len(bodies) - 1)]
        state["calls"] += 1
        return body
    call.state = state
    return call


def run(name, bodies, expect_attempts, expect_sleeps, expect_prefix):
    call = make_call(bodies)
    sleeps = []
    out = R.deepwiki_ask(None, "ggml-org/llama.cpp", "q?", tries=3,
                         sleeper=sleeps.append, call=call)
    attempts = call.state["calls"]
    ok = (attempts == expect_attempts and sleeps == expect_sleeps
          and out.startswith(expect_prefix))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"   attempts={attempts} want={expect_attempts} "
              f"sleeps={sleeps} want={expect_sleeps} out={out[:80]!r}")
    return ok


def backoff_shape():
    got = R.backoff_sleeps(tries=4, base=2.0, limit=16.0)
    want = [2.0, 4.0, 8.0]
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  backoff doubles and is bounded: {got}")
    return ok


def limit_vs_permanent():
    ok = (R.is_ratelimit(LIMIT) and not R.is_ratelimit(PERMANENT)
          and R.looks_like_error(LIMIT))
    print(f"{'PASS' if ok else 'FAIL'}  rate limit and permanent errors are told apart")
    return ok


def cache_poison_paths():
    """A cached error body must never be served back as evidence."""
    store = {}
    key = R.cache_key("deepwiki", "q?", "ggml-org/llama.cpp")
    poisoned = "【DeepWiki:ggml-org/llama.cpp】Error processing question: 429 Too Many Requests"
    real = {"calls": 0}
    orig_cache, orig_put, orig_offline = R.cache, R.cache_put, R.OFFLINE

    def fetch():
        real["calls"] += 1
        return "【DeepWiki:ggml-org/llama.cpp】Q4_K_M is a 4-bit K-quant format."
    try:
        R.cache = lambda: store
        R.cache_put = lambda k, v: store.setdefault(k, {"t": 0, "v": v})
        store[key] = {"t": time.time(), "v": poisoned}
        R.OFFLINE = False
        txt, note = R.cached_lookup("deepwiki", "q?", fetch, "ggml-org/llama.cpp")
        ok_online = txt.startswith("【DeepWiki") and "429" not in txt \
            and real["calls"] == 1 and note == "deepwiki:live"
        store[key] = {"t": time.time(), "v": poisoned}
        R.OFFLINE = True
        txt2, note2 = R.cached_lookup("deepwiki", "q?", fetch, "ggml-org/llama.cpp")
        ok_offline = txt2 == "" and note2 == "deepwiki:cache-dropped-error" \
            and real["calls"] == 1
    finally:
        R.cache, R.cache_put, R.OFFLINE = orig_cache, orig_put, orig_offline
    print(f"{'PASS' if ok_online else 'FAIL'}  poisoned cache entry is refetched when online")
    print(f"{'PASS' if ok_offline else 'FAIL'}  poisoned cache entry is dropped, not fetched, when offline")
    return ok_online and ok_offline


if __name__ == "__main__":
    results = [
        backoff_shape(),
        limit_vs_permanent(),
        cache_poison_paths(),
        # 3 rate-limited answers: 3 attempts, 2 waits, then give up as limit
        run("rate limit retries with backoff", [LIMIT] * 3, 3, [2.0, 4.0],
            "__ERR__ deepwiki ratelimited"),
        # limit then answer: 2 attempts, 1 wait, evidence returned
        run("rate limit then success", [LIMIT, ANSWER], 2, [2.0],
            "【DeepWiki:"),
        # a permanent error is not retried at all
        run("permanent error is not retried", [PERMANENT] * 3, 1, [],
            "__ERR__ deepwiki error"),
        # clean answer: one attempt, no sleep
        run("success needs no retry", [ANSWER], 1, [], "【DeepWiki:"),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)
