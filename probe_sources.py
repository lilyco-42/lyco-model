"""Source reachability probe: what DeepWiki and Wikipedia zh actually return
from this machine right now. Output is committed so the report's numbers in
section 12 are checkable without re-running.

Usage: python probe_sources.py
"""
import json, io, os, time
import rag_loop as R

OUT = os.path.join(R.OUTPUT_DIR, "source_probe.json")
REPO = "ggml-org/llama.cpp"
Q = "量化有什么缺点？"


def deepwiki_rounds(n=3, pause=2.0):
    rows = []
    client = R.http()
    sess, _ = R.mcp_rpc(client, None, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "lyco-probe", "version": "0.1"}}})
    for i in range(n):
        t0 = time.time()
        try:
            s2, msgs = R.mcp_rpc(client, sess, {
                "jsonrpc": "2.0", "id": 2 + i, "method": "tools/call",
                "params": {"name": "ask_wiki_question",
                           "arguments": {"repoName": REPO, "question": Q}}})
            body = ""
            for m in msgs:
                c = (m.get("result") or {}).get("content") or []
                body += "".join(x.get("text", "") for x in c
                                if x.get("type") == "text")
            rows.append({"try": i + 1, "elapsed": round(time.time() - t0, 2),
                         "chars": len(body),
                         "is_error_body": R.looks_like_error(body),
                         "head": body[:120]})
        except Exception as e:
            rows.append({"try": i + 1, "elapsed": round(time.time() - t0, 2),
                         "exception": str(e)[:160]})
        time.sleep(pause)
    client.close()
    return rows


def deepwiki_retry_live(tries=3):
    """Run the production retry path once against the live endpoint so the
    recorded note string is real, not just the offline test's."""
    client = R.http()
    state = {"calls": 0}
    sleeps = []

    def call():
        state["calls"] += 1
        return R.deepwiki_raw(client, REPO, Q)

    def sleeper(secs):
        sleeps.append(secs)
        time.sleep(secs)          # real wait: the wall clock figure must be true

    t0 = time.time()
    out = R.deepwiki_ask(client, REPO, Q, tries=tries,
                         sleeper=sleeper, call=call)
    client.close()
    return {"attempts": state["calls"], "slept_s": sleeps,
            "wall_s": round(time.time() - t0, 2),
            "returned": out[:200], "cached_as_evidence": bool(out) and not out.startswith("__ERR__")}


def wiki_variants():
    """Different exits for zh Wikipedia from this egress."""
    cands = [
        ("zh opensearch (project UA)",
         "GET", "https://zh.wikipedia.org/w/api.php",
         {"action": "opensearch", "search": "GGUF", "limit": 3, "format": "json"},
         R.UA),
        ("zh opensearch (browser UA)",
         "GET", "https://zh.wikipedia.org/w/api.php",
         {"action": "opensearch", "search": "GGUF", "limit": 3, "format": "json"},
         {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}),
        ("zh opensearch (no UA)",
         "GET", "https://zh.wikipedia.org/w/api.php",
         {"action": "opensearch", "search": "GGUF", "limit": 3, "format": "json"},
         {"User-Agent": ""}),
        ("zh REST summary",
         "GET", "https://zh.wikipedia.org/api/rest_v1/page/summary/语言模型",
         None, R.UA),
        ("zh action=query extracts",
         "GET", "https://zh.wikipedia.org/w/api.php",
         {"action": "query", "prop": "extracts", "explaintext": 1,
          "exintro": 1, "titles": "量化 (机器学习)", "format": "json"},
         R.UA),
        ("en REST summary (fallback)",
         "GET", "https://en.wikipedia.org/api/rest_v1/page/summary/GPU",
         None, R.UA),
        ("zh no-proxy direct",
         "GET", "https://zh.wikipedia.org/w/api.php",
         {"action": "opensearch", "search": "GGUF", "limit": 3, "format": "json"},
         R.UA),
    ]
    out = []
    for name, method, url, params, ua in cands:
        use_proxy = "no-proxy" not in name
        try:
            tr = httpx_transport(use_proxy)
            r = tr.get(url, params=params, headers=({k: v for k, v in ua.items() if v}
                                                    or None), timeout=20)
            text = r.text
            out.append({"variant": name, "proxy": use_proxy, "status": r.status_code,
                        "chars": len(text),
                        "ok_json": _is_json(text),
                        "head": text[:100].replace("\n", " ")})
        except Exception as e:
            out.append({"variant": name, "proxy": use_proxy,
                        "exception": str(e)[:160]})
    return out


def _is_json(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def httpx_transport(use_proxy):
    if use_proxy:
        return R.http()
    os.environ["NO_PROXY"] = "*"
    c = R.httpx.Client(timeout=20, proxy=None, trust_env=False)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    return c


if __name__ == "__main__":
    data = {"deepwiki": deepwiki_rounds(),
            "deepwiki_retry_live": deepwiki_retry_live(),
            "wikipedia": wiki_variants()}
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(json.dumps(data, ensure_ascii=False, indent=1)[:2500])
    print("saved", OUT)
