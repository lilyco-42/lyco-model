"""lyco rag_loop: router -> retrieve -> summarize -> verify.

Retrieval sources (no self-build, all adopted):
  1. Wikipedia zh API (encyclopedia facts, on-demand)
  2. DeepWiki official MCP https://mcp.deepwiki.com/mcp (repo knowledge)
Summarizer: local chat model via llama-cli (never tested on memory).
Loop cap: max 2 retrieval rounds (lyco OODA rule).
"""
import subprocess, json, time, os, re, sys
import httpx

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
LLAMA_CLI = r"D:\APP\scoop\shims\llama-cli.exe"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"
CHAT_MODEL = os.path.join(MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")
DEEPWIKI_MCP = "https://mcp.deepwiki.com/mcp"

# Router: knowledge questions trigger search; chit-chat goes direct.
SEARCH_TRIGGERS = ["什么是", "什么叫", "为何", "为什么", "如何", "怎么",
                   "介绍", "含义", "意思", "关系", "区别", "是谁", "有哪些",
                   "最新", "多少", "何时"]
# repo-question -> DeepWiki repo mapping (extensible)
REPO_MAP = [("llama", "ggml-org/llama.cpp"),
            ("gguf", "ggml-org/llama.cpp"),
            ("量化", "ggml-org/llama.cpp"),
            ("qwen", "QwenLM/Qwen3"),
            ("transformer", "huggingface/transformers")]

STOPWORDS = set("什么是为什么如何怎么的了着是在与和或一个以及"
                "什么是吗呢吧啊呀")


def need_search(question):
    return any(t in question for t in SEARCH_TRIGGERS)


def pick_repo(question):
    q = question.lower()
    for key, repo in REPO_MAP:
        if key.lower() in q:
            return repo
    return None


def http():
    # Use system proxy from env (do NOT strip it); sanitize NO_PROXY
    # (a bare ::1 entry crashes httpx's URL parser).
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ.pop("no_proxy", None)
    return httpx.Client(timeout=30)


UA = {"User-Agent": "lyco-rag/0.1 (local RAG demo; contact: lyco42)"}


def wiki_search(client, question):
    """Wikipedia zh: opensearch -> summary. Returns evidence str or ''."""
    try:
        r = client.get("https://zh.wikipedia.org/w/api.php",
                       params={"action": "opensearch", "search": question,
                               "limit": 3, "format": "json"},
                       headers=UA)
        if r.status_code != 200:
            return f"__ERR__ wiki: http={r.status_code}"
        titles = r.json()[1]
        titles = r.json()[1]
        if not titles:
            return ""
        title = titles[0]
        s = client.get(
            f"https://zh.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers=UA)
        if s.status_code != 200:
            return f"__ERR__ wiki: summary http={s.status_code}"
        data = s.json()
        extract = data.get("extract", "")[:600]
        url = (data.get("content_urls", {})
               .get("desktop", {}).get("page", ""))
        if not extract:
            return ""
        return f"【维基百科:{title}】{extract} 来源:{url}"
    except Exception as e:
        return f"__ERR__ wiki: {str(e)[:120]}"


def mcp_rpc(client, session, payload):
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
    r = client.post(DEEPWIKI_MCP, headers=headers, json=payload)
    sess = r.headers.get("Mcp-Session-Id", session)
    out = []
    for line in r.text.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:].strip()))
            except Exception:
                pass
    return sess, out


def deepwiki_ask(client, repo, question):
    """Official DeepWiki MCP: initialize -> tools/list -> ask_question."""
    try:
        sess, msgs = mcp_rpc(client, None, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "lyco-rag", "version": "0.1"}}})
        sess, msgs2 = mcp_rpc(client, sess, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "ask_wiki_question",
                       "arguments": {"repoName": repo,
                                     "question": question}}})
        for m in msgs2:
            res = (m.get("result") or {})
            contents = res.get("content") or []
            texts = [c.get("text", "") for c in contents if c.get("type")
                     in ("text",)]
            if texts:
                return f"【DeepWiki:{repo}】{''.join(texts)[:600]}"
        err = next((m.get("error") for m in msgs2 if m.get("error")), None)
        return f"__ERR__ deepwiki: {str(err)[:120]}" if err else ""
    except Exception as e:
        return f"__ERR__ deepwiki: {str(e)[:120]}"


def retrieve(question, max_rounds=2):
    """OODA Observe/Orient: gather evidence, up to max_rounds."""
    client = http()
    evidence = []
    notes = []
    repo = pick_repo(question)
    for rnd in range(1, max_rounds + 1):
        w = wiki_search(client, question)
        if w and not w.startswith("__ERR__"):
            evidence.append(w)
            notes.append(f"round{rnd}:wiki-hit")
        elif w:
            notes.append(f"round{rnd}:{w}")
        else:
            notes.append(f"round{rnd}:wiki-empty")
        if repo and rnd == 1:
            d = deepwiki_ask(client, repo, question)
            if d and not d.startswith("__ERR__"):
                evidence.append(d)
                notes.append(f"round{rnd}:deepwiki-hit:{repo}")
            elif d:
                notes.append(f"round{rnd}:{d}")
        if evidence or rnd >= max_rounds:
            break
    client.close()
    return evidence, notes


def dec(b):
    if not b:
        return ""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")


def summarize(question, evidence, searched):
    """OODA Act: chat model only summarizes retrieved evidence."""
    if evidence:
        prompt = ("【检索到的资料】\n" + "\n".join(evidence) +
                  f"\n\n【用户问题】\n{question}\n\n"
                  "要求：只根据上述资料回答，资料没有的内容就说不知道。"
                  "用2-4句话总结要点。")
    elif searched:
        prompt = (f"【用户问题】\n{question}\n\n"
                  "要求：没有检索到相关资料，请直接说不知道，并说明可以去哪里查。")
    else:
        prompt = question
    t0 = time.time()
    p = subprocess.run(
        [LLAMA_CLI, "-m", CHAT_MODEL, "-p", prompt, "-n", "256",
         "--single-turn", "--simple-io", "--no-display-prompt"],
        capture_output=True, timeout=180)
    dt = time.time() - t0
    out = dec(p.stdout) + dec(p.stderr)
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    # answer = text after the LAST occurrence of the question
    # (echo may truncate long prompts, but the short question survives)
    if question in out:
        body = out.split(question)[-1]
    else:
        body = out
    body = re.sub(r"\[ ?Prompt:.*", "", body)
    body = body.split("Exiting...")[0].strip()
    return {"response": body[-3000:],
            "gen_ts": float(m.group(1)) if m else None,
            "elapsed": round(dt, 2), "rc": p.returncode}


def verify(question, evidence, response, searched=True):
    """OODA Re-observe: keyword overlap between evidence and response."""
    if not searched:
        return {"pass": True, "reason": "direct"}
    if not evidence:
        return {"pass": "不知道" in response, "reason": "no-evidence"}
    words = set()
    for ev in evidence:
        for w in re.findall(r"[\u4e00-\u9fffA-Za-z]{2,12}", ev):
            if w not in STOPWORDS:
                words.add(w)
    hits = [w for w in words if w in response]
    return {"pass": len(hits) >= 2, "reason": f"overlap={len(hits)}",
            "hits": hits[:8]}


def answer(question):
    """One closed OODA loop per dialogue turn."""
    t0 = time.time()
    routed = need_search(question)
    if not routed:
        ev, notes = [], ["router:direct"]
    else:
        ev, notes = retrieve(question)
    s = summarize(question, ev, searched=routed)
    v = verify(question, ev, s["response"], searched=routed)
    return {"question": question, "routed_search": routed,
            "evidence": ev, "retrieve_notes": notes,
            "verify": v, "total_elapsed": round(time.time() - t0, 2), **s}


if __name__ == "__main__":
    demo = [
        "什么是 GGUF？它和 llama.cpp 是什么关系？",
        "什么是模型量化？Q4_K_M 是什么意思？",
        "llama.cpp 里 Q4_K_M 的精度损失是多少？",
        "给我讲一个笑话",
    ]
    results = []
    for i, q in enumerate(demo, 1):
        r = answer(q)
        results.append(r)
        print(f"[{i}/{len(demo)}] search={r['routed_search']} "
              f"ev={len(r['evidence'])} verify={r['verify']} "
              f"el={r['total_elapsed']}s :: {q[:30]}", flush=True)
    with open(os.path.join(OUTPUT_DIR, "rag_loop_demo.json"),
              "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("saved rag_loop_demo.json")
