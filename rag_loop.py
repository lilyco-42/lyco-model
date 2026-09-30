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
SEARCH_TRIGGERS = ["什么是", "是什么", "什么叫", "是啥", "啥是", "何为",
                   "为何", "为什么", "如何", "怎么",
                   "介绍", "含义", "意思", "关系", "区别", "是谁", "有哪些",
                   "最新", "多少", "何时"]

RETRIEVE_STOP = {"什么", "么是", "什么是", "是什么", "怎么", "为什么", "为什",
                 "如何", "哪些", "多少", "介绍", "含义", "意思", "关系",
                 "怎样", "何为", "是啥", "啥是", "什么用", "有什么", "干什么",
                 "用处", "作用", "这是什么", "那是什么", "它和", "和", "的"}
# repo-question -> DeepWiki repo mapping (extensible)
REPO_MAP = [("llama", "ggml-org/llama.cpp"),
            ("gguf", "ggml-org/llama.cpp"),
            ("量化", "ggml-org/llama.cpp"),
            ("qwen", "QwenLM/Qwen3"),
            ("transformer", "huggingface/transformers")]

STOPWORDS = set("什么是为什么如何怎么的了着是在与和或一个以及"
                "什么是吗呢吧啊呀")

KB_DIR = r"D:\gal\AliceInCradle\kb"
RSS_URL = "https://rustcc.cn/rss"
RSS_CACHE = os.path.join(OUTPUT_DIR, "rustcc_rss.xml")
RSS_TTL = 3600


def tokens(question):
    toks = set()
    for w in re.findall(r"[A-Za-z0-9_+\-#]{2,}", question):
        toks.add(w.lower())
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        toks.add(run)
        for i in range(len(run) - 1):
            toks.add(run[i:i + 2])
    return toks


def score_text(toks, text):
    tl = text.lower()
    return sum(1 for t in toks
               if t not in RETRIEVE_STOP and (t in tl or t in text))


def is_anchor(t):
    if re.fullmatch(r"[A-Za-z0-9_+\-#]{2,}", t):
        return True
    return len(t) >= 3 and t not in RETRIEVE_STOP


def has_anchor(toks, text):
    tl = text.lower()
    return any(is_anchor(t) and (t in tl or t in text) for t in toks)


def local_search(question, topn=2):
    """Local kb (gh-cloned repos): keyword score over md/toml/txt."""
    toks = tokens(question)
    hits = []
    if not os.path.isdir(KB_DIR):
        return []
    for root, dirs, files in os.walk(KB_DIR):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "target", "node_modules",
                                ".obsidian")]
        for fn in files:
            if not fn.lower().endswith((".md", ".markdown", ".txt",
                                        ".toml")):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception:
                continue
            s = score_text(toks, text)
            if s >= 2 and has_anchor(toks, text):
                idx = max([text.find(t) for t in toks if t in text],
                          default=0)
                start = max(0, idx - 100)
                rel = os.path.relpath(p, KB_DIR)
                hits.append((s, f"【本地库:{rel}】{text[start:start + 500]}"))
    hits.sort(key=lambda x: -x[0])
    return [h[1] for h in hits[:topn]]


def rss_search(client, question, topn=2):
    """RustCC RSS: fetch (1h cache) -> keyword match title+desc."""
    import xml.etree.ElementTree as ET
    toks = tokens(question)
    try:
        if (os.path.exists(RSS_CACHE) and
                time.time() - os.path.getmtime(RSS_CACHE) < RSS_TTL):
            with open(RSS_CACHE, encoding="utf-8",
                      errors="replace") as f:
                xml_text = f.read()
        else:
            r = client.get(RSS_URL, headers=UA)
            r.raise_for_status()
            xml_text = r.text
            with open(RSS_CACHE, "w", encoding="utf-8") as f:
                f.write(xml_text)
        root = ET.fromstring(xml_text)
    except Exception as e:
        return [], [f"rss-err: {str(e)[:100]}"]
    scored = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        desc = re.sub(r"<[^>]+>", "",
                      item.findtext("description") or "")[:300]
        s = score_text(toks, title) * 3 + score_text(toks, desc)
        if s >= 2 and (has_anchor(toks, title) or has_anchor(toks, desc)):
            scored.append((s, f"【RustCC:{title}】{desc} 来源:{link}"))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:topn]], []


def source_order(question):
    if any(k in question for k in ("mpkg", "lilyco", "lyco", "记忆包")):
        return ["local", "deepwiki", "rss", "wiki"]
    if re.search(r"[Rr]ust|cargo|日报", question):
        return ["rss", "local", "deepwiki", "wiki"]
    return ["local", "deepwiki", "rss", "wiki"]


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


def retrieve(question, max_rounds=2, max_chars=1200):
    """OODA Observe/Orient: gather up to max_rounds sources (no early
    stop: a weak local hit must not block a better DeepWiki hit)."""
    client = http()
    evidence, notes = [], []
    repo = pick_repo(question)
    tried = 0
    for src in source_order(question):
        if tried >= max_rounds:
            break
        if src == "local":
            hits = local_search(question)
            notes.append(f"local:{len(hits)}")
            evidence += hits
            tried += 1
        elif src == "rss":
            hits, errs = rss_search(client, question)
            notes += errs if not hits else [f"rss:{len(hits)}"]
            evidence += hits
            tried += 1
        elif src == "deepwiki":
            if repo:
                d = deepwiki_ask(client, repo, question)
                if d and not d.startswith("__ERR__"):
                    evidence.append(d)
                    notes.append(f"deepwiki:{repo}")
                elif d:
                    notes.append(d[:80])
            else:
                notes.append("deepwiki:skip-no-repo")
            tried += 1
        elif src == "wiki":
            w = wiki_search(client, question)
            if w and not w.startswith("__ERR__"):
                evidence.append(w)
                notes.append("wiki:hit")
            elif w:
                notes.append(w[:80])
            else:
                notes.append("wiki:empty")
            tried += 1
    client.close()
    # cap total evidence for the 0.6B context window
    kept, total = [], 0
    for ev in evidence:
        if total >= max_chars:
            break
        kept.append(ev[:max_chars - total])
        total += len(kept[-1])
    return kept, notes


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
         "--single-turn", "--no-display-prompt"],
        capture_output=True, timeout=180)
    dt = time.time() - t0
    out = dec(p.stdout) + dec(p.stderr)
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    # long-prompt echo is truncated with a "...(truncated)" marker and the
    # answer follows it; short prompts echo verbatim as "> "+prompt
    if "(truncated)" in out:
        body = out.split("(truncated)")[-1]
    else:
        marker = "> " + prompt
        body = out.split(marker)[-1] if marker in out else out
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
        "mpkg 记忆包是什么？有什么用？",
        "纯 Rust 实现的 Luau 运行时是什么？",
        "什么是 GGUF？它和 llama.cpp 是什么关系？",
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
