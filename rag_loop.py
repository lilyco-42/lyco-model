"""lyco rag_loop: router -> retrieve (cached) -> summarize (with memory) -> verify.

Retrieval sources (no self-build, all adopted):
  1. local kb: gh-cloned own repos (KB_DIR)
  2. RustCC RSS (1h file cache)
  3. DeepWiki official MCP https://mcp.deepwiki.com/mcp (repo knowledge)
  4. Wikipedia zh API (encyclopedia facts, on-demand)

Runner: llama-server (OpenAI-compatible endpoint, greedy decoding), so a turn
carries real chat history. llama-cli --single-turn reloaded the model every
turn and had no context at all -- that was the memory gap.

Network evidence is cached on disk (EVIDENCE_TTL) because DeepWiki answers the
same question differently on every call; `--offline` replays the cache only.
"""
import subprocess, json, time, os, re, sys, hashlib
import httpx

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"
SERVER_BIN = r"D:\APP\scoop\shims\llama-server.exe"
SERVER_LOG = os.path.join(OUTPUT_DIR, "llama_server.log")
HOST, PORT = "127.0.0.1", 8079
BASE = f"http://{HOST}:{PORT}"
N_CTX = "8192"


def arg_value(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# --model so the same suite can be replayed against a different summarizer.
CHAT_MODEL = arg_value("--model") or os.path.join(
    MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")
OUT_TAG = arg_value("--tag", "rag_loop_memory")

CACHE_FILE = os.path.join(OUTPUT_DIR, "evidence_cache.json")
EVIDENCE_TTL = 86400          # 24h: DeepWiki/RSS text is stable enough
OFFLINE = "--offline" in sys.argv
UNSUP_TOL = 1                 # evidence-free latin tokens we tolerate in an answer

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

# Anaphora: a turn that refers back to the previous topic instead of naming it.
ANAPHORA = ["它", "他", "她", "这个", "那个", "这些", "那些", "其中", "还有",
            "然后", "继续", "刚才", "前面", "上面", "又", "这样", "那样"]

# Routing is narrower than rewriting: the discourse markers above may borrow a
# topic for search, but "然后给我讲个笑话" must stay chit-chat. Only something
# that actually points at the world (a referent, or a prediction/explanation
# ask) forces a retrieval round -- otherwise a factual question with no
# 什么/为什么 cue used to skip search and get rubber-stamped by verify.
REFERENTIAL = ["它", "他", "她", "它们", "其中", "这个", "那个", "这些", "那些",
               "这样", "那样", "刚才", "前面", "上面"]
PREDICTION_CUES = ["会不会", "能不能", "是不是", "有没有", "影响", "导致",
                   "需要", "支持", "适合", "缺点", "优点", "可靠", "安全", "准确"]

# Memory window: prior user/assistant turns kept in the prompt. The 0.6B model
# loses the thread long before the context window fills up, so cap by content.
HISTORY_TURNS = 4
HISTORY_CHARS = 900
TOPIC_CHARS = 160       # resolved-query memory kept for follow-up retrieval
EVIDENCE_POOL_CHARS = 4000  # conversation-wide claims verify may cite


def tokens(question):
    toks = set()
    for w in re.findall(r"[A-Za-z0-9_+\-#]{2,}", question):
        toks.add(w.lower())
    for run in re.findall(r"[一-鿿]{2,}", question):
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
        fresh = (os.path.exists(RSS_CACHE) and
                 time.time() - os.path.getmtime(RSS_CACHE) < RSS_TTL)
        if fresh or OFFLINE:
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

# A 200 response is not necessarily an answer: DeepWiki relays upstream
# failures inside `content`, and that junk used to be cached as evidence and
# fed to the model (第 8 节的"垃圾证据进 -> 幻觉出"从缓存层回来了).
ERR_MARKERS = ("Error processing question", "Too Many Requests", "Client error",
               "Server error", "Traceback", "for url ")


def looks_like_error(text):
    return any(m in text for m in ERR_MARKERS)


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
    """Official DeepWiki MCP: initialize -> tools/call -> ask_wiki_question."""
    try:
        sess, _ = mcp_rpc(client, None, {
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
                body = ''.join(texts)
                if looks_like_error(body):
                    return f"__ERR__ deepwiki: {body[:120]}"
                return f"【DeepWiki:{repo}】{body[:600]}"
        err = next((m.get("error") for m in msgs2 if m.get("error")), None)
        return f"__ERR__ deepwiki: {str(err)[:120]}" if err else ""
    except Exception as e:
        return f"__ERR__ deepwiki: {str(e)[:120]}"


# ---------------------------------------------------------------- cache layer

_cache = None


def cache():
    global _cache
    if _cache is None:
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def cache_get(key, allow_stale=False):
    e = cache().get(key)
    if not e:
        return None
    if not allow_stale and time.time() - e["t"] > EVIDENCE_TTL:
        return None
    return e["v"]


def cache_put(key, val):
    cache()[key] = {"t": time.time(), "v": val}
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache(), f, ensure_ascii=False, indent=0)
    os.replace(tmp, CACHE_FILE)


def cache_key(source, question, repo=""):
    raw = f"{source}|{repo}|{question}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def cached_lookup(source, question, fetch, repo=""):
    """Return (text, note). Network only on miss, unless --offline."""
    key = cache_key(source, question, repo)
    hit = cache_get(key, allow_stale=OFFLINE)
    if hit is not None:
        if looks_like_error(hit):
            cache().pop(key, None)
            return "", f"{source}:cache-dropped-error"
        return hit, f"{source}:cache"
    if OFFLINE:
        return "", f"{source}:cache-miss-offline"
    val = fetch()
    if not val:
        return "", f"{source}:empty"
    if val.startswith("__ERR__"):
        return "", val[:80]
    cache_put(key, val)
    return val, f"{source}:live"


# ------------------------------------------------------------------- retrieve

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
                txt, note = cached_lookup(
                    "deepwiki", question,
                    lambda: deepwiki_ask(client, repo, question), repo)
                if txt:
                    evidence.append(txt)
                notes.append(note)
            else:
                notes.append("deepwiki:skip-no-repo")
            tried += 1
        elif src == "wiki":
            txt, note = cached_lookup(
                "wiki", question, lambda: wiki_search(client, question))
            if txt:
                evidence.append(txt)
            notes.append(note)
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


# ---------------------------------------------------------------------- model

def server_up():
    try:
        return httpx.get(BASE + "/health", timeout=3).status_code == 200
    except Exception:
        return False


def server_model():
    try:
        d = httpx.get(BASE + "/v1/models", timeout=3).json()
        return (d.get("models") or [{}])[0].get("model", "")
    except Exception:
        return ""


def ensure_server(model=CHAT_MODEL, timeout=240):
    """Reuse a running llama-server, else start one. Returns the Popen we
    spawned (None when an existing server was reused)."""
    if server_up():
        loaded = server_model()
        if os.path.normcase(loaded) != os.path.normcase(model):
            raise RuntimeError(
                f"a llama-server is already on {BASE} serving {loaded!r}, "
                f"not {model!r}; stop it or change PORT")
        return None
    with open(SERVER_LOG, "ab") as log:
        p = subprocess.Popen(
            [SERVER_BIN, "-m", model, "--host", HOST, "--port", str(PORT),
             "-c", N_CTX, "-np", "1", "--temp", "0"],
            stdout=log, stderr=log)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={p.returncode} "
                               f"(see {SERVER_LOG})")
        if server_up():
            return p
        time.sleep(2)
    p.terminate()
    raise RuntimeError(f"llama-server not healthy after {timeout}s")


def chat(messages, n_predict=256):
    """One greedy completion. Greedy + cached evidence == replayable."""
    t0 = time.time()
    r = httpx.post(BASE + "/v1/chat/completions", timeout=300, json={
        "messages": messages,
        "max_tokens": n_predict,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    d = r.json()
    dt = time.time() - t0
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content")
    text = (text or "").strip()
    ct = (d.get("usage") or {}).get("completion_tokens") or 0
    return {"response": text,
            "gen_ts": round(ct / dt, 1) if dt and ct else None,
            "elapsed": round(dt, 2)}


def evidence_block(evidence, question):
    return ("【检索到的资料】\n" + "\n".join(evidence) +
            f"\n\n【用户问题】\n{question}")


def system_prompt(searched, has_evidence):
    if searched and has_evidence:
        return ("你是问答助手。用户消息里会附上【检索到的资料】，"
                "只根据这些资料回答，资料没有的内容就说不知道。")
    if searched:
        return ("你是问答助手。这次没有检索到相关资料，"
                "请直接说不知道，并说明可以去哪里查。")
    return "你是一个中文助手，简洁自然地回答。"


def summarize(question, evidence, searched, history=()):
    """OODA Act: chat model summarizes retrieved evidence, with the prior
    turns of this conversation in context. Evidence rides on the current turn
    only -- the stored history stays clean user/assistant text."""
    if evidence:
        user = (evidence_block(evidence, question) +
                "\n\n要求：只根据上述资料回答，资料没有的内容就说不知道。"
                "用 2-4 句话总结要点。")
    else:
        user = question
    msgs = [{"role": "system",
             "content": system_prompt(searched, bool(evidence))}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history]
    msgs.append({"role": "user", "content": user})
    return chat(msgs)


# --------------------------------------------------------------------- verify

def latin_terms(text):
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_+#.\-]{2,}", text))


def digit_claims(text):
    """Numbers that carry a precision claim (decimal, %, or 3+ digits)."""
    return set(n for n in re.findall(r"\d+(?:[.,]\d+)?%?", text)
               if re.search(r"[.,%]\d|\d{3}", n))


def relevance_terms(question):
    """Content terms an answer must touch to count as answering. is_anchor()
    deliberately drops 2-char Chinese strings for retrieval noise, but 量化 and
    缺点 are exactly what a correct answer repeats -- judging relevance with
    that rule marked correct answers off-topic."""
    return [t for t in tokens(question)
            if len(t) >= 2 and t not in RETRIEVE_STOP]


DECLINE = ["不知道", "不清楚", "不了解", "没有相关", "没有关于",
           "没有足够", "没有具体", "无法确定", "无法回答", "请提供", "提供更多",
           "尚未", "没听说过"]


def verify(question, evidence, response, searched=True, anchors=None,
           prior_evidence=()):
    """OODA Re-observe. Overlap alone only proves the model copied; it does
    not prove it copied *right*, so every concrete claim in the answer has to
    exist in the evidence too.

    The haystack is the whole conversation's evidence, not just this turn's:
    with memory a fact learned two turns ago is still supported, and claiming
    it again must not read as a hallucination."""
    if not searched:
        # A direct turn is not evidence-grounded, but it still must not hand
        # out made-up precision. Chit-chat passes; "延迟是 12.5ms" does not.
        pool = question + "\n" + "\n".join(prior_evidence)
        fab = sorted(n for n in digit_claims(response) if n not in pool)
        return {"pass": not fab,
                "reason": "direct" if not fab else "direct-unsourced-numbers",
                "invented_numbers": fab}
    ev_all = list(evidence) + list(prior_evidence)
    if not ev_all:
        # No evidence: the only acceptable answer is an explicit decline.
        # Match the family of declines, not the literal "不知道" -- a model
        # that says "我没有关于这一术语的具体信息" did the right thing.
        said = next((d for d in DECLINE if d in response), None)
        return {"pass": said is not None, "reason": "no-evidence",
                "decline": said or ""}
    hay = "\n".join(ev_all) + "\n" + question
    hay_l = hay.lower()

    words = set()
    for ev in ev_all:
        for w in re.findall(r"[一-鿿A-Za-z]{2,12}", ev):
            if w not in STOPWORDS:
                words.add(w)
    # sorted: set order is hash-randomized per process, and the record has to
    # replay identically (verdict was stable, the hits sample was not).
    hits = [w for w in sorted(words) if w in response]

    # a claim the evidence never made -> hallucination
    unsup = sorted(t for t in latin_terms(response) if t.lower() not in hay_l)
    fab_num = sorted(n for n in digit_claims(response) if n not in hay)

    anch = anchors if anchors is not None else relevance_terms(question)
    anch = [a for a in anch if len(a) >= 2]
    rel = any(a in response or a.lower() in response.lower() for a in anch)

    reasons = []
    if len(hits) < 2:
        reasons.append("low-overlap")
    if not rel:
        reasons.append("off-topic")
    if fab_num:
        reasons.append("invented-numbers")
    if len(unsup) > UNSUP_TOL:
        reasons.append("unsupported-terms")
    return {"pass": not reasons,
            "reason": ",".join(reasons) or f"overlap={len(hits)}",
            "overlap": len(hits), "hits": hits[:8],
            "invented_numbers": fab_num,
            "unsupported_terms": unsup[:8],
            "on_topic": rel}


def judge(question, evidence, response):
    """Second model pass: does the answer actually rest on the evidence?
    Advisory only -- a 0.6B grading a 0.6B is not a gate."""
    msgs = [{"role": "system", "content": "你只做一件事：判断答案是否用了给定资料的内容。只回答 是 或 否。"},
            {"role": "user",
             "content": f"资料：\n{(evidence[0] if evidence else '')[:400]}\n\n"
                        f"问题：{question}\n\n答案：{response}\n\n答案用到资料了吗？"}]
    try:
        return chat(msgs, n_predict=4)["response"][:8]
    except Exception as e:
        return f"judge-err:{str(e)[:60]}"


# --------------------------------------------------------------- conversation

class Conversation:
    """Multi-turn chat over the RAG loop. Each turn keeps its own evidence,
    but the user/assistant history is shared -- that is the memory."""

    def __init__(self):
        self.turns = []      # clean user/assistant pairs, no evidence blobs
        self.topic = ""      # last resolved query that went through retrieval
        self.evidence_pool = ""   # what this conversation is allowed to claim

    def window(self):
        sel, total = [], 0
        for m in reversed(self.turns):
            if len(sel) >= HISTORY_TURNS * 2:
                break
            if total + len(m["content"]) > HISTORY_CHARS:
                break
            sel.append(m)
            total += len(m["content"])
        return list(reversed(sel))

    def is_followup(self, q):
        return bool(self.topic) and any(a in q for a in ANAPHORA)

    def needs_evidence(self, q):
        """Route to search whenever the turn makes a factual ask -- not only
        when it happens to contain a 什么/为什么 phrase. Without this, "它会让模型
        跑得更快吗？" and "Q4_K_M 影响精度吗？" went direct and verify (which
        trusts direct turns) rubber-stamped the guess."""
        if need_search(q) or any(c in q for c in PREDICTION_CUES):
            return True
        if not self.topic:
            return False
        # A yes/no question asked in the middle of a knowledge conversation is
        # about that knowledge ("精度会掉吗？"), and it may carry no referent.
        return "吗" in q or any(r in q for r in REFERENTIAL)

    def retrieval_query(self, q):
        """An '它是什么' turn has no retrievable terms; borrow the previous
        topic so search still finds something. topic is the *resolved* query,
        so a chain of follow-ups keeps accumulating the original anchors
        instead of degrading into pronouns."""
        if self.is_followup(q) and self.topic:
            return f"{self.topic} {q}"
        return q

    def ask(self, q, run_judge=False):
        t0 = time.time()
        rq = self.retrieval_query(q)
        routed = self.needs_evidence(q) or (rq != q and need_search(rq))
        if routed:
            ev, notes = retrieve(rq)
        else:
            ev, notes = [], ["router:direct"]
        hist = self.window()
        s = summarize(q, ev, searched=routed, history=hist)
        prior = [self.evidence_pool] if self.evidence_pool else ()
        v = verify(rq, ev, s["response"], searched=routed,
                   anchors=relevance_terms(q), prior_evidence=prior)
        rec = {"question": q, "retrieval_query": rq,
               "routed_search": routed, "evidence": ev,
               "retrieve_notes": notes, "memory_turns": len(hist),
               "prior_evidence_chars": len(self.evidence_pool),
               "verify": v,
               "total_elapsed": round(time.time() - t0, 2), **s}
        if run_judge:
            rec["judge"] = judge(q, ev, s["response"])
        self.turns.append({"role": "user", "content": q})
        self.turns.append({"role": "assistant", "content": s["response"]})
        if routed and ev:
            self.topic = rq[-TOPIC_CHARS:]
            self.evidence_pool = (
                "\n".join(ev) + "\n" + self.evidence_pool)[-EVIDENCE_POOL_CHARS:]
        return rec


INDEPENDENT = [
    "mpkg 记忆包是什么？有什么用？",
    "纯 Rust 实现的 Luau 运行时是什么？",
    "什么是 GGUF？它和 llama.cpp 是什么关系？",
    "llama.cpp 里 Q4_K_M 的精度损失是多少？",
    "给我讲一个笑话",
]

# One topic carried across turns: turns 2-5 name nothing, they only point.
# Turn 5 has no 什么/为什么 phrasing at all, and turn 6 must stay chit-chat
# even inside a knowledge conversation -- both are routing regressions.
MEMORY_SESSION = [
    "什么是 GGUF？",
    "它和 llama.cpp 是什么关系？",
    "那 Q4_K_M 又是什么？",
    "它会让模型跑得更快吗？",
    "量化有什么缺点？",
    "给我讲个笑话",
]


def run():
    proc = ensure_server()
    out = {"model": os.path.basename(CHAT_MODEL), "tag": OUT_TAG,
           "independent": [], "memory": [], "memory_control": []}
    try:
        print("=== A. independent turns (regression) ===", flush=True)
        for i, q in enumerate(INDEPENDENT, 1):
            r = Conversation().ask(q)
            out["independent"].append(r)
            print(f"[{i}/{len(INDEPENDENT)}] search={r['routed_search']} "
                  f"ev={len(r['evidence'])} notes={r['retrieve_notes']} "
                  f"verify={r['verify']['pass']}/{r['verify']['reason']} "
                  f"el={r['total_elapsed']}s :: {q[:26]}", flush=True)

        print("\n=== B. one conversation with memory ===", flush=True)
        conv = Conversation()
        for i, q in enumerate(MEMORY_SESSION, 1):
            r = conv.ask(q, run_judge=True)
            out["memory"].append(r)
            print(f"[{i}/{len(MEMORY_SESSION)}] mem={r['memory_turns']} "
                  f"followup={r['retrieval_query'] != q} "
                  f"ev={len(r['evidence'])} judge={r.get('judge')} "
                  f"verify={r['verify']['pass']}/{r['verify']['reason']} "
                  f":: {q[:24]}", flush=True)
            print(f"      -> {r['response'][:150]}", flush=True)

        print("\n=== C. same turns, memory disabled (control) ===", flush=True)
        for i, q in enumerate(MEMORY_SESSION, 1):
            r = Conversation().ask(q)
            out["memory_control"].append(r)
            print(f"[{i}/{len(MEMORY_SESSION)}] mem=0 "
                  f"ev={len(r['evidence'])} :: {q[:24]}", flush=True)
            print(f"      -> {r['response'][:150]}", flush=True)
    finally:
        if proc is not None:
            proc.terminate()

    dst = os.path.join(OUTPUT_DIR, f"{OUT_TAG}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    run()
