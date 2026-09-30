"""llama-server concurrency and memory benchmark, on CPU, for the two GGUFs.

Answers the three deployment questions the report left open:
  - does raising -np cost single-request latency?
  - what does real concurrency (2 requests in flight) buy?
  - what does it cost to keep BOTH models resident at once?

Notes on method: throughput uses the server's own completion_tokens (a small
0.6B answers in ~15 tokens, so assuming max_tokens would inflate it ~8x), and
sequential rounds rotate prompts so the prompt cache cannot flatter them.

Run: python bench_server.py   (writes results/server_bench.json)
"""
import json, io, os, subprocess, time, threading
import httpx

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"
SERVER = r"D:\APP\scoop\shims\llama-server.exe"
M6 = os.path.join(MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")
M17 = os.path.join(MODEL_DIR, "Qwen3-1.7B-Q4_K_M.gguf")
MAX_TOKENS = 128
LONG_PROMPT = ("资料：" + "GGUF 是 llama.cpp 使用的单文件二进制格式，包含四段。" * 300
               + "问题：GGUF 有几段？")          # ~5.1k tokens, the probe case
PROMPTS = [
    "资料：GGUF 是 llama.cpp 使用的单文件二进制格式，包含 Header、KV Metadata、"
    "Tensor Info 与 Tensor Data 四段。问题：GGUF 有几段？",
    "资料：Q4_K_M 是 4 bit 的 K-quant 超级块量化档，兼顾体积与精度。"
    "问题：Q4_K_M 是什么？",
    "资料：llama-server 提供 OpenAI 兼容的 /v1/chat/completions 接口。"
    "问题：怎么用 HTTP 调用 llama.cpp？",
    "资料：量化会减小权重体积，通常会增加困惑度。问题：量化有什么代价？",
]

CASES = [
    {"label": "0.6B ctx=8192 np=1", "specs": [(M6, 8079, 1, 8192)]},
    {"label": "0.6B ctx=8192 np=4", "specs": [(M6, 8079, 4, 8192)]},
    {"label": "1.7B ctx=8192 np=1", "specs": [(M17, 8079, 1, 8192)]},
    {"label": "1.7B ctx=8192 np=4", "specs": [(M17, 8079, 4, 8192)]},
    {"label": "0.6B ctx=16384 np=4 (4096 per slot)", "specs": [(M6, 8079, 4, 16384)]},
    {"label": "0.6B ctx=16384 np=1", "specs": [(M6, 8079, 1, 16384)]},
    {"label": "both resident (0.6B + 1.7B, ctx=8192 np=1)",
     "specs": [(M6, 8079, 1, 8192), (M17, 8080, 1, 8192)]},
]


def start(model, port, np_, ctx=8192):
    log = open(os.path.join(OUTPUT_DIR, f"bench_{port}.log"), "ab")
    p = subprocess.Popen([SERVER, "-m", model, "--host", "127.0.0.1",
                          "--port", str(port), "-c", str(ctx), "-np", str(np_),
                          "--temp", "0"], stdout=log, stderr=log)
    deadline = time.time() + 300
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"server on {port} exited rc={p.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
                return p
        except Exception:
            pass
        time.sleep(2)
    p.terminate()
    raise RuntimeError(f"server on {port} never became healthy")


def rss_mb(pid):
    """Working set of the whole process tree, in MB.

    scoop's shim is only a launcher: the shim itself sits at ~9 MB and the
    llama.cpp worker is a child process, so measuring the spawned PID alone
    reports nothing useful. tasklist's columns are localized on this machine,
    so ask PowerShell for numbers."""
    script = (
        f"$t = 0; $ids = @({pid});"
        f"$ids += Get-CimInstance Win32_Process "
        f"-Filter \"ParentProcessId={pid}\" | ForEach-Object ProcessId;"
        "foreach ($i in $ids) { try { $t += (Get-Process -Id $i).WorkingSet64 }"
        " catch {} };"
        "$ids = $ids -join ',';"
        "[pscustomobject]@{MB=[math]::Round($t/1MB,1); Pids=$ids} | "
        "ConvertTo-Json -Compress"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True).stdout.strip()
    try:
        d = json.loads(out)
        return float(d["MB"]), d["Pids"]
    except Exception:
        return None, out[:80]


def one(port, prompt=None):
    prompt = prompt or PROMPTS[0]
    t0 = time.time()
    r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", timeout=300,
                   json={"messages": [{"role": "user", "content": prompt}],
                         "max_tokens": MAX_TOKENS, "temperature": 0.0,
                         "chat_template_kwargs": {"enable_thinking": False}})
    d = r.json()
    dt = time.time() - t0
    use = d.get("usage") or {}
    toks = use.get("completion_tokens") or 0
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return dt, toks, use.get("prompt_tokens") or 0, text


def parallel(port, n, offset=0):
    """Fire n requests at once (distinct prompts), return wall/latency/tokens."""
    res = [None] * n

    def work(i):
        res[i] = one(port, PROMPTS[(offset + i) % len(PROMPTS)])
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    return wall, [round(r[0], 2) for r in res], sum(r[1] for r in res)


def one_raw(port, prompt, max_tokens=MAX_TOKENS):
    r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", timeout=300,
                   json={"messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0.0,
                         "chat_template_kwargs": {"enable_thinking": False}})
    return r.status_code, r.json()


def measure(case):
    procs = [start(*s) for s in case["specs"]]
    ports = [s[1] for s in case["specs"]]
    rows = []
    try:
        time.sleep(2)
        idle_mem = [rss_mb(p.pid) for p in procs]
        for port in ports:
            one(port, PROMPTS[0])                       # load + warm
        for port in ports:
            seq = [one(port, PROMPTS[i % len(PROMPTS)]) for i in range(3)]
            wall = sum(s[0] for s in seq)
            toks = sum(s[1] for s in seq)
            rows.append({"port": port, "mode": "sequential x3 (rotating prompts)",
                         "mean_s": round(wall / 3, 2),
                         "min_s": round(min(s[0] for s in seq), 2),
                         "max_s": round(max(s[0] for s in seq), 2),
                         "mean_tokens": round(toks / 3, 1),
                         "tok_s": round(toks / wall, 1)})
        if len(ports) == 1:
            wall, each, toks = parallel(ports[0], 2, offset=1)
            rows.append({"port": ports[0], "mode": "2 concurrent",
                         "wall_s": round(wall, 2), "per_request_s": each,
                         "tokens": toks, "agg_tok_s": round(toks / wall, 1)})
        else:
            res = {}

            def go(p, i):
                res[p] = one(p, PROMPTS[i])
            ts = [threading.Thread(target=go, args=(p, i))
                  for i, p in enumerate(ports)]
            t0 = time.time()
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            rows.append({"port": ports, "mode": "both models, 1 request each",
                         "wall_s": round(time.time() - t0, 2),
                         "per_request_s": [round(res[p][0], 2) for p in ports]})
        # Last: a 5.4k-token request changes the KV/prompt-cache state, so it
        # must not sit before the latency rounds.
        code, big = one_raw(ports[0], LONG_PROMPT, max_tokens=40)
        ch = (big.get("choices") or [{}])[0]
        rows.append({"port": ports[0], "mode": "long prompt (about 5.4k tokens)",
                     "http": code,
                     "prompt_tokens": (big.get("usage") or {}).get("prompt_tokens"),
                     "finish_reason": ch.get("finish_reason"),
                     "reply_chars": len(((ch.get("message") or {}).get("content")) or ""),
                     "server_error": (big.get("error") or {}).get("message", "")[:160]})
        mem = [rss_mb(p.pid) for p in procs]
        rows.append({"mode": "memory, idle after load",
                     "per_instance_mb": [m[0] for m in idle_mem]})
        rows.append({"mode": "memory after the request rounds",
                     "per_instance_mb": [m[0] for m in mem],
                     "pids_measured": [m[1] for m in mem],
                     "total_mb": round(sum(m[0] or 0 for m in mem), 1)})
    finally:
        for p in procs:
            p.terminate()
        time.sleep(3)
    return rows


if __name__ == "__main__":
    out = {"max_tokens_cap": MAX_TOKENS,
           "note": "CPU-only llama.cpp build (b11045), -c 8192, temp 0, "
                   "throughput from server completion_tokens",
           "cases": []}
    dst = os.path.join(OUTPUT_DIR, "server_bench.json")
    for case in CASES:
        print(f"### {case['label']}", flush=True)
        rows = measure(case)
        out["cases"].append({"label": case["label"], "rows": rows})
        for r in rows:
            print("   ", json.dumps(r, ensure_ascii=False), flush=True)
        with io.open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", dst)
