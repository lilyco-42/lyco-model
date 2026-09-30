import subprocess, json, time, os, re, sys

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
LLAMA_CLI = r"D:\APP\scoop\shims\llama-cli.exe"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGENT_MODEL = os.path.join(MODEL_DIR, "grpo-Q4_K_M.gguf")
CHAT_MODEL = os.path.join(MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")

AGENT_BASE = [
    "你好，请介绍一下你自己",
    "帮我列出当前目录下的文件",
    "创建一个名为 test.txt 的文件，内容是 Hello World",
    "查看系统信息",
    "帮我写一个 Python 脚本，计算 1 到 100 的和",
    "解释什么是机器学习",
    "帮我写一个 bash 脚本，批量重命名文件",
    "什么是 Docker？如何使用？",
    "帮我写一个简单的 HTTP 服务器",
    "解释什么是 REST API",
    "帮我写一个正则表达式，匹配邮箱地址",
    "什么是 Git？如何创建分支？",
    "帮我写一个 SQL 查询，查找所有用户",
    "解释什么是递归",
    "帮我写一个 JavaScript 函数，实现数组去重",
    "什么是 Kubernetes？",
    "帮我写一个 Dockerfile",
    "解释什么是微服务架构",
    "帮我写一个 Python 爬虫",
    "什么是区块链？",
]

CHAT_BASE = [
    "你好，请介绍一下你自己",
    "给我讲一个笑话",
    "帮我写一首关于秋天的诗",
    "什么是人工智能？",
    "什么是机器学习？",
    "解释一下深度学习",
    "什么是神经网络？",
    "什么是大语言模型？",
    "什么是 Transformer？",
    "什么是注意力机制？",
    "什么是微调？",
    "什么是 RAG？",
    "什么是 Agent？",
    "什么是 GGUF？",
    "什么是 llama.cpp？",
    "什么是模型量化？",
    "什么是端侧模型？",
    "什么是 Qwen3-0.6B？",
    "什么是过拟合？",
    "什么是正则化？",
]


def build_rounds(base, n=100):
    out = []
    for i in range(n):
        out.append(f"【第{i+1}轮】{base[i % len(base)]}")
    return out


def run_once(model, prompt, n_predict=64):
    t0 = time.time()
    try:
        p = subprocess.run(
            [LLAMA_CLI, "-m", model, "-p", prompt, "-n", str(n_predict),
             "--single-turn", "--simple-io", "--no-display-prompt"],
            capture_output=True, text=True, timeout=150)
        dt = time.time() - t0
        out = (p.stdout or "") + (p.stderr or "")
        m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
        return {"ok": p.returncode == 0, "output": out[-2000:],
                "gen_ts": float(m.group(1)) if m else None,
                "elapsed": round(dt, 2), "rc": p.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "TIMEOUT", "gen_ts": None,
                "elapsed": 150.0, "rc": -1}
    except Exception as e:
        return {"ok": False, "output": str(e)[:500], "gen_ts": None,
                "elapsed": round(time.time() - t0, 2), "rc": -2}


def run_suite(tag, model, prompts):
    path = os.path.join(OUTPUT_DIR, f"{tag}_100rounds.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["round"])
                except Exception:
                    pass
    print(f"[{tag}] resume from {len(done)}/100", flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, pr in enumerate(prompts, 1):
            if i in done:
                continue
            r = run_once(model, pr)
            f.write(json.dumps({"round": i, "prompt": pr, **r},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == 1:
                print(f"[{tag}] {i}/100 ok={r['ok']} "
                      f"ts={r['gen_ts']} el={r['elapsed']}s", flush=True)
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("agent", "both"):
        run_suite("agent", AGENT_MODEL, build_rounds(AGENT_BASE))
    if which in ("chat", "both"):
        run_suite("chat", CHAT_MODEL, build_rounds(CHAT_BASE))
