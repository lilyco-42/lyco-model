import subprocess, json, time, os, re, sys

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
LLAMA_CLI = r"D:\APP\scoop\shims\llama-cli.exe"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"
SYS_FILE = r"D:\gal\AliceInCradle\lyco-model\ooda_sys.txt"

AGENT_MODEL = os.path.join(MODEL_DIR, "grpo-Q4_K_M.gguf")
CHAT_MODEL = os.path.join(MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")

SYS_PROMPT = open(SYS_FILE, encoding="utf-8").read().strip()

# 20 representative rounds per model (subset of the 100-round suite)
AGENT_OODA = [
    "帮我写一个 Python 脚本，计算 1 到 100 的和",
    "帮我写一个 bash 脚本，批量把 .txt 重命名为 .md",
    "解释什么是 RAG，并给一个最小可用示例思路",
    "帮我写一个 Dockerfile，把 Python 应用容器化",
    "帮我写一个 SQL 查询，查找所有用户",
    "解释什么是 REST API",
    "帮我写一个正则表达式，匹配邮箱地址",
    "什么是 Docker？如何使用？",
    "帮我写一个简单的 HTTP 服务器",
    "解释什么是模型量化",
    "帮我写一个 LoRA 训练脚本",
    "什么是 CI/CD？",
    "帮我写一个 Nginx 配置",
    "解释什么是微服务架构",
    "帮我写一个 Python 爬虫",
    "什么是提示词注入？",
    "帮我写一个安全防护脚本",
    "解释什么是模型评估",
    "帮我写一个评估脚本",
    "什么是边缘计算？",
]

CHAT_OODA = [
    "什么是 Transformer？用通俗的话解释",
    "什么是模型量化？Q4_K_M 是什么意思？",
    "什么是端侧模型？有什么优缺点？",
    "什么是 RAG？",
    "什么是 Agent？",
    "什么是注意力机制？",
    "什么是微调？",
    "什么是提示工程？",
    "什么是向量数据库？",
    "什么是 Embedding？",
    "什么是过拟合？",
    "什么是正则化？",
    "什么是交叉验证？",
    "什么是 F1 分数？",
    "什么是贝叶斯定理？",
    "什么是 GAN？",
    "什么是扩散模型？",
    "什么是 BERT？",
    "什么是 Qwen3-0.6B？",
    "什么是 llama.cpp？",
]

OODA_MARKS = ["Observe", "Orient", "Decide", "Act", "Re-observe",
              "侦察", "研判", "决策", "进攻", "复盘"]


def run_once(model, prompt, n_predict=512):
    t0 = time.time()
    try:
        p = subprocess.run(
            [LLAMA_CLI, "-m", model, "-sys", SYS_PROMPT,
             "-p", prompt, "-n", str(n_predict),
             "--single-turn", "--simple-io", "--no-display-prompt"],
            capture_output=True, text=True, timeout=240)
        dt = time.time() - t0
        out = (p.stdout or "") + (p.stderr or "")
        m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
        body = out.split(prompt)[-1] if prompt in out else out
        body = re.sub(r"\[ ?Prompt:.*", "", body).strip()
        marks = [k for k in OODA_MARKS if k in body]
        return {"ok": p.returncode == 0, "output": body[-4000:],
                "gen_ts": float(m.group(1)) if m else None,
                "elapsed": round(dt, 2), "rc": p.returncode,
                "ooda_marks": marks,
                "ooda_hit": sum(1 for k in ("Observe", "Orient", "Decide",
                                            "Act", "Re-observe") if k in body)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "TIMEOUT", "gen_ts": None,
                "elapsed": 240.0, "rc": -1, "ooda_marks": [], "ooda_hit": 0}
    except Exception as e:
        return {"ok": False, "output": str(e)[:500], "gen_ts": None,
                "elapsed": round(time.time() - t0, 2), "rc": -2,
                "ooda_marks": [], "ooda_hit": 0}


def run_suite(tag, model, prompts):
    path = os.path.join(OUTPUT_DIR, f"{tag}_ooda.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["round"])
                except Exception:
                    pass
    print(f"[{tag}] resume from {len(done)}/{len(prompts)}", flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, pr in enumerate(prompts, 1):
            if i in done:
                continue
            r = run_once(model, pr)
            f.write(json.dumps({"round": i, "prompt": pr, **r},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{tag}] {i}/{len(prompts)} ok={r['ok']} "
                  f"ooda={r['ooda_hit']}/5 el={r['elapsed']}s", flush=True)
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("agent", "both"):
        run_suite("agent", AGENT_MODEL, AGENT_OODA)
    if which in ("chat", "both"):
        run_suite("chat", CHAT_MODEL, CHAT_OODA)
