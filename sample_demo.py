import subprocess, json, time, os, re

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
LLAMA_CLI = r"D:\APP\scoop\shims\llama-cli.exe"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"

AGENT_MODEL = os.path.join(MODEL_DIR, "grpo-Q4_K_M.gguf")
CHAT_MODEL = os.path.join(MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")

AGENT_DEMO = [
    "你好，请介绍一下你自己",
    "帮我写一个 Python 脚本，计算 1 到 100 的和",
    "帮我写一个 bash 脚本，批量把 .txt 重命名为 .md",
    "解释什么是 RAG，并给一个最小可用示例思路",
    "帮我写一个 Dockerfile，把 Python 应用容器化",
]

CHAT_DEMO = [
    "你好，请介绍一下你自己",
    "什么是 Transformer？用通俗的话解释",
    "什么是 GGUF？它和 llama.cpp 是什么关系？",
    "什么是模型量化？Q4_K_M 是什么意思？",
    "什么是端侧模型？有什么优缺点？",
]


def run_once(model, prompt, n_predict=256):
    t0 = time.time()
    p = subprocess.run(
        [LLAMA_CLI, "-m", model, "-p", prompt, "-n", str(n_predict),
         "--single-turn", "--simple-io", "--no-display-prompt"],
        capture_output=True, text=True, timeout=180)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    body = out.split("> " + prompt)[-1] if "> " + prompt in out else out
    body = re.sub(r"\[ ?Prompt:.*", "", body).strip()
    return {"prompt": prompt, "response": body[-3000:],
            "gen_ts": float(m.group(1)) if m else None,
            "elapsed": round(dt, 2), "rc": p.returncode}


res = {"agent": [], "chat": []}
for tag, model, prompts in (("agent", AGENT_MODEL, AGENT_DEMO),
                            ("chat", CHAT_MODEL, CHAT_DEMO)):
    for i, pr in enumerate(prompts, 1):
        r = run_once(model, pr)
        res[tag].append(r)
        print(f"[{tag}] {i}/{len(prompts)} el={r['elapsed']}s", flush=True)

with open(os.path.join(OUTPUT_DIR, "demo_samples.json"),
          "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("saved demo_samples.json")
