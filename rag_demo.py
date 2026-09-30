import subprocess, json, time, os, re

MODEL_DIR = r"D:\gal\AliceInCradle\lyco-model\models"
LLAMA_CLI = r"D:\APP\scoop\shims\llama-cli.exe"
OUTPUT_DIR = r"D:\gal\AliceInCradle\lyco-model\results"
CHAT_MODEL = os.path.join(MODEL_DIR, "chat_slm_qwen3_0p6b-Q4_K_M.gguf")

CTX_GGUF = (
    "llama.cpp 是用 C/C++ 写的本地大模型推理引擎，轻量，不需要 Python、CUDA 等重型依赖。"
    "llama.cpp 使用 GGUF 文件格式。GGUF 支持量化权重和内存映射，降低设备内存带宽占用。"
    "GGUF 是单文件二进制格式，包含模型权重、tokenizer 和元数据。"
    "来源：https://huggingface.co/docs/transformers/main/community_integrations/llama_cpp")

CTX_Q4KM = (
    "Q4_K_M 是 llama.cpp 的 K-quant 量化预设之一：Q 表示约 4 bit 权重，K 表示 K-quant 系列，"
    "M 表示 Medium（中等）的混合精度分配策略（S 小 / M 中 / L 大）。"
    "Q4_K_M 在 Llama-3-8B 上约为 4.58G，perplexity 上升约 +0.1754，是默认推荐的平衡档。"
    "来源：llama.cpp tools/quantize/quantize.cpp QUANT_OPTIONS；"
    "https://github.com/iuliaturc/gguf-docs/blob/main/naming.md")

CASES = [
    {"q": "什么是 GGUF？它和 llama.cpp 是什么关系？", "ctx": CTX_GGUF},
    {"q": "什么是模型量化？Q4_K_M 是什么意思？", "ctx": CTX_Q4KM},
]

TEMPLATE = ("【检索到的资料】\n{ctx}\n\n【用户问题】\n{q}\n\n"
            "要求：只根据上述资料回答，资料没有的内容就说不知道。"
            "用2-4句话总结要点。")


def run_once(prompt, n_predict=256):
    t0 = time.time()
    p = subprocess.run(
        [LLAMA_CLI, "-m", CHAT_MODEL, "-p", prompt, "-n", str(n_predict),
         "--single-turn", "--simple-io", "--no-display-prompt"],
        capture_output=True, timeout=180)

    def dec(b):
        if not b:
            return ""
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    dt = time.time() - t0
    out = dec(p.stdout) + dec(p.stderr)
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    body = out.split(prompt[-20:])[-1] if prompt[-20:] in out else out
    body = re.sub(r"\[ ?Prompt:.*", "", body).strip()
    return {"ok": p.returncode == 0, "response": body[-3000:],
            "gen_ts": float(m.group(1)) if m else None,
            "elapsed": round(dt, 2)}


res = []
for i, c in enumerate(CASES, 1):
    prompt = TEMPLATE.format(ctx=c["ctx"], q=c["q"])
    r = run_once(prompt)
    r["question"] = c["q"]
    res.append(r)
    print(f"[rag] {i}/{len(CASES)} el={r['elapsed']}s", flush=True)

with open(os.path.join(OUTPUT_DIR, "rag_demo.json"),
          "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("saved rag_demo.json")
