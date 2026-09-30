# 100 轮实际对话 Demo 报告

**Date:** 2026-09-30
**Models:** `lyco42/lyco-agent-qwen3-0.6b-ondevice` / `lyco42/chat-slm-qwen3-0.6b-zh`
**Runner:** llama.cpp `llama-cli` 0.4.1-dev, Q4_K_M GGUF, CPU, `--single-turn`, prompt 100 轮/模型
**Raw logs:** `results/agent_100rounds.jsonl`, `results/chat_100rounds.jsonl`, `results/demo_samples.json`

---

## 1. 100 轮稳定性实测

| 模型 | 轮数 | 成功 | 成功率 | 平均生成速度 | 平均单轮耗时 |
|------|------|------|--------|--------------|--------------|
| lyco-agent (grpo-Q4_K_M) | 100 | 100 | 100% | ~114.8 t/s | ~3.24 s |
| chat-slm (chat_slm-Q4_K_M) | 100 | 100 | 100% | ~114.3 t/s | ~3.03 s |

结论：两个模型 100 轮零失败，端侧 CPU 上 ~115 t/s，单轮 ~3 秒（含模型加载）。**稳定性和速度达标。**

---

## 2. Agent 模型 demo（定位：CLI agent / NL2CLI）

特点：带 `[Start thinking]...[End thinking]` 推理过程，中文回应。

- 第 1 轮（自我介绍）：思考过程完整，但自称"小明，18 岁，QQ 1234567890"——**身份不稳定**（另一次自称"罗志远"）。
- 第 2/3/5 轮（写 Python / bash / Dockerfile）：256 token 预算被 thinking 吃完，**只剩思考、没有输出正文代码**。
- 第 4 轮（解释 RAG）：给出了回答，但把 RAG 说成 "Retrieval-Augmented Knowledge…结合检索与知识图谱"——**事实错误**（应为 Retrieval-Augmented Generation，检索增强生成）。

**定位符合度：✅ 形态对（会思考、会规划步骤），❌ 内容不可直接用。**
要落地 NL2CLI，必须：① system prompt 钉死身份；② 给 reasoning 单独预算（或关闭 thinking 直接出答案）；③ 关键命令加执行前二次确认。

---

## 3. Chat 模型 demo（定位：中文聊天）

特点：无 thinking，直接回答，中文流畅。

- 第 1 轮（自我介绍）：正常。
- 第 2 轮（Transformer）：回答通顺但含糊（"把输入分成多个序列再组合"——不准确）。
- 第 3 轮（GGUF 与 llama.cpp 关系）：**严重幻觉**——把 GGUF 说成 "Lua 引擎"，把 llama.cpp 说成"基于 C++ 的开源语言"。完全错误。
- 第 4 轮（量化 / Q4_K_M）：把量化说反（"将模型参数转换为浮点数"），Q4_K_M 解释为"4 个时间窗口"——错误。
- 第 5 轮（端侧模型）：大体方向对，但"训练和推理都需要依赖云端"自相矛盾。

**定位符合度：✅ 聊天形态对（流畅、中文好），❌ 技术问答不可信。**
0.6B 参数决定了它记不住准确的技术定义。适合闲聊、写作辅助、简单问答；**不适合作为技术知识来源**。

---

## 4. 总 verdict

| 维度 | Agent 模型 | Chat 模型 |
|------|-----------|-----------|
| 运行稳定性 | ✅ 100/100 | ✅ 100/100 |
| 推理速度 | ✅ ~115 t/s | ✅ ~114 t/s |
| 中文流畅度 | ✅ | ✅ |
| 身份稳定性 | ❌（小明/罗志远乱跳） | ✅（自称 AI 助手） |
| 推理过程 | ✅ 有 thinking | —（无） |
| 代码/命令可用性 | ⚠️ thinking 挤占输出预算 | ⚠️ 未重点测 |
| 技术事实准确性 | ❌（RAG 定义错） | ❌（GGUF/Lua 幻觉） |

**一句话：两个模型"跑得快、聊得顺、但不懂装懂"。**
这是 0.6B 端侧模型的物理上限，不是训练事故。正确用法：闲聊、草稿、CLI 草稿生成——凡是输出都要过一道人工/程序校验。

## 5. 后续建议

1. 给 agent 模型加 system prompt 钉死身份和输出格式（先给命令、再给解释）。
2. thinking 预算独立（如 `--reasoning-budget`），或 demo 时关闭 thinking。
3. 技术问答场景接 RAG/搜索，不要考模型记忆。
4. Bedrock 128 皮肤 / Live2D 那条线不受影响——那是美术资产，不是模型能力。

---

*完整 200 轮原始记录见 `results/` 目录。*
