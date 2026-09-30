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

## 6. OODA 每轮闭环（lyco-skill）

方法：system prompt 取自 lyco skill 165-173 行模板（`ooda_sys.txt`），经 `-sys` 传入，
`-n 512`，每模型 20 轮代表性场景（`ooda_demo.py`，原始记录 `results/*_ooda.jsonl`）。

| 模型 | 五段全中 | 说明 |
|------|----------|------|
| agent | 18/20（90%） | thinking 里先规划五段再输出；第 1 轮是教科书级 OODA；2 轮缺一段（Decision/Decide 措辞漂移、thinking 挤占） |
| chat | 0/20（0%） | 完全无视 `-sys` 结构指令，直接给 plain 答案 |

结论：OODA-per-turn 只在带推理的 agent 模型上成立；chat 模型不跟结构指令——它的路是 RAG（下一节）。

## 7. Chat RAG：检索准确内容 + 模型总结

动机：chat 模型基线技术问答全靠编（GGUF→"Lua 引擎"、Q4_K_M→"4 个时间窗口"）。
做法：先检索准确资料拼进 prompt，模型只做总结，不考记忆（`rag_demo.py`，记录 `results/rag_demo.json`）。

| 问题 | 无检索（基线） | 有检索（RAG） |
|------|---------------|--------------|
| GGUF 与 llama.cpp 关系 | "开源 Lua 引擎，由 10 个开源项目组成" ❌ | "存储模型权重和元数据的单文件二进制格式，支持量化权重和内存映射" ✅ |
| Q4_K_M 含义 | "4 个时间窗口，四舍五入误差 0.1%" ❌ | "约 4bit K-quant，M=Medium 混合精度，4.58G/+0.1754，默认推荐平衡档" ✅ |

结论：符合要求——chat 不记知识，只做"搜到准确内容 + 合适总结"，这条路走通了。
生产形态 = 检索器（搜） + chat 模型（总结），模型本身不需要变。

---

## 8. rag_loop 全自动闭环（OODA per-turn 落地）

`rag_loop.py`：一轮对话 = 一轮 OODA。router（关键词启发式，闲聊直答）→
检索（维基百科 + DeepWiki 官方 MCP）→ chat 总结 → verify（证据关键词交集）→
超 2 轮或无证据则认不知道。DeepWiki MCP 已注册进全局 `opencode.json`，所有 agent 可用。

端到端 4 题全过（记录 `results/rag_loop_demo.json`）：

| 问题 | 路由 | 证据源 | 结果 |
|------|------|--------|------|
| GGUF 与 llama.cpp 关系 | search | DeepWiki | ✅ 四段结构/Header/KV/单文件容器，全对 |
| Q4_K_M 含义 | search | DeepWiki | ✅ 4bit K-quant/超块/+0.1754，全对 |
| Q4_K_M 精度损失 | search | DeepWiki | ✅ +0.1754（此前幻觉成 0.121） |
| 讲笑话 | direct | — | ✅ 不检索，直接答 |

验证方法论教训（OODA verify 喂回的真实发现）：
- DeepWiki 工具名是 `ask_wiki_question`、参数是 `repoName`——第一次调错，拿错误信息反推出正确 schema
- 维基百科 zh API 在本机出口 403（User-Agent 也救不回来），降级：DeepWiki 主力，维基待换出口再验
- 垃圾证据进 → 幻觉出（第一轮拿报错文本当证据，模型编出 0.121）——检索质量是总闸门
- 笑话类 direct 问题不能套"说不知道"模板；verify 不能把 prompt 回显算进交集

结论：chat 委派 search agent 的架构成立——router（笨规则）+ DeepWiki MCP（证据）+
chat（总结）+ verify（交集检查）。剩余缺口：lilyco 私有文档本地检索、RSS 接入。

---

*完整 200 轮原始记录见 `results/` 目录。*
