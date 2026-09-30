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
chat（总结）+ verify（交集检查）。

## 9. 本地库 + RSS 接入（gh 自家 repo + rustcc.cn/rss）

`retrieve()` 升级为多源排序：mpkg/lilyco 类问题 → 本地 gh-clone repo 优先；
Rust 类 → RustCC RSS 优先；repo 问题 → DeepWiki；其余默认顺序。取满 2 源
（证据上限 1200 字），不再首中即停。RSS 1 小时本地缓存（`results/rustcc_rss.xml`）。

5 题验证（记录 `results/rag_loop_demo.json`）：

| 问题 | 路由 | 证据源 | 结果 |
|------|------|--------|------|
| mpkg 记忆包是什么 | search | 本地 mpkg-registry README | ✅ 内容寻址/可回放，答对 |
| 纯 Rust 的 Luau 运行时 | search | RustCC 当日 ulua 文章 | ✅ 寄存器虚拟机/渐进类型，答对 |
| GGUF 与 llama.cpp 关系 | search | 本地 + DeepWiki | ✅ 四段结构/Header/KV，全对 |
| Q4_K_M 精度损失 | search | DeepWiki | ⚠️ 此轮 DeepWiki 未给 +0.1754，模型用表格数作答（4.58GiB/速度），有据、无编造 |
| 讲笑话 | direct | — | ✅ 不检索 |

修 bug 记录（验证喂回）：
- router 漏"是什么"句式 → 补触发词；"什么/关系"等高频词污染检索 → 停用词表 + anchor 规则（alnum≥2 或中文≥3 字才算锚点）
- 长 prompt 回显被截断 → 按 `...(truncated)` 标记切答案；去掉 `--simple-io` 后短 prompt 回显固定为 `"> "+prompt`
- DeepWiki `ask_question` 问答是非确定性的——同一题两次返回细节不同，关键数字以多次为准

结论：四源（本地/RSS/DeepWiki/维基降级）+ router + verify 的闭环成立。
维基 zh API 在本机出口 403 仍是已知缺口；RSS/本地已补上，百科类走 DeepWiki 顶。

---

## 10. 多轮记忆 + 证据缓存 + verify 升级

第 9 节末尾自评的三个缺口，动了两个半：没有多轮记忆、verify 是玩具；"0.6B 总结薄"属于模型层，没动。

**Runner：llama-cli → llama-server**

`llama-cli --single-turn` 每轮重载模型、没有上下文，答案还得从 stdout 回显里按 `...(truncated)`
切——第 9 节修的 4 个 bug 里，"回显截断提取"和"verify 被回显污染"这两个都出在这套解析上。
`llama-server` 常驻 + OpenAI 兼容
`/v1/chat/completions`：历史直接进 messages，`temperature=0` 贪心，回显解析整段删掉。
脚本自己拉起或复用 server（复用前校验 `/v1/models` 真的是目标 GGUF，防止静默连到别的模型）。

| | llama-cli 单轮 | llama-server |
|---|---|---|
| 单轮耗时（含检索） | ~3.0 s（模型加载占大头） | 0.31–3.46 s |
| 上下文 | 无 | 最近 4 轮 user/assistant（≤900 字） |
| 答案提取 | 按回显截断标记切 | `choices[0].message.content` |

### ① 多轮记忆（`Conversation`）

- 结构：system(指令) + 历史轮 + 本轮(资料+问题)。资料只挂本轮、不进历史，否则第 2 轮上下文就爆。
- 指代解析：出现 `它/那/还有/然后…` 判为 follow-up，检索词 = 上一轮**已解析**的查询 + 本轮问题，
  锚点因此逐轮累积。第一版把 topic 存成原始问题，第 3 轮就退化成"它和 llama.cpp…"，GGUF 丢了
  ——这坑是 dry-run（不起模型、只 stub 检索）打出来的。

同一 4 题，有记忆 vs 每题新开会话（`results/rag_loop_memory.json` 的 memory / memory_control）：

| 轮 | 有记忆 | 无记忆（对照） |
|---|---|---|
| "它和 llama.cpp 是什么关系？" | ev=2，答出 GGUF↔llama.cpp | ev=1，只讲 llama.cpp/ggml，"它"丢了 |
| "那 Q4_K_M 又是什么？" | ev=2，量化 + GGUF 张量存储格式 | **ev=0**，编出"衡量模型在 Q4 期间的性能表现" → verify `no-evidence` fail |
| "它会让模型跑得更快吗？" | ev=3，router 走检索 | router 判 direct 不检索，直接"是的，它会让模型跑得更快。" |

4 轮累计耗时：首次联网 64.98 s → `--offline` 重放 7.26 s；对照组 13.14 s → 4.01 s。
记忆比对照贵近一倍，但对照组省下的时间花在编造上。

### ② 证据缓存（`results/evidence_cache.json`，24h TTL，`--offline` 只读缓存）

DeepWiki 同一题每次返回细节不同（第 9 节的已知缺口）。现在 key = `sha1(source|repo|question)`，
命中即不联网。live 列取自 `results/rag_loop_live.json`（首次联网），cache 列取自
`results/rag_loop_memory.json`（杀掉 server 后脚本自己拉起、`--offline` 重放那次）。

| 轮 | live | cache 命中 |
|---|---|---|
| GGUF↔llama.cpp | 16.00 s | 2.71 s |
| Q4_K_M 精度损失 | 16.81 s | 1.45 s |
| 三组 13 轮总计 | 118.56 s | 22.33 s |

跨 4 次运行（首次 live、cache 首触、两次 `--offline`）：**13/13 轮证据逐字一致**——DeepWiki
曾是唯一的非确定源，现已消除。
文本层面只用入库的两个 artifact 复核：`rag_loop_live.json` vs `rag_loop_memory.json` →
回复 13/13 逐字相同，verdict 12/13；唯一变化的那行是记忆组第 4 轮，它原先被误判成幻觉（见③）。
但别当定理：server prompt cache 处于热状态时，同代码的另一次 `--offline` 重放只有 8/13 逐字相同
（分叉形如"同机器上的回放是可靠的" ↔ "会丢失或损坏"，并顺着历史传给后续轮），verdict 仍 13/13。
结论：可复现的是**证据与结论**；逐字文本取决于 server 的缓存状态。

### ③ verify 不再是玩具

交集只证明"抄了"，不证明"抄对"。四道机检 + 一道参考：

- `invented_numbers`：答案里带精度含义的数字（小数/百分比/≥3 位）不在证据里 → fail
  （正是第 9 节那个 0.121 的形态）
- `unsupported_terms`：证据里没有的英文技术词 >1 个 → fail
- `on_topic`：检索词锚点必须出现在答案里，否则算答非所问
- 证据池 = 本轮 + 本会话历史证据。只查本轮时，第 4 轮复述第 1 轮证据里的 `Header Section`
  被判成幻觉——误杀，测出来的
- `judge`：第二遍让模型自查"答案用到资料了吗"，只记录不作门禁。0.6B 判 0.6B 不可信，
  实测连"只回答 是/否"都守不住（答成"答案用到了资料"）
- `hits` 采样改 sorted：set 遍历顺序受 hash 随机化影响，判定不变但记录会抖，重放要求下必须钉死

回归：独立 5 题 5/5 过、记忆 4 轮 4/4 过、对照组两处按预期 flag（1 处 fail + 1 处 direct 漏网）。

**已知缺口（没藏）**：`router:direct` 完全免检，对照组第 4 轮就是这么蒙的——指代句要靠记忆
才路由得到，接更多源治不了；0.6B 总结仍会缝合（"模型参数和元数据"重复两遍），换 1.7B/4B 仍是
下一步；维基 zh API 本机出口仍 403，缓存层对它只能记 miss；server 单实例 `-np 1`，并发要另起端口。

（前两条在第 11 节处理掉了，后两条仍在。）

---

## 11. 路由补洞 + 缓存投毒 + 0.6B vs 1.7B 对照

套件扩到 **17 轮**（5 独立 + 6 记忆 + 6 对照），产物 `results/rag_loop_0p6b_v2.json`、
`results/rag_loop_1p7b_v2.json`。两次运行**证据 17/17 逐字一致**，所以这张对照表是可比对的。

### ① router 的 direct 免检洞

触发条件从"句子里有没有 什么/为什么"扩成三类：`SEARCH_TRIGGERS`、`PREDICTION_CUES`
（缺点/优点/影响/需要/会不会…）、以及"会话已有 topic 时出现指代词或带 吗 的问句"。

- `量化有什么缺点？`（不含任何"是什么"句式）现在两个模型都走检索。
- `给我讲个笑话`在有 topic 的会话里仍然判 direct（无指代、无 吗）——路由不能把闲聊拖进检索，
  这条是回归项。
- direct 轮不再完全免检：仍查"带精度含义的数字是否在问题或证据池里"。stub 用例：
  `延迟大约是 12.5ms` → `direct-unsourced-numbers` fail；正常讲笑话 → pass。
- 没记忆时 `它会让模型跑得更快吗？` 依旧无法路由（指代没有先行词，搜也无从搜起）。
  这个洞只能靠记忆补，写在这里不藏。

### ② verify 的两处误杀/漏判

- `on_topic` 原先用 `is_anchor()` 取锚点，而它故意丢掉所有两字中文词——可 `量化`、`缺点`正是
  正确答复必含的词，结果正确答复被判跑题。改为 `relevance_terms()`：按**本轮问题**取非停用词的
  内容词（含双字），不再拿累积后的检索串当判据。
- 无证据分支原先只认字面"不知道"。1.7B 答"我目前没有关于这一术语的具体信息"是**正确行为**却被判
  fail。改为 13 条 `DECLINE` 短语族。改后：0.6B 编的"Q4_K_M 是衡量模型在 Q4 期间性能的指标"照旧
  fail，1.7B 的诚实拒答转 pass。
- 证据池语义被验证有效：`量化有什么缺点？` 这一轮正赶上 DeepWiki 429、本轮零证据，有记忆的会话靠
  池子里前几轮检索到的量化资料判 pass；对照组同样问题没有池子 → **两个模型都 fail**（0.6B 还把
  "量化"定义成"通过数学模型和算法描述分析数据的方法"，彻底答错域）。
- 第 10 节那 13 轮记录用新检查器重打分后判定不变（12/13，零改判），两节不打架。

### ③ 缓存层把垃圾冻住了（②查出来的新缺口）

`deepwiki_ask` 收到 HTTP 200，但 `content` 里是上游报错原文：
`Error processing question: Client error '429 Too Many Requests' for url 'https://api.devin.ai/ada/query'`。
旧代码只滤 `__ERR__` 前缀，于是这段 429 被当证据缓存 24 小时，还被模型当资料复述。第 8 节
"垃圾证据进 → 幻觉出"从缓存层又回来了一次。
修：`looks_like_error()` 写入前拦、读缓存时也拦（命中即丢弃并记 `cache-dropped-error`），
本地缓存清掉 1 条已投毒条目。
这个失效模式做成了可复现的记录 `results/cache_poison_demo.json`（生成脚本 `poison_demo.py`）：
把那段 429 手工当证据喂给 0.6B 问"量化有什么缺点？"，它答——
> 1. 无法处理大规模数据，因为每次请求都会触发一次429错误，导致请求次数过多。
> 2. 无法处理实时数据…（同句）3. 无法处理高并发请求…（同句）4. 与第 1 条一字不差
即：垃圾资料被原样复述 4 遍，其中两条完全相同。同一份记录里 `gate_rejects_this_text: true`
证明现在的代码不会再缓存它，`verify` 也判 `low-overlap,off-topic` fail——拦、弃、查三层都在。

### ④ 0.6B vs 1.7B（同套件、同证据）

| 指标 | chat_slm 0.6B | Qwen3-1.7B |
|---|---|---|
| verify 通过 | 15/17 | 16/17 |
| judge 守"只回答是/否" | 4/6 | 6/6 |
| 出现 ≥12 字重复缝合 | 6/17 | 4/17 |
| 平均回复长度 | 171.6 字 | 124.9 字 |
| 平均生成速度 | 55.6 t/s | 24.9 t/s |
| 17 轮总耗时（证据全命中缓存） | 30.35 s | 48.95 s |

判定用的是当轮的检查器；第 12 节把带单位数字的漏检补上后，同一批记录重打分改了 1 行
（1.7B 那句无据的 "8-bit"），见 `results/rescore_claims.json`。

内容差别比数字直接，两条都取自 `memory` 组：

- `它会让模型跑得更快吗？`（ev=3）：0.6B 列 4 条，**第 3、4 条一字不差**
  （"GGUF 文件的大小约为 1.1GB，比 llama.cpp 更小，因此在推理速度方面可能更快。"）；
  1.7B 一段话收住，无复读。
- `量化有什么缺点？`（本轮 429、ev=0，见③）：0.6B 的 4 条共用同一个模板
  "量化会将模型参数压缩为小的二进制向量，从而减少了模型的 X"，X 依次填 精度/鲁棒性/泛化能力/
  计算开销，第 4 条还自相矛盾（"减少计算开销 → 推理速度变慢"）；1.7B 给 5 条各不相同
  （精度损失/性能下降/可解释性/训练难度/硬件支持）。

1.7B 也不干净，照写：它说量化"可能导致推理速度变慢"（实际量化是加速推理），还举了证据里没有的
"8-bit" 举例——`digit_claims` 只认小数/百分比/3 位以上数字，个位数不当精度声明，所以这条漏检。
判官有信号：1.7B 在第 2 轮答"否"，自己认定那轮答案没用到资料。

代价：慢 2.2×（24.9 vs 55.6 t/s），但回复短 27%，单轮墙钟 1.79 s → 2.88 s，CPU 端侧仍可用；
权重从 0.48 GB 到 1.11 GB。

**口径提醒**：1.7B 用的是 `unsloth/Qwen3-1.7B-GGUF` 通用指令模型，**不是 lyco 自家微调**，
而且没走 `chat-slm` 那条中文数据管线。本轮只证明"总结位换个更大的模型就够用"，不证明 1.7B
微调版行为一致——`lyco42` 目前没有 1.7B 版本。

结论：第 5 节"后续建议"和第 10 节自评的缺口到本节闭合——路由、记忆、缓存、校验四层都有实测边界。
下一步：(a) 用 1.7B 蒸馏/微调一版中文总结器；(b) DeepWiki 加 429 退避重试，而不是只丢弃。

---

## 12. 限流退避、claims 补漏、维基判死、server 并发

第 11 节末尾留下的洞，这节收掉四个。全部有可复核产物：`results/source_probe.json`、
`results/rescore_claims.json`、`results/server_bench.json`，加两个离线断言脚本
`test_retry.py`（7/7）、`test_claims.py`（12/12）。

### ① DeepWiki 限流：退避重试，且只重试限流

实测（`source_probe.json`）：连续 3 次问答都在 **0.20–0.24 s** 返回同一个 429 —— 秒回，
说明上游（`api.devin.ai`）配额是窗口制的，亚秒级重试纯属白打。于是 `deepwiki_ask()`：

- 区分限流与真错误：`is_ratelimit()` 认 429/Too Many Requests/rate limit/quota；
  `repo not found` 这类**只试一次**就返回，不浪费等待。
- 指数退避 2 s → 4 s（`backoff_sleeps()`，上限 `RETRY_MAX`），**故意不带随机抖动**：
  加了随机数，重放的记录就对不上了。
- 放弃后返回 `__ERR__ deepwiki ratelimited after 3 try: …`，`cached_lookup` 拒收 `__ERR__`，
  所以限流文本不会进缓存。

线上跑一遍真实路径：3 次尝试、等待 [2.0, 4.0]、总 7.49 s、`cached_as_evidence: false`。
代价是限流期每题多花 ~7 s，换来的是不再拿报错当资料。

### ② 带单位的数字不再漏检（顺带修了一个假支持）

`digit_claims()` 原来只认小数/百分比/3 位以上，于是 1.7B 那句"低精度整数（如 **8-bit**）"
（证据里根本没提 8-bit）大摇大摆过了门禁。现在规则是：小数 / 百分比 / ≥3 位 / **任何粘着单位的数字**
（bit、byte、[KMGT]iB、t/s、ms、秒/分钟/小时/天/年/倍/%…）。

支持判定也从"子串包含"改成 token 边界 + 单位 span 折叠，两条各治一个错：

- `(?<![\w.])8(?![\w.])`——之前证据里的 **`UINT8`**（GGUF 张量类型列表）能给 "8-bit" 当支持，
  这是假阴性；现在字母粘连不算。
- 单位 span 折掉空格/连字符再比——`400MB` 必须能支持答复里的 `400 MB`，否则就是误杀。
- `12.5` 不再因为证据里有个 `112.55` 就蒙混过关。

**在已知答案上验，再谈数字**：`test_claims.py` 12 条断言（含"0.6B 的 1.1GB 必须不算编造"这条
防误杀回归）。4 份入库记录 60 行重打分（`rescore_claims.py`）只改判 **2 行**：

| 记录 | 行 | 改判 | 原因 |
|---|---|---|---|
| `rag_loop_1p7b_v2` | memory `量化有什么缺点？` | pass → **fail** | 无据的 `8-bit`，新规则抓到 |
| `rag_loop_live` | memory `它会让模型跑得更快吗？` | fail → pass | 第 11 节证据池修复（复述上一轮证据不算幻觉） |

54 行原本 pass 的只有 1 行翻成 fail，且翻的那行确实是编造 —— 规则变严但没误伤。
按现检查器重算，第 11 节那张对照表变成 0.6B 15/17、1.7B 15/17。

### ③ 维基百科：判死，不是"待验证"

7 种出口全试（`source_probe.json`）：`opensearch` 项目 UA / 浏览器 UA / 空 UA、REST summary、
`action=query&prop=extracts`、英文站 REST —— **全部 403**（Wikimedia 错误页 HTML）；
绕开系统代理直连 —— **超时**。结论是代理出口被 Wikimedia 挡，且本机没有直连路径，
不是 UA 问题。于是把 `wiki_search()` 整段删掉、`source_order()` 不再含 wiki，
百科类问题回落 DeepWiki。留着的注释指向这份产物，换出口要接回来时看得到依据。

### ④ llama-server 并发与内存（CPU，`-temp 0`）

先说踩到的坑：**`-c` 是总上下文，按槽位分摊**。`-np 4` 时一个槽只有 2048 token，
5423 token 的请求被 HTTP 400 直接拒绝（`request (5423 tokens) exceeds the available
context size (2048 tokens)`）。`-c 16384 -np 4` 每槽 4096，仍然拒。只测延迟和内存的话，
会得出"np 免费"的错结论。

| 配置 | 单请求均值 | 平均生成速度 | 2 并发墙钟 | 2 并发聚合速度 | 5.4k token 长请求 | 内存（空载→跑完） |
|---|---|---|---|---|---|---|
| 0.6B ctx8192 np1 | 0.85 s | 92.4 t/s | 2.13 s | 99.7 t/s | 接受 | 1523 → 1600 MB |
| 0.6B ctx8192 np4 | 0.75 s | 89.0 t/s | 1.39 s | 133.9 t/s（+34%） | 拒（2048/槽） | 1511 → 1571 MB |
| 1.7B ctx8192 np1 | 2.40 s | 43.8 t/s | 5.62 s | 45.5 t/s（无收益） | 接受 | 2382 → 2480 MB |
| 1.7B ctx8192 np4 | 2.24 s | 47.2 t/s | 2.86 s | 89.5 t/s（+97%） | 拒 | 2373 → 2449 MB |
| 0.6B ctx16384 np4 | 0.73 s | 91.0 t/s | 1.35 s | 138.0 t/s | 仍拒（4096/槽） | 2413 → 2474 MB |
| 0.6B ctx16384 np1 | 0.82 s | 95.7 t/s | 2.10 s | 100.9 t/s | 接受 | 2436 → 2528 MB |
| 两模型常驻（各 np1 ctx8192） | 0.83 / 2.23 s | 94.4 / 47.0 t/s | 各 1 请求：墙钟 2.83 s | — | — | 1608 + 2432 = **4040 MB** |

口径：

1. `rag_loop.py` 保持 `-np 1`。np>1 的并发收益是真的（1.7B 近 2 倍），但它把每槽上下文
   砍成 `ctx/np`，而我们的证据+历史 prompt 就是几千 token 级别 —— 并发不能拿正确性换。
   要多 agent 共享一个 server，就 `-c` 和 `-np` 一起加，保证 `ctx/np` ≥ 最长 prompt。
2. 想并发就开新实例更省心：两个模型各自常驻互不干扰（0.6B 仍 94 t/s、1.7B 仍 47 t/s），
   代价是 4.0 GB 内存。
3. 上下文池才是内存大头：0.6B 权重 0.48 GB，但 `-c 8192` 空载已占 1.5 GB；
   `-c 16384` 再加 ~0.9 GB。端侧要省内存先砍 ctx，不是砍 np。
4. `chat()` 现在把 server 的 400 原文抛出来（含 available context size），不再是一句
   含糊的 `raise_for_status`；槽位配错一眼就能看出来。
5. 采样口径提醒：单请求 128 token 上限、每题 3 轮、CPU 独占，均值 ±10% 属噪声；
   `agg_tok_s` 用服务端 `completion_tokens` 实算 —— 0.6B 平均只输出 78 token，
   拿 `max_tokens` 当实际输出量会按 128/78 虚高 1.6 倍（第一版 bench 就是这么错的）。

**端到端回归**（改完这 6 处后用最终代码重跑同一套件，`results/rag_loop_0p6b_v3.json`，
`--offline` 17 轮 26.56 s）：独立 5/5、记忆组 6/6、对照组 4/6 —— 两处 fail 正是 0.6B 在无证据时
编造（"Q4_K_M 是衡量模型在 Q4 期间性能的指标"、把"量化"定义成统计学方法），而记忆组靠证据池
把同一题判过（`量化有什么缺点？` ev=0、overlap=4 走池子）；`给我讲个笑话` 在三个场景里都仍是
`router:direct`，闲聊没被拖进检索。

**仍未解决**：DeepWiki 配额一到，多源退化成 local+RSS 两源，报告里那些"有据"的答案
会跟着变薄；429 的**退避重试只解决了不投毒，没解决拿不到资料**。下一步是 (a) 用 1.7B
蒸馏/微调一版中文总结器，(b) 给 DeepWiki 加第二个仓（或本地 llama.cpp 文档库）做热备，
(c) RSS 自建（FreshRSS）以摆脱单一 feed。

---

*完整 200 轮原始记录见 `results/` 目录。*
