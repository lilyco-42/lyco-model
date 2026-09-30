"""Offline assertions for the claim checker (digit_claims + unsourced_claims).

Two of these are the bugs the first version had:
  - "UINT8" in a list of GGUF tensor types "supported" an unsourced 8-bit claim
  - evidence written 400MB must support an answer saying 400 MB
Case "0.6B's 1.1GB" is the must-not-false-fail regression.

Run: python test_claims.py
"""
import rag_loop as R

CASES = [
    ("unit-attached number with no source",
     "量化会将浮点数转换为低精度整数（如 8-bit），导致精度下降。",
     "【DeepWiki】Q4_K_M 是 4 bit 的 K-quant 分块格式，困惑度 +0.1754。",
     {"8"}, {"8"}),
    ("UINT8 must not pass as support for 8-bit",
     "量化会将浮点数转换为低精度整数（如 8-bit）。",
     "数据类型如 `UINT8`、`INT32`、`FLOAT32`、`STRING` 等",
     {"8"}, {"8"}),
    ("same number present in evidence as its own token",
     "文件约 400 MB。", "size: 400 MB, params: 614.7M", {"400"}, set()),
    ("evidence 400MB supports an answer saying 400 MB",
     "文件约 400 MB。", "the file is 400MB", {"400"}, set()),
    ("list markers are not claims",
     "1. 精度损失\n2. 性能下降", "irrelevant", set(), set()),
    ("quant names are not claims",
     "Q4_K_M 是常用档位。", "Q4_K_M 是常用档位", set(), set()),
    ("percentage is a claim",
     "速度提升 30%。", "困惑度 +0.1754", {"30%"}, {"30%"}),
    ("percentage supported by evidence",
     "速度提升 30%。", "throughput improves by 30%", {"30%"}, set()),
    ("token match, not substring",
     "延迟 12.5ms。", "总耗时 112.55ms", {"12.5"}, {"12.5"}),
    ("0.6B's 1.1GB is supported as 1.1 GB",
     "GGUF 文件的大小约为 1.1GB，比 llama.cpp 更小。",
     "the Q4_K_M file is 1.1 GB", {"1.1"}, set()),
    ("three-digit number",
     "上下文长度 8192。", "ctx 8192", {"8192"}, set()),
    ("bare small int in prose stays harmless",
     "由四个主要部分组成，第 2 部分是 KV 元数据。", "四个主要部分",
     set(), set()),
]


def check(name, resp, hay, want_claims, want_flagged):
    claims = R.digit_claims(resp)
    flagged = set(R.unsourced_claims(resp, hay))
    ok = claims == want_claims and flagged == want_flagged
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"   claims={claims} want={want_claims} "
              f"flagged={flagged} want={want_flagged}")
    return ok


if __name__ == "__main__":
    results = [check(*c) for c in CASES]
    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)
