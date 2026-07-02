# 01. Tokenizer 与数据治理

[返回执行手册](../02-training-runbook.md)

从零预训练前，先把两件事固定下来：词表和数据边界。词表一旦开始训练模型就不要再改；数据一旦进入训练集，就要能追溯来源、许可证、仓库、commit 和 split。

## 1. 训练 32K tokenizer

目标是固定 Mini-GLM 的 tokenizer：

```text
vocab_size = 32768
tokenizer output = tokenizer/miniglm-32k
```

训练语料放在：

```text
data/miniglm/tokenizer_corpus/train.jsonl
```

每行格式：

```json
{"text": "这里是一段训练 tokenizer 的文本或代码"}
```

语料需要覆盖：

```text
中文、英文、代码、Markdown、JSON/YAML/TOML、shell
diff / patch
traceback / pytest / terminal logs
tool_call / tool_response / think 标签
```

先看前三行：

```bash
head -n 3 data/miniglm/tokenizer_corpus/train.jsonl
```

这一步只检查格式，不检查质量。先确认 JSONL 不为空、字段名正确、编码没有明显问题。

开始训练：

```bash
python trainer/train_tokenizer.py \
  --input data/miniglm/tokenizer_corpus/train.jsonl \
  --vocab-size 32768 \
  --output-dir tokenizer/miniglm-32k
```

如果当前脚本还没有这些参数，先把 `trainer/train_tokenizer.py` 的硬编码常量改成 CLI。

这条命令怎么读：

```text
--input:
  tokenizer 学习切分规则的语料。

--vocab-size 32768:
  固定词表大小。后续模型 config 的 vocab_size 必须等于它。

--output-dir:
  输出 tokenizer.json、tokenizer_config.json、special_tokens_map.json 等文件。
```

验收 tokenizer：

```bash
python - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer/miniglm-32k")
samples = [
    "你好，请解释 MoE 和 dense 模型的区别。",
    "def add(a, b):\n    return a + b\n",
    "diff --git a/src/a.py b/src/a.py\n@@ -1,3 +1,3 @@\n-pytest.fail()\n+assert True\n",
    "Traceback (most recent call last):\n  File \"tests/test_a.py\", line 10, in test_x\nAssertionError\n",
]

print("vocab", len(tok))
for s in samples:
    ids = tok.encode(s)
    print("-" * 80)
    print("chars", len(s), "tokens", len(ids), "chars/token", round(len(s) / max(len(ids), 1), 2))
    print(tok.decode(ids)[:200])
PY
```

这段验收看三件事：词表长度是否等于 32768，中文/代码/diff/traceback 是否能正常往返，工程文本是否被切得过碎。不要追求一个固定的 chars/token；它只是诊断信号。

过关标准：

```text
len(tok) == 32768
中文、英文、代码能正常 encode/decode。
diff/traceback 不被切得过碎。
chat_template 存在并能 apply_chat_template。
```

## 2. 建立数据审计

训练数据不是“能下载就能训”。尤其你想参与 SWE-bench 一类测评时，训练集必须能证明没有污染官方评测集。

manifest 文件：

```text
data/miniglm/audit/data_manifest.jsonl
```

每行记录一条样本：

```json
{
  "sample_id": "pretrain_base_00000001",
  "source": "the-stack-v2",
  "license": "mit",
  "repo": "owner/name",
  "commit": "abc123",
  "url": "https://example.com/owner/name",
  "split": "train",
  "sha256": "hex",
  "tags": ["code", "python", "pretrain"]
}
```

SWE-bench 黑名单：

```text
data/miniglm/audit/swebench_blacklist.json
```

至少记录：

```json
{
  "instances": [
    {
      "benchmark": "swe-bench-verified",
      "instance_id": "repo__name-12345",
      "repo": "owner/name",
      "base_commit": "abc123",
      "issue_url": "https://github.com/owner/name/issues/1",
      "pr_url": "https://github.com/owner/name/pull/2",
      "problem_statement_hash": "hex",
      "patch_hash": "hex",
      "test_patch_hash": "hex"
    }
  ]
}
```

建议 split：

```text
repo-level split:
  同一个 repo 不同时出现在 train 和 eval。

time split:
  早期 commit 训练，较晚 commit 评估。

task split:
  同一个 issue/PR 不同时出现在 train 和 eval。
```

推荐文件布局：

```text
data/miniglm/pretrain_base/train.jsonl
data/miniglm/pretrain_base/valid.jsonl
data/miniglm/repo_context/train.jsonl
data/miniglm/repo_context/valid.jsonl
data/miniglm/sft_issue_patch/train.jsonl
data/miniglm/sft_issue_patch/valid.jsonl
data/miniglm/sft_agent_trajectory/train.jsonl
data/miniglm/sft_agent_trajectory/valid.jsonl
```

过关标准：

```text
训练数据能按 repo / url / commit / patch hash 排除官方 SWE-bench 实例。
每条训练样本能追溯 source、license、repo、commit、split。
```
