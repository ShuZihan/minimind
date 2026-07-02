# 01. Tokenizer 与数据治理

[返回执行手册](../02-training-runbook.md)

从零预训练前，先把两件事固定下来：词表和数据边界。词表一旦开始训练模型就不要再改；数据一旦进入训练集，就要能追溯来源、许可证、仓库、commit 和 split。

先回到仓库根目录：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
```

前置条件：

```text
1. 已完成 runbook/00-setup-and-smoke.md 的依赖安装和官方数据下载。
2. 已完成 00-engineering-migration.md 中 train_tokenizer.py 的 CLI 改造。
3. dataset/pretrain_t2t_mini.jsonl 已存在。
```

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

先从官方预训练数据生成一个最小可跑的 tokenizer 训练语料。它不是最终语料，只是保证第一轮命令可以闭合：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
dst = Path("data/miniglm/tokenizer_corpus/train.jsonl")
target_bytes = 200 * 1024 * 1024

if not src.exists():
    raise SystemExit(f"missing {src}; finish runbook/00 data download first")

dst.parent.mkdir(parents=True, exist_ok=True)
written = 0
with src.open("rb") as fin, dst.open("wb") as fout:
    for line in fin:
        if not line.strip():
            continue
        fout.write(line)
        written += len(line)
        if written >= target_bytes:
            break

print(dst, written, "bytes")
PY
```

先看前三行：

```bash
cd "$MINIMIND_ROOT"
test -s data/miniglm/tokenizer_corpus/train.jsonl
head -n 3 data/miniglm/tokenizer_corpus/train.jsonl
```

这一步只检查格式，不检查质量。先确认 JSONL 不为空、字段名正确、编码没有明显问题。

开始训练：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_tokenizer.py \
  --input data/miniglm/tokenizer_corpus/train.jsonl \
  --vocab-size 32768 \
  --output-dir tokenizer/miniglm-32k
```

如果这里出现 `unrecognized arguments`，说明你还没完成 [../00-engineering-migration.md](../00-engineering-migration.md) 中 `train_tokenizer.py` 的 CLI 改造。先回去补，不要继续往后跑。

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
cd "$MINIMIND_ROOT"
python3 - <<'PY'
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

## 2. 准备 base pretraining 最小可跑文件

下一阶段会用到：

```text
data/miniglm/pretrain_base/train_100m.jsonl
data/miniglm/pretrain_base/train.jsonl
data/miniglm/pretrain_base/valid.jsonl
```

先用官方 `pretrain_t2t_mini.jsonl` 建一个最小可跑版本。正式训练时，你可以把这些文件替换为清洗后的 Mini-GLM 数据。

```bash
cd "$MINIMIND_ROOT"
mkdir -p data/miniglm/pretrain_base
ln -sfn ../../../dataset/pretrain_t2t_mini.jsonl data/miniglm/pretrain_base/train.jsonl
python3 - <<'PY'
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
train_100m = Path("data/miniglm/pretrain_base/train_100m.jsonl")
valid = Path("data/miniglm/pretrain_base/valid.jsonl")

if not src.exists():
    raise SystemExit(f"missing {src}; finish runbook/00 data download first")

def copy_bytes(dst, target_bytes, skip_bytes=0):
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        for line in fin:
            if skipped < skip_bytes:
                skipped += len(line)
                continue
            if not line.strip():
                continue
            fout.write(line)
            written += len(line)
            if written >= target_bytes:
                break
    print(dst, written, "bytes")

copy_bytes(train_100m, 100 * 1024 * 1024)
copy_bytes(valid, 10 * 1024 * 1024, skip_bytes=100 * 1024 * 1024)
PY
ls -lh data/miniglm/pretrain_base/train.jsonl \
       data/miniglm/pretrain_base/train_100m.jsonl \
       data/miniglm/pretrain_base/valid.jsonl
```

过关标准：

```text
tokenizer/miniglm-32k/tokenizer.json 存在。
data/miniglm/pretrain_base/train.jsonl 存在。
data/miniglm/pretrain_base/train_100m.jsonl 存在。
data/miniglm/pretrain_base/valid.jsonl 存在。
```

## 3. 建立数据审计

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
