# 04. Patch SFT、Agent SFT 与偏好优化

[返回执行手册](../02-training-runbook.md)

预训练让模型“会读代码”，SFT 才开始教它“按任务修代码”。这里分三步：先学直接输出 patch，再学按工具流程解决问题，最后用测试反馈做偏好优化。

先回到仓库根目录，并准备最小可跑样本：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
test -f tokenizer/miniglm-32k/tokenizer.json
ls out/miniglm_2b_a0_6b/stage2_repo_32k* >/dev/null
mkdir -p data/miniglm/sft_issue_patch data/miniglm/sft_agent_trajectory data/miniglm/preference_pairs
python3 - <<'PY'
import json
from pathlib import Path

def msg(role, content):
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }

issue_patch = Path("data/miniglm/sft_issue_patch/train.jsonl")
agent = Path("data/miniglm/sft_agent_trajectory/train.jsonl")
pref = Path("data/miniglm/preference_pairs/train.jsonl")

if not issue_patch.exists():
    issue_patch.write_text(json.dumps({
        "conversations": [
            msg("system", "You are a coding assistant. Produce a minimal unified diff."),
            msg("user", "Fix add(a, b) so it returns a + b."),
            msg("assistant", "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"),
        ]
    }, ensure_ascii=False) + "\n", encoding="utf-8")

if not agent.exists():
    agent.write_text(json.dumps({
        "conversations": [
            msg("system", "You are a coding agent. Use tools to inspect and fix the repository."),
            msg("user", "Fix add(a, b)."),
            msg("assistant", "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"src/a.py\"}}\n</tool_call>"),
            msg("tool", "def add(a, b):\n    return a - b\n"),
            msg("assistant", "Final patch:\ndiff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"),
        ]
    }, ensure_ascii=False) + "\n", encoding="utf-8")

if not pref.exists():
    pref.write_text(json.dumps({
        "chosen": [
            {"role": "user", "content": "Fix add(a, b)."},
            {"role": "assistant", "content": "return a + b"},
        ],
        "rejected": [
            {"role": "user", "content": "Fix add(a, b)."},
            {"role": "assistant", "content": "return a - b"},
        ],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
PY
```

这三个小文件只用于验证格式和训练链路。正式训练时，应替换为清洗后的 SWE-smith、自建 issue/PR、工具轨迹和测试反馈数据。

## 1. Issue-to-patch SFT

目标：输入 issue、repo context、相关文件、失败测试，输出 unified diff。

数据文件：

```text
data/miniglm/sft_issue_patch/train.jsonl
```

每行格式：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a coding assistant. Produce a minimal unified diff."
    },
    {
      "role": "user",
      "content": "Repository context:\n...\n\nIssue:\n...\n\nFailing tests:\n..."
    },
    {
      "role": "assistant",
      "content": "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,3 +1,3 @@\n-...\n+...\n"
    }
  ]
}
```

可训练：SWE-smith task instances、自建 GitHub issue/PR、内部 bug-fix tasks。

不可训练：官方 SWE-bench / Lite / Verified / Multilingual 及其 issue、PR、patch、test_patch。

训练命令：

```bash
torchrun --nproc_per_node 8 trainer/train_full_sft.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/sft_issue_patch/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage3_patch_sft \
  --from_weight stage2_repo_32k \
  --max_seq_len 16384 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --epochs 1 \
  --learning_rate 1e-5 \
  --save_interval 500 \
  --log_interval 20 \
  --dtype bfloat16 \
  --use_wandb
```

这条命令把 base/repo 模型转向“按需求生成补丁”。`train_full_sft.py` 训练 assistant 侧回答；`--from_weight stage2_repo_32k` 表示从长上下文模型继续；`--max_seq_len 16384` 要能装下 issue、相关文件、失败测试和 diff。

先做格式评估：

```bash
if test -f predicted.patch; then
  git apply --check predicted.patch
else
  echo "predicted.patch not found; save one model output to predicted.patch before running git apply --check"
fi
```

`git apply --check` 只检查补丁能否应用，不会修改文件。patch 格式都过不了时，不要急着看测试通过率。

过关标准：

```text
稳定输出 diff --git。
git apply --check 能通过。
不大量改无关文件。
不会复读 issue 原文。
```

记录指标：

```text
format_valid_rate
apply_success_rate
average_changed_files
average_changed_lines
```

## 2. Agent trajectory SFT

目标：训练模型按工具流程解决问题，而不是只直接输出 patch。

数据文件：

```text
data/miniglm/sft_agent_trajectory/train.jsonl
```

每行格式：

```json
{
  "conversations": [
    {"role": "system", "content": "You are a coding agent. Use tools to inspect and fix the repository."},
    {"role": "user", "content": "Fix this issue: ..."},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"function_name\", \"path\": \".\"}}\n</tool_call>"},
    {"role": "tool", "content": "src/a.py:10:def function_name(...):"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/a.py\"}}\n</tool_call>"},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"src/a.py\", \"patch\": \"...\"}}\n</tool_call>"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"run_tests\", \"arguments\": {\"cmd\": \"pytest tests/test_a.py -q\"}}\n</tool_call>"},
    {"role": "tool", "content": "1 passed"},
    {"role": "assistant", "content": "Final patch:\n<diff>\n...\n</diff>"}
  ]
}
```

训练命令：

```bash
torchrun --nproc_per_node 8 trainer/train_full_sft.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/sft_agent_trajectory/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage4_agent_sft \
  --from_weight stage3_patch_sft \
  --max_seq_len 32768 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --epochs 1 \
  --learning_rate 7e-6 \
  --save_interval 500 \
  --log_interval 20 \
  --dtype bfloat16 \
  --use_wandb
```

这条命令训练“过程能力”。数据里要有搜索、读文件、编辑、跑测试和工具返回；`--max_seq_len 32768` 用来容纳多轮工具调用和最终补丁。

过关标准：

```text
tool_call JSON 可解析。
先 inspect 再 edit。
不凭空编造文件路径。
能根据 test failure 继续修改。
```

## 3. 测试反馈优化

第一版先做 rejection sampling，不直接上复杂 RL。

流程：

```text
同一个 issue 采样 N 个 patch。
git apply --check。
运行相关测试。
pass 样本加入高质量 SFT。
pass/fail 组成 preference pairs。
```

preference 数据：

```text
data/miniglm/preference_pairs/train.jsonl
```

每行格式：

```json
{
  "chosen": [
    {"role": "user", "content": "Issue + repo context ..."},
    {"role": "assistant", "content": "Passing patch ..."}
  ],
  "rejected": [
    {"role": "user", "content": "Issue + repo context ..."},
    {"role": "assistant", "content": "Failing patch ..."}
  ]
}
```

偏好顺序：

```text
tests pass
> compiles but tests fail
> patch apply fail
> syntax error
```

DPO 命令：

```bash
torchrun --nproc_per_node 8 trainer/train_dpo.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/preference_pairs/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage5_preference \
  --from_weight stage4_agent_sft \
  --max_seq_len 32768 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --epochs 1 \
  --learning_rate 5e-6 \
  --save_interval 500 \
  --log_interval 20 \
  --dtype bfloat16
```

DPO 学的是偏好，不是新知识。`chosen` 应该是能 apply、能通过测试、改动更小的回答；`rejected` 是失败或更差的回答。学习率要小，避免破坏 SFT 学到的输出格式。

过关标准：

```text
patch apply 成功率不下降。
测试通过率上升。
输出没有明显变啰嗦。
```
