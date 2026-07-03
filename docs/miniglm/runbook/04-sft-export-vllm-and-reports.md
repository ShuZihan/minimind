# 04. SFT、偏好优化、导出、vLLM 与报告

[返回目录](../README.md)

这一章从预训练 checkpoint 走到可推理 checkpoint。顺序是 Patch SFT、Agent SFT、偏好优化、Hugging Face 导出、vLLM 验收，最后写阶段报告。

先回到仓库根目录：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
test -f tokenizer/miniglm-32k/tokenizer.json
ls out/miniglm_2b_a0_6b/stage2_repo_32k* >/dev/null
```

## 1. 准备最小 SFT/DPO 样本

这三个小文件只用于验证格式和训练链路。正式训练时，应替换为清洗后的 SWE-smith、自建 issue/PR、工具轨迹和测试反馈数据。

```bash
cd "$MINIMIND_ROOT"
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

## 2. Patch SFT

```bash
cd "$MINIMIND_ROOT"
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

如果保存了一个模型输出到 `predicted.patch`，可以检查 patch 格式：

```bash
if test -f predicted.patch; then
  git apply --check predicted.patch
else
  echo "predicted.patch not found; save one model output to predicted.patch before running git apply --check"
fi
```

过关标准：

```text
稳定输出 diff --git。
git apply --check 能通过。
不大量改无关文件。
不会复读 issue 原文。
```

## 3. Agent trajectory SFT

```bash
cd "$MINIMIND_ROOT"
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

过关标准：

```text
tool_call JSON 可解析。
先 inspect 再 edit。
不凭空编造文件路径。
能根据 test failure 继续修改。
```

## 4. 偏好优化

第一版先做 DPO。`chosen` 应该是能 apply、能通过测试、改动更小的回答；`rejected` 是失败或更差的回答。

```bash
cd "$MINIMIND_ROOT"
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

过关标准：

```text
patch apply 成功率不下降。
测试通过率上升。
输出没有明显变啰嗦。
```

## 5. 导出 Hugging Face checkpoint

前置检查：

```bash
cd "$MINIMIND_ROOT"
test -f scripts/export_miniglm_hf.py
test -f tokenizer/miniglm-32k/tokenizer.json
test -f configs/miniglm_2b_a0_6b.json
ls out/miniglm_2b_a0_6b/stage4_agent_sft_1536*.pth >/dev/null
```

导出：

```bash
cd "$MINIMIND_ROOT"
python3 scripts/export_miniglm_hf.py \
  --ckpt out/miniglm_2b_a0_6b/stage4_agent_sft_1536.pth \
  --tokenizer tokenizer/miniglm-32k \
  --config configs/miniglm_2b_a0_6b.json \
  --out hf/mini-glm \
  --safe-serialization true
```

检查：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

path = "hf/mini-glm"
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

print("model_type", cfg.model_type)
print("architectures", cfg.architectures)
print("vocab", cfg.vocab_size, len(tok))

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
)
print(type(model))
PY
```

## 6. vLLM 推理验收

启动前检查：

```bash
command -v vllm
```

启动 8K 服务：

```bash
vllm serve hf/mini-glm \
  --served-model-name mini-glm \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --trust-remote-code \
  --port 8998
```

保持服务运行，另开终端检查：

```bash
curl http://localhost:8998/v1/models
```

基础请求：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mini-glm",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Write a Python function that returns the sum of two numbers."}
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

稳定后再逐步尝试 `--max-model-len 16384`、`32768`、`65536`。

## 7. 阶段报告模板

每个阶段都写到 `reports/miniglm_2b_a0_6b/`：

```markdown
# stage_name report

## Config

- model:
- tokenizer:
- max_seq_len:
- data:
- lr:
- batch:
- accumulation:

## Training

- processed tokens:
- tokens/s:
- loss start:
- loss end:
- aux_loss:
- max GPU memory:

## Evaluation

- held-out loss:
- generation samples:
- patch valid rate:
- apply success rate:
- tests pass rate:
- long-context result:

## Decision

- [ ] promote to next stage
- [ ] repeat same stage
- [ ] change data
- [ ] change hyperparameters
```

## 8. 停止条件

遇到下面情况不要继续放大：

```text
loss 不降或周期性爆炸。
MoE aux_loss 异常大。
64K 训练不能引用远距离信息。
patch 格式有效率低于 50%。
vLLM 不能稳定加载 checkpoint。
训练数据 audit 不能证明 SWE-bench 无污染。
```

处理方式：

```text
回到上一个阶段。
缩小数据和长度。
复现实验。
只改一个变量。
写报告。
```

这就是 Mini-GLM 主线 runbook 的最后一章。
