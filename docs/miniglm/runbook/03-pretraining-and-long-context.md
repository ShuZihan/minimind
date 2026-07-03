# 03. Base Pretraining 与长上下文训练

[返回目录](../README.md)

这一章从 Mini-GLM target smoke 进入正式预训练。顺序是：先用 100M tokens 标定吞吐，再做 2K/4K base pretraining，然后逐步进入 8K/16K/32K/64K repo-level 长上下文。

先回到仓库根目录：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
test -f tokenizer/miniglm-32k/tokenizer.json
test -s data/miniglm/pretrain_base/train_100m.jsonl
test -e data/miniglm/pretrain_base/train.jsonl
```

## 1. 100M tokens 吞吐标定

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/pretrain_base/train_100m.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage1_base_2k_100m \
  --vocab-size 32768 \
  --max-position-embeddings 65536 \
  --hidden-size 1536 \
  --num-hidden-layers 30 \
  --first-k-dense-replace 3 \
  --n-routed-experts 16 \
  --n-shared-experts 1 \
  --num-experts-per-tok 2 \
  --moe-intermediate-size 768 \
  --dense-intermediate-size 4096 \
  --max_seq_len 2048 \
  --batch_size 1 \
  --accumulation_steps 16 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --save_interval 500 \
  --log_interval 20 \
  --dtype bfloat16 \
  --use_wandb
```

这条命令不追求能力，只估算吞吐、显存和 loss 稳定性。

## 2. Base pretraining 2K

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/pretrain_base/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage1_base_2k \
  --vocab-size 32768 \
  --max-position-embeddings 65536 \
  --hidden-size 1536 \
  --num-hidden-layers 30 \
  --first-k-dense-replace 3 \
  --n-routed-experts 16 \
  --n-shared-experts 1 \
  --num-experts-per-tok 2 \
  --moe-intermediate-size 768 \
  --dense-intermediate-size 4096 \
  --max_seq_len 2048 \
  --batch_size 1 \
  --accumulation_steps 16 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --save_interval 1000 \
  --log_interval 50 \
  --dtype bfloat16 \
  --use_wandb
```

## 3. Base pretraining 4K

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/pretrain_base/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage1_base_4k \
  --from_weight stage1_base_2k \
  --vocab-size 32768 \
  --max-position-embeddings 65536 \
  --hidden-size 1536 \
  --num-hidden-layers 30 \
  --first-k-dense-replace 3 \
  --n-routed-experts 16 \
  --n-shared-experts 1 \
  --num-experts-per-tok 2 \
  --moe-intermediate-size 768 \
  --dense-intermediate-size 4096 \
  --max_seq_len 4096 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --epochs 1 \
  --learning_rate 1.5e-4 \
  --save_interval 1000 \
  --log_interval 50 \
  --dtype bfloat16 \
  --use_wandb
```

`--from_weight stage1_base_2k` 表示从上一阶段继续，不是随机重训。

## 4. 准备 repo-level 长上下文数据

如果只是做流程 smoke，可以先用 base 数据链接成 repo context 输入：

```bash
cd "$MINIMIND_ROOT"
mkdir -p data/miniglm/repo_context
ln -sfn ../pretrain_base/train.jsonl data/miniglm/repo_context/train.jsonl
ln -sfn ../pretrain_base/train.jsonl data/miniglm/repo_context/train_long_32k.jsonl
ln -sfn ../pretrain_base/train.jsonl data/miniglm/repo_context/train_long_64k.jsonl
```

这只是工程 smoke。正式长上下文训练应替换为真实 repo tree、多文件源码、traceback、issue、测试日志和 agent trajectory 组合出的长样本。

统计长度分布：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer/miniglm-32k")
path = Path("data/miniglm/repo_context/train.jsonl")
buckets = {2048: 0, 4096: 0, 8192: 0, 16384: 0, 32768: 0, 65536: 0, "gt_65536": 0}

for line in path.open("r", encoding="utf-8"):
    obj = json.loads(line)
    text = obj.get("text") or obj.get("prompt") or json.dumps(obj, ensure_ascii=False)
    n = len(tok.encode(text))
    for b in [2048, 4096, 8192, 16384, 32768, 65536]:
        if n <= b:
            buckets[b] += 1
            break
    else:
        buckets["gt_65536"] += 1

print(buckets)
PY
```

## 5. Repo-level 8K

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/repo_context/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage2_repo_8k \
  --from_weight stage1_base_4k \
  --vocab-size 32768 \
  --max-position-embeddings 65536 \
  --hidden-size 1536 \
  --num-hidden-layers 30 \
  --first-k-dense-replace 3 \
  --n-routed-experts 16 \
  --n-shared-experts 1 \
  --num-experts-per-tok 2 \
  --moe-intermediate-size 768 \
  --dense-intermediate-size 4096 \
  --max_seq_len 8192 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --epochs 1 \
  --learning_rate 1e-4 \
  --save_interval 1000 \
  --log_interval 20 \
  --dtype bfloat16 \
  --use_wandb
```

## 6. Repo-level 16K、32K、64K

16K：

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/repo_context/train.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage2_repo_16k \
  --from_weight stage2_repo_8k \
  --max_seq_len 16384 \
  --batch_size 1 \
  --accumulation_steps 2 \
  --learning_rate 7e-5 \
  --epochs 1 \
  --dtype bfloat16
```

32K：

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/repo_context/train_long_32k.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage2_repo_32k \
  --from_weight stage2_repo_16k \
  --max_seq_len 32768 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --learning_rate 5e-5 \
  --epochs 1 \
  --dtype bfloat16
```

64K：

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/repo_context/train_long_64k.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage2_repo_64k \
  --from_weight stage2_repo_32k \
  --max_seq_len 65536 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --learning_rate 3e-5 \
  --epochs 1 \
  --dtype bfloat16
```

16K/32K/64K 命令省略了结构参数。如果你的训练脚本要求完整参数，就把 8K 命令中的结构参数一并带上。

## 7. 过关标准

```text
100M 标定能给出稳定 tokens/s。
2K/4K loss 正常下降。
8K/16K 不 OOM。
32K/64K 小规模能跑通。
长上下文评估能引用前文信息。
```

下一步：打开 [04-sft-export-vllm-and-reports.md](04-sft-export-vllm-and-reports.md)。
