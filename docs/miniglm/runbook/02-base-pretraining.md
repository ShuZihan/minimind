# 02. Base Pretraining

[返回执行手册](../02-training-runbook.md)

Base pretraining 是让模型先学会语言、代码和常见工程文本的分布。这个阶段不要急着教它“修 bug”，先让它熟悉代码语法、API 习惯、Markdown、diff、traceback 和测试日志。

建议先跑短序列，因为短序列吞吐高，能快速验证数据质量和训练稳定性。

先回到仓库根目录并检查上一阶段产物：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
test -f tokenizer/miniglm-32k/tokenizer.json
test -s data/miniglm/pretrain_base/train_100m.jsonl
test -e data/miniglm/pretrain_base/train.jsonl
```

如果这里失败，先回到 [01-tokenizer-and-data.md](01-tokenizer-and-data.md)，不要直接跑训练命令。

## 1. 长度与预算

```text
stage1_base_2k:
  max_seq_len = 2048
  token budget = 1B-3B for learning run

stage1_base_4k:
  max_seq_len = 4096
  token budget = 1B-5B
```

## 2. 用 100M tokens 标定吞吐

```bash
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

这条命令用于标定吞吐，不是追求能力。`train_100m.jsonl` 足够小，可以快速估算 tokens/s；`--max_seq_len 2048` 先学习局部模式；`--use_wandb` 用来记录 loss、吞吐和显存曲线。

计算每次优化器更新看过多少 token：

```text
processed_tokens_per_optimizer_step =
  nproc_per_node * batch_size * accumulation_steps * max_seq_len

8 * 1 * 16 * 2048 = 262144 tokens / optimizer step
```

用总 token 预算除以实际 tokens/s，就能得到大致训练时长。

## 3. 正式训练 2K

```bash
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

这条命令把数据从 100M 扩到正式训练集。结构参数保持不变，目的是只放大数据量，不同时改变模型形状。

## 4. 继续训练 4K

```bash
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

这里从 `stage1_base_2k` 继续，把训练长度从 2K 提到 4K，并略降学习率。`--from_weight` 表示加载上一阶段权重，不是随机重训。

## 5. 过关标准

```text
loss 稳定下降。
没有 nan/inf。
aux_loss 没有异常放大。
checkpoint 可恢复。
held-out code loss 下降。
简单生成不乱码。
```
