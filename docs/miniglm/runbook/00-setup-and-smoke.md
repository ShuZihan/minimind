# 00. 环境准备与最小链路验证

[返回执行手册](../02-training-runbook.md)

训练一个模型，第一天不要急着追 loss。更重要的是证明链路能闭合：模型能初始化，数据能读进来，8 卡能启动，checkpoint 能保存和恢复，最后还能导出给 vLLM。

这一阶段只做工程验证，不评价模型能力。

除非特别注明，文档里的路径都以仓库根目录为起点。当前 MiniMind 官方训练脚本是例外：它默认从 `trainer/` 目录执行，因为 tokenizer 默认路径是 `../model`。

## 1. 准备目录

```bash
mkdir -p \
  data/miniglm/{raw,tokenizer_corpus,pretrain_base,repo_context,sft_issue_patch,sft_agent_trajectory,preference_pairs,audit} \
  tokenizer/miniglm-32k \
  configs \
  runs/miniglm_2b_a0_6b/{stage0_smoke,stage1_base_2k,stage1_base_4k,stage2_repo_8k,stage2_repo_16k,stage2_repo_32k,stage2_repo_64k,stage3_patch_sft,stage4_agent_sft,stage5_preference} \
  out/miniglm_2b_a0_6b \
  hf/mini-glm \
  reports/miniglm_2b_a0_6b

touch runs/miniglm_2b_a0_6b/learning_log.md
```

这条命令先把训练空间分清楚。`data/miniglm/` 放数据和审计记录，`tokenizer/miniglm-32k/` 放自训词表，`out/miniglm_2b_a0_6b/` 放训练中的 `.pth`，`hf/mini-glm/` 放导出的 Hugging Face 格式，`reports/miniglm_2b_a0_6b/` 放阶段报告。

## 2. 记录环境

```bash
pwd
git status --short
python --version
nvidia-smi
```

这组命令是在给实验留底。路径、git 状态、Python 版本、GPU、驱动和 CUDA 信息都会影响复现；后面 loss 或吞吐异常时，先对照这里。

## 3. 下载官方数据并生成 smoke 数据

MiniMind 官方数据不在仓库里，需要单独下载。官方 README 推荐从 ModelScope 或 Hugging Face 下载，并把数据放到 `./dataset/`。第一轮只需要 `pretrain_t2t_mini.jsonl`，它是预训练 smoke test 的来源。

在仓库根目录执行：

```bash
modelscope download \
  --dataset gongjy/minimind_dataset \
  pretrain_t2t_mini.jsonl \
  --local_dir ./dataset
```

如果 ModelScope CLI 不可用，先安装：

```bash
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
```

也可以从浏览器下载：

```text
ModelScope:
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files

Hugging Face:
https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main
```

下载后检查文件：

```bash
ls -lh dataset/pretrain_t2t_mini.jsonl
head -n 2 dataset/pretrain_t2t_mini.jsonl
```

`smoke_10m.jsonl` 不是官方文件，它是从 `pretrain_t2t_mini.jsonl` 抽样出来的小训练集，只用于快速验证链路。生成方式如下：

```bash
python - <<'PY'
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
dst = Path("data/miniglm/pretrain_base/smoke_10m.jsonl")
target_bytes = 10 * 1024 * 1024

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

检查 smoke 数据：

```bash
ls -lh data/miniglm/pretrain_base/smoke_10m.jsonl
head -n 2 data/miniglm/pretrain_base/smoke_10m.jsonl
```

这一步只是从官方预训练数据里切出一个 10MB 小样本。后续真正训练时，应回到完整 `pretrain_t2t_mini.jsonl`、`pretrain_t2t.jsonl`，或你为 Mini-GLM 准备的新数据集。

建议每次实验都追加一段日志：

```markdown
## stage-name

- config:
  - vocab_size:
  - max_position_embeddings:
  - max_seq_len:
  - global_batch_tokens:
  - learning_rate:
- data:
  - source:
  - tokens:
  - split:
- result:
  - loss start:
  - loss end:
  - tokens/s:
  - max GPU memory:
  - checkpoint:
- decision:
  - continue / repeat / change data / change hyperparameters
```

## 4. 训练前需要补齐的工程接口

Mini-GLM 不是简单改几个超参数就能训练。它要有自己的模型工厂、模型实现、训练参数和导出脚本。

```text
模型工厂:
  build_model(config)
  支持 model_type = mini_glm / glm_moe_dsa

模型实现:
  model/model_miniglm.py
  MiniGLMConfig
  MiniGLMForCausalLM
  DSA/MLA attention
  routed experts
  shared expert

训练参数:
  --model-name mini-glm
  --tokenizer-path tokenizer/miniglm-32k
  --vocab-size 32768
  --max-position-embeddings 65536
  --hidden-size 1536
  --num-hidden-layers 30
  --first-k-dense-replace 3
  --n-routed-experts 16
  --n-shared-experts 1
  --num-experts-per-tok 2
  --moe-intermediate-size 768
  --dense-intermediate-size 4096

数据管线:
  sequence packing
  attention mask / sample boundary
  长上下文长度统计

导出:
  scripts/export_miniglm_hf.py
  .pth -> safetensors
  glm_moe_dsa-compatible config.json
```

## 5. 跑当前仓库可执行的 smoke test

当前仓库的 `trainer/train_pretrain.py` 还不支持 `--model-name`、`--tokenizer-path`、`--vocab-size`、`--max-position-embeddings`、`--n-routed-experts` 这些 Mini-GLM 参数。先用它已经支持的 MiniMind 参数验证容器、数据、DDP 和 checkpoint。

从仓库根目录进入 `trainer/`：

```bash
cd /path/to/minimind/trainer
```

然后执行：

```bash
torchrun --nproc_per_node 8 train_pretrain.py \
  --data_path ../data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir ../out/miniglm_2b_a0_6b \
  --save_weight stage0_smoke \
  --hidden_size 1536 \
  --num_hidden_layers 30 \
  --max_seq_len 2048 \
  --use_moe 1 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --save_interval 50 \
  --log_interval 10 \
  --dtype bfloat16
```

这条命令怎么读：

```text
torchrun --nproc_per_node 8:
  用 8 张卡启动当前 MiniMind 训练脚本。

--data_path:
  smoke test 数据。每行需要有 text 字段。

--save_dir / --save_weight:
  checkpoint 输出目录和文件名前缀。实际文件名会带 hidden_size 和 moe 后缀。

--hidden_size / --num_hidden_layers:
  当前 MiniMind 脚本已经支持的模型规模参数。

--use_moe 1:
  使用当前 MiniMind 的 MoE 实现。注意这不是 Mini-GLM 的 16E/top-2 设计。

--max_seq_len:
  本次真实训练长度。
```

这一条命令只验证当前代码链路，不代表 Mini-GLM-5.1-2.1B-A0.6B 已经实现。

## 6. Mini-GLM 工程改造后的目标命令

完成模型工厂、`model/model_miniglm.py`、结构参数 CLI 和导出脚本后，再运行下面这类命令。

```bash
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage0_smoke \
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
  --accumulation_steps 8 \
  --epochs 1 \
  --learning_rate 2e-4 \
  --save_interval 50 \
  --log_interval 10 \
  --dtype bfloat16
```

这条命令怎么读：

```text
torchrun --nproc_per_node 8:
  用 8 张卡启动分布式训练。

--model-name / --tokenizer-path:
  选择 Mini-GLM 实现和 32K tokenizer。当前脚本尚未支持。

--data_path:
  用 10M 级小数据做 smoke test，只验证工程链路。

--save_dir / --save_weight:
  指定 checkpoint 输出目录和文件名前缀。

--vocab-size 到 --dense-intermediate-size:
  定义 Mini-GLM 模型形状。当前脚本尚未支持。

--max_seq_len 2048:
  本次真实训练长度。它不是 64K 上限，只是先用短序列验证链路。

--batch_size 1 / --accumulation_steps 8:
  每张卡一次放 1 条样本，用梯度累积凑更大的有效 batch。

--learning_rate / --save_interval / --log_interval:
  控制更新步长、保存频率和日志频率。
```

## 7. 过关标准

```text
DDP 能启动。
loss 是有限数值，不是 nan/inf。
loss 有下降趋势。
显存不 OOM。
out/miniglm_2b_a0_6b/stage0_smoke_*.pth 或 stage0_smoke_1536_moe.pth 能生成。
resume checkpoint 能恢复。
短训 checkpoint 能导出 HF。
vLLM 能加载导出目录。
```

常见失败优先这样排查：

| 现象 | 优先排查 |
|---|---|
| OOM | batch_size、max_seq_len、activation checkpoint、模型参数 |
| loss=nan | dtype、lr、初始化、router aux loss |
| DDP 卡住 | NCCL、CUDA_VISIBLE_DEVICES、torchrun 参数 |
| checkpoint 不能恢复 | state_dict key、模型 config 是否一致 |
| vLLM 不能加载 | config.json、权重 key、tokenizer、Transformers/vLLM 版本 |
