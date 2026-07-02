# 00. 环境准备与最小链路验证

[返回执行手册](../02-training-runbook.md)

训练一个模型，第一天不要急着追 loss。更重要的是证明链路能闭合：模型能初始化，数据能读进来，8 卡能启动，checkpoint 能保存和恢复，最后还能导出给 vLLM。

这一阶段只做工程验证，不评价模型能力。

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

## 3. 训练前需要补齐的工程接口

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

## 4. 跑最小链路

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
  选择 Mini-GLM 实现和 32K tokenizer。

--data_path:
  用 10M 级小数据做 smoke test，只验证工程链路。

--save_dir / --save_weight:
  指定 checkpoint 输出目录和文件名前缀。

--vocab-size 到 --dense-intermediate-size:
  定义模型形状。恢复 checkpoint、导出 HF、vLLM 加载时必须保持一致。

--max_seq_len 2048:
  本次真实训练长度。它不是 64K 上限，只是先用短序列验证链路。

--batch_size 1 / --accumulation_steps 8:
  每张卡一次放 1 条样本，用梯度累积凑更大的有效 batch。

--learning_rate / --save_interval / --log_interval:
  控制更新步长、保存频率和日志频率。
```

## 5. 过关标准

```text
DDP 能启动。
loss 是有限数值，不是 nan/inf。
loss 有下降趋势。
显存不 OOM。
out/miniglm_2b_a0_6b/stage0_smoke_*.pth 能生成。
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
