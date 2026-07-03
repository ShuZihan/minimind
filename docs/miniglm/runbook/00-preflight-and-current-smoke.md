# 00. 环境准备与当前 MiniMind Smoke

[返回目录](../README.md)

这一章只验证当前仓库已经具备的能力：环境能用、依赖能装、官方数据能下载、当前 MiniMind 训练链路能启动、checkpoint 能保存。这里还不训练 Mini-GLM。

除非特别说明，命令都从仓库根目录执行：

```bash
cd "$(git rev-parse --show-toplevel)"
export MINIMIND_ROOT="$(pwd)"
```

## 1. 准备目录

```bash
cd "$MINIMIND_ROOT"
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

这些目录把实验材料分开：`data/` 放数据，`tokenizer/` 放 32K 词表，`out/` 放训练中的 `.pth`，`hf/` 放导出的 Hugging Face checkpoint，`reports/` 放阶段记录。

## 2. 记录环境并安装依赖

```bash
cd "$MINIMIND_ROOT"
pwd
git status --short
python3 --version
nvidia-smi
python3 -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

确认关键依赖能导入：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
import torch
import datasets
import transformers
import modelscope

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("datasets", datasets.__version__)
print("transformers", transformers.__version__)
print("modelscope", modelscope.__version__)
PY
```

如果 `torch.cuda.is_available()` 是 `False`，先处理容器 GPU 挂载、CUDA 驱动或 PyTorch CUDA 版本。不要继续跑 8 卡训练。

## 3. 下载官方数据并生成 smoke 数据

MiniMind 官方数据不在仓库里。第一轮只需要 `pretrain_t2t_mini.jsonl`。

```bash
cd "$MINIMIND_ROOT"
modelscope download \
  --dataset gongjy/minimind_dataset \
  pretrain_t2t_mini.jsonl \
  --local_dir ./dataset
```

如果 `modelscope` 命令不存在：

```bash
python3 -m pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
```

也可以从浏览器下载：

```text
ModelScope:
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files

Hugging Face:
https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main
```

检查文件：

```bash
cd "$MINIMIND_ROOT"
ls -lh dataset/pretrain_t2t_mini.jsonl
head -n 2 dataset/pretrain_t2t_mini.jsonl
```

生成 10MB smoke 数据：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
dst = Path("data/miniglm/pretrain_base/smoke_10m.jsonl")
target_bytes = 10 * 1024 * 1024

if not src.exists():
    raise SystemExit(f"missing {src}")

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

ls -lh data/miniglm/pretrain_base/smoke_10m.jsonl
head -n 2 data/miniglm/pretrain_base/smoke_10m.jsonl
```

## 4. 确认当前训练脚本状态

当前仓库还不能直接训练 Mini-GLM。先看 `train_pretrain.py` 真实支持哪些参数：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_pretrain.py -h
```

此时应该只能看到当前 MiniMind 路径已有的结构参数：

```text
--hidden_size
--num_hidden_layers
--max_seq_len
--use_moe
```

如果你已经能看到 `--model-name`、`--tokenizer-path`、`--vocab-size`、`--n-routed-experts` 等参数，说明工程改造已经做过，可以继续按下一章验证。

## 5. 跑当前 MiniMind Smoke

当前脚本默认 tokenizer 路径是 `../model`，所以这里从 `trainer/` 目录执行：

```bash
cd "$MINIMIND_ROOT/trainer"
torchrun --nproc_per_node 8 train_pretrain.py \
  --data_path ../data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir ../out/miniglm_2b_a0_6b \
  --save_weight stage0_current_minimind_smoke \
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

这条命令只验证当前代码链路，不代表 Mini-GLM 已经实现。它通过时，你应该看到有限 loss，并在 `out/miniglm_2b_a0_6b/` 下看到 `stage0_current_minimind_smoke_1536_moe.pth` 一类文件。

## 6. 过关标准

```text
依赖能导入。
数据文件存在。
torchrun 能启动 8 个进程。
loss 是有限数值，不是 nan/inf。
out/miniglm_2b_a0_6b/ 下有 smoke checkpoint。
```

下一步：打开 [01-engineering-migration.md](01-engineering-migration.md)。
