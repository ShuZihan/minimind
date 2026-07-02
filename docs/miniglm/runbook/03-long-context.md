# 03. Repo-level 长上下文训练

[返回执行手册](../02-training-runbook.md)

长上下文不是把 `max_seq_len` 改大就自然获得的能力。模型需要在训练中反复遇到“前文信息会在后文被使用”的样本，才会学会跨文件、跨日志、跨多轮工具调用定位信息。

这一阶段从 8K 开始，逐步走到 64K。

## 1. 长度课程

```text
8K:
  文件级 + 小 repo context。

16K:
  多文件 + tests + README。

32K:
  长 issue + traceback + 多文件源码。

64K:
  少量真实远距离依赖样本。
```

64K 样本不要随机拼接，至少要有一种远距离依赖：

```text
开头 API 在末尾被测试。
开头 issue 对应末尾某个文件 bug。
traceback 对应后半段源码。
repo tree 决定后半段需要读哪个文件。
agent trajectory 跨多轮搜索、阅读、编辑、测试。
```

## 2. 统计长度分布

```bash
python - <<'PY'
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

这段脚本统计 tokenized length 分桶。长上下文训练前要确认 8K、16K、32K、64K 样本真的存在；否则 `max_seq_len` 变大只会制造 padding 和显存浪费。

## 3. 训练 8K

```bash
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

8K 是 repo-level 长上下文起点。这里从 `stage1_base_4k` 继续，先让模型看到文件树、多文件源码和测试失败上下文；`accumulation_steps` 降低，是因为序列变长后单步显存和计算都会上升。

## 4. 训练 16K

```bash
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

16K 继续放大上下文，并进一步降低学习率。序列越长，每一步越贵，训练也更容易不稳定，所以这里不急着加大 batch。

## 5. 训练 32K

```bash
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

32K 需要专门的 `train_long_32k.jsonl`。这批数据应包含真实多文件依赖、长 issue、长 traceback 或多轮 agent trajectory，而不是随便拼接短文本。

## 6. 训练 64K

```bash
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

64K 是强化阶段。目标不是让模型多读一点长文本，而是让它学会引用远距离信息；如果评估只能流畅续写，却找不到前文证据，说明数据没有教会长上下文。

16K/32K/64K 命令省略了结构参数。若脚本要求完整参数，就把 8K 命令中的结构参数一并带上。

## 7. 过关标准

```text
8K/16K loss 正常下降。
32K 不 OOM。
64K 小规模能跑通。
长上下文评估能引用前文。
```

长上下文评估至少做：

```text
Needle:
  在 64K 前 10% 放唯一事实，末尾提问。

Repo lookup:
  前面放 repo tree 和多个文件，末尾问测试失败应看哪个文件。

Patch location:
  前面放 traceback，后面放源码，要求指出最小修改位置。
```
