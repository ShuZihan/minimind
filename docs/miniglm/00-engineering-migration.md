# Mini-GLM 工程化改造手册

[返回目录](README.md)

这份文档先把话说透：当前仓库可以训练 MiniMind，但还不能直接训练 Mini-GLM-5.1-2.1B-A0.6B。训练 runbook 里的 Mini-GLM 命令是目标态命令，必须完成本文件的工程化改造后再执行。

本文只做一件事：把当前 MiniMind 仓库一步步改造成能识别、训练、保存、导出 Mini-GLM 的工程结构。

除非某一节特别说明，本文所有 shell 命令都从仓库根目录执行。先设置一个根目录变量，后面不要再手写临时路径：

```bash
cd "$(git rev-parse --show-toplevel)"
export MINIMIND_ROOT="$(pwd)"
```

如果你只想知道“今天到底按什么顺序做”，就是下面这条线：

```text
1. 设置 MINIMIND_ROOT。
2. 安装 requirements.txt。
3. 下载 MiniMind 官方预训练数据，生成 smoke_10m.jsonl。
4. 跑当前 MiniMind smoke test，确认容器、数据、DDP 和 checkpoint 没问题。
5. 按本文第 2-9 节改工程接口，每改完一节就跑该节验证命令。
6. 完成 runbook/01-tokenizer-and-data.md，训练 tokenizer/miniglm-32k，并生成 train_100m.jsonl / train.jsonl。
7. 全部验证通过后，再跑 Mini-GLM 目标 smoke 训练。
8. 目标 smoke 通过后，再进入正式训练 runbook。
```

## 0. 命令可执行性约定

本文里的命令分三类：

```text
当前可执行:
  不改代码，只要依赖和数据准备好就能跑。

阶段验证命令:
  完成当前小节描述的代码修改后执行。

目标态命令:
  只有全部工程改造完成后才执行。
```

不要跳步。尤其不要在 `model/model_miniglm.py`、模型工厂、训练参数和导出脚本完成之前运行 Mini-GLM 训练命令。

## 1. 先确认当前仓库基线

目标：证明问题不在容器、CUDA、DDP 或数据读取，而在 Mini-GLM 工程接口尚未实现。

在仓库根目录执行：

```bash
cd "$MINIMIND_ROOT"
git status --short
python3 --version
nvidia-smi
```

如果依赖还没有安装，先安装 MiniMind 官方依赖：

```bash
python3 -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

先完成 [runbook/00-setup-and-smoke.md](runbook/00-setup-and-smoke.md) 中的数据准备，确保 `data/miniglm/pretrain_base/smoke_10m.jsonl` 已经存在。

查看当前 `train_pretrain.py` 支持哪些参数：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_pretrain.py -h
```

你应该只能看到这些和模型结构相关的参数：

```text
--hidden_size
--num_hidden_layers
--max_seq_len
--use_moe
```

如果你看到下面这些参数，说明你已经做过后续改造：

```text
--model-name
--tokenizer-path
--vocab-size
--max-position-embeddings
--first-k-dense-replace
--n-routed-experts
--n-shared-experts
--num-experts-per-tok
--moe-intermediate-size
--dense-intermediate-size
```

当前仓库可执行的 smoke test 见 [runbook/00-setup-and-smoke.md](runbook/00-setup-and-smoke.md)。先跑当前 MiniMind smoke test，不要直接跑 Mini-GLM 目标命令。

## 2. 第一步：把 tokenizer 路径显式参数化

当前 `trainer/trainer_utils.py` 的 `init_model` 已经支持 `tokenizer_path` 参数，但 `trainer/train_pretrain.py` 没有把 CLI 参数传进去。第一步只改这一件事。

修改文件：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_dpo.py
```

在 argparse 中加入：

```python
parser.add_argument(
    "--tokenizer-path", "--tokenizer_path",
    dest="tokenizer_path",
    type=str,
    default="../model",
    help="tokenizer 路径"
)
```

把原来的：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
```

改成：

```python
model, tokenizer = init_model(
    lm_config,
    args.from_weight,
    tokenizer_path=args.tokenizer_path,
    save_dir=args.save_dir,
    device=args.device,
)
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_pretrain.py -h | grep tokenizer
python3 trainer/train_full_sft.py -h | grep tokenizer
python3 trainer/train_dpo.py -h | grep tokenizer
```

通过标准：

```text
三个脚本都能看到 --tokenizer-path / --tokenizer_path。
没有 ImportError。
```

## 3. 第二步：把 vocab 和 position 显式参数化

MiniMindConfig 本身已经能接收 `vocab_size` 和 `max_position_embeddings`，但训练脚本没有暴露 CLI。先让 MiniMind 路径支持这两个参数，后面 Mini-GLM 才能共用。

修改文件：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_dpo.py
```

在 argparse 中加入 vocab 和 position 参数：

```python
parser.add_argument("--vocab-size", "--vocab_size", dest="vocab_size", type=int, default=6400, help="词表大小")
parser.add_argument("--max-position-embeddings", "--max_position_embeddings", dest="max_position_embeddings", type=int, default=32768, help="最大位置编码长度")
```

同时把已有的 `hidden_size`、`num_hidden_layers` 改成横线/下划线双别名。否则目标命令里的 `--hidden-size` 和 `--num-hidden-layers` 仍然会报 `unrecognized arguments`：

```python
parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=512, help="hidden size")
parser.add_argument("--num-hidden-layers", "--num_hidden_layers", dest="num_hidden_layers", type=int, default=8, help="number of hidden layers")
```

注意：这是替换原来的两行 `--hidden_size`、`--num_hidden_layers`，不要重复添加两个同名 `dest` 的参数。

把原来的：

```python
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe),
)
```

改成：

```python
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe),
    vocab_size=args.vocab_size,
    max_position_embeddings=args.max_position_embeddings,
)
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_pretrain.py -h | grep vocab
python3 trainer/train_pretrain.py -h | grep max-position
python3 trainer/train_pretrain.py -h | grep hidden-size
```

再跑一个最小单进程解析测试：

```bash
cd "$MINIMIND_ROOT/trainer"
python3 train_pretrain.py \
  --data_path ../data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir ../out/miniglm_2b_a0_6b \
  --save_weight cli_check \
  --tokenizer-path ../model \
  --vocab-size 6400 \
  --max-position-embeddings 32768 \
  --hidden-size 128 \
  --num-hidden-layers 2 \
  --max_seq_len 64 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --epochs 1 \
  --save_interval 1 \
  --log_interval 1
```

这条命令会真的开始训练一个极小模型。如果只想验证参数解析，先按 `Ctrl+C` 停掉；如果数据只有 10MB，也可以让它跑完。

## 4. 第三步：把 tokenizer 训练脚本参数化

当前 `trainer/train_tokenizer.py` 是学习脚本，默认使用硬编码常量：

```text
DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400
```

Mini-GLM 需要 32K tokenizer，所以这里必须先把 tokenizer 训练脚本改成 CLI。否则 [runbook/01-tokenizer-and-data.md](runbook/01-tokenizer-and-data.md) 里的命令会直接报 `unrecognized arguments`。

修改文件：

```text
trainer/train_tokenizer.py
```

先让 `get_texts` 同时支持预训练 JSONL 的 `text` 字段和 SFT JSONL 的 `conversations` 字段：

```python
def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("text"):
                yield data["text"]
                continue

            contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
            if contents:
                yield "\n".join(contents)
```

再把文件底部的：

```python
if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
```

改成：

```python
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train MiniMind/Mini-GLM tokenizer")
    parser.add_argument("--input", type=str, default=DATA_PATH, help="训练 tokenizer 的 JSONL 文件")
    parser.add_argument("--vocab-size", "--vocab_size", dest="vocab_size", type=int, default=VOCAB_SIZE, help="词表大小")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=TOKENIZER_DIR, help="tokenizer 输出目录")
    args = parser.parse_args()

    train_tokenizer(args.input, args.output_dir, args.vocab_size)
    eval_tokenizer(args.output_dir)
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_tokenizer.py -h | grep -E 'input|vocab-size|output-dir'
```

通过标准：

```text
能看到 --input、--vocab-size、--output-dir。
不会出现 unrecognized arguments。
```

## 5. 第四步：引入模型工厂

现在训练脚本硬编码 `MiniMindConfig` 和 `MiniMindForCausalLM`。要支持 Mini-GLM，先把“训练脚本选择哪个模型”抽成工厂。

新建文件：

```text
model/model_factory.py
```

内容：

```python
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def build_config(args):
    model_name = getattr(args, "model_name", "minimind")

    if model_name == "minimind":
        return MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
        )

    if model_name == "mini-glm":
        from model.model_miniglm import MiniGLMConfig

        return MiniGLMConfig.from_args(args)

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model(config):
    model_type = getattr(config, "model_type", None)

    if model_type == "minimind":
        return MiniMindForCausalLM(config)

    if model_type == "glm_moe_dsa":
        from model.model_miniglm import MiniGLMForCausalLM

        return MiniGLMForCausalLM(config)

    raise ValueError(f"Unsupported model_type: {model_type}")
```

修改 `trainer/trainer_utils.py`：

把：

```python
from model.model_minimind import MiniMindForCausalLM
```

替换为：

```python
from model.model_factory import build_model
```

把 `init_model` 里的：

```python
model = MiniMindForCausalLM(lm_config)
```

替换为：

```python
model = build_model(lm_config)
```

修改训练脚本：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_dpo.py
```

把 `MiniMindConfig` 的 import 改成：

```python
from model.model_factory import build_config
```

在 argparse 中加入：

```python
parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default="minimind", choices=["minimind", "mini-glm"], help="模型实现名称")
```

把构造 config 的代码改成：

```python
lm_config = build_config(args)
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from argparse import Namespace
from model.model_factory import build_config, build_model

args = Namespace(
    model_name="minimind",
    hidden_size=128,
    num_hidden_layers=2,
    use_moe=0,
    vocab_size=6400,
    max_position_embeddings=32768,
)

cfg = build_config(args)
model = build_model(cfg)
print(cfg.model_type, type(model).__name__)
PY
```

通过标准：

```text
输出 minimind MiniMindForCausalLM。
```

## 6. 第五步：先加 Mini-GLM 配置，不急着训练

先创建 Mini-GLM 的 config，让模型规格有一个清晰入口。此时可以只实现配置类，不要急着跑训练。

新建文件：

```text
model/model_miniglm.py
```

先写配置骨架：

```python
from transformers import PretrainedConfig


class MiniGLMConfig(PretrainedConfig):
    model_type = "glm_moe_dsa"

    def __init__(
        self,
        vocab_size=32768,
        max_position_embeddings=65536,
        hidden_size=1536,
        num_hidden_layers=30,
        first_k_dense_replace=3,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=768,
        dense_intermediate_size=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.tie_word_embeddings = False

    @classmethod
    def from_args(cls, args):
        return cls(
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_position_embeddings,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            first_k_dense_replace=args.first_k_dense_replace,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            moe_intermediate_size=args.moe_intermediate_size,
            dense_intermediate_size=args.dense_intermediate_size,
        )
```

同时在训练脚本 argparse 中加入 Mini-GLM 结构参数：

```python
parser.add_argument("--first-k-dense-replace", "--first_k_dense_replace", dest="first_k_dense_replace", type=int, default=3)
parser.add_argument("--n-routed-experts", "--n_routed_experts", dest="n_routed_experts", type=int, default=16)
parser.add_argument("--n-shared-experts", "--n_shared_experts", dest="n_shared_experts", type=int, default=1)
parser.add_argument("--num-experts-per-tok", "--num_experts_per_tok", dest="num_experts_per_tok", type=int, default=2)
parser.add_argument("--moe-intermediate-size", "--moe_intermediate_size", dest="moe_intermediate_size", type=int, default=768)
parser.add_argument("--dense-intermediate-size", "--dense_intermediate_size", dest="dense_intermediate_size", type=int, default=4096)
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from argparse import Namespace
from model.model_factory import build_config

args = Namespace(
    model_name="mini-glm",
    vocab_size=32768,
    max_position_embeddings=65536,
    hidden_size=1536,
    num_hidden_layers=30,
    first_k_dense_replace=3,
    n_routed_experts=16,
    n_shared_experts=1,
    num_experts_per_tok=2,
    moe_intermediate_size=768,
    dense_intermediate_size=4096,
)

cfg = build_config(args)
print(cfg.model_type)
print(cfg.vocab_size, cfg.hidden_size, cfg.n_routed_experts, cfg.num_experts_per_tok)
PY
```

通过标准：

```text
输出 glm_moe_dsa。
输出 32768 1536 16 2。
```

## 7. 第六步：实现 MiniGLMForCausalLM

这一步才是真正的模型实现。不要用空壳类伪装完成，否则训练能启动但不是 Mini-GLM。

需要实现的模块：

```text
MiniGLMForCausalLM
MiniGLMModel
MiniGLMDecoderLayer
DSA/MLA-style attention
dense MLP
routed experts
shared expert
router auxiliary loss
RMSNorm
RoPE / YaRN 兼容
```

建议迁移顺序：

```text
1. 先复用 MiniMind 的 RMSNorm、RoPE、CausalLM 输出结构。
2. 先实现 dense layer，跑通 tiny forward。
3. 再实现 MoE layer，跑通 tiny forward。
4. 最后加入 DSA/MLA attention，不要一开始把 attention 和 MoE 同时改。
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
import torch
from model.model_miniglm import MiniGLMConfig, MiniGLMForCausalLM

cfg = MiniGLMConfig(
    vocab_size=128,
    max_position_embeddings=256,
    hidden_size=64,
    num_hidden_layers=2,
    first_k_dense_replace=1,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    dense_intermediate_size=128,
)

model = MiniGLMForCausalLM(cfg)
input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
labels = input_ids.clone()
out = model(input_ids, labels=labels)
print(out.loss.item())
print(out.logits.shape)
PY
```

通过标准：

```text
loss 是有限数值。
logits shape 是 torch.Size([2, 16, 128])。
```

## 8. 第七步：checkpoint 保存完整 config

当前权重文件名主要依赖 `hidden_size` 和 `_moe` 后缀。Mini-GLM 参数更多，不能只靠文件名判断结构。

修改保存逻辑：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_dpo.py
trainer/trainer_utils.py
```

要求：

```text
1. `.pth` 仍保存 state_dict，保持轻量。
2. 同目录保存一份 `config.json` 或 checkpoint meta。
3. resume checkpoint 保存完整 config。
4. 加载权重前校验 config 是否匹配当前模型。
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT/trainer"
torchrun --nproc_per_node 1 train_pretrain.py \
  --model-name minimind \
  --tokenizer-path ../model \
  --data_path ../data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir ../out/miniglm_2b_a0_6b \
  --save_weight checkpoint_check \
  --hidden_size 128 \
  --num_hidden_layers 2 \
  --max_seq_len 64 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --epochs 1 \
  --save_interval 1 \
  --log_interval 1
```

通过标准：

```text
out/miniglm_2b_a0_6b/checkpoint_check_128.pth 存在。
同目录能看到对应 config/meta 文件。
```

## 9. 第八步：实现 Hugging Face 导出

新增：

```text
scripts/export_miniglm_hf.py
```

输入：

```text
内部 .pth 权重
MiniGLM config json
tokenizer 目录
```

输出：

```text
hf/mini-glm/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
```

导出脚本必须做三件事：

```text
1. 读取内部 pth。
2. 按 HF/Transformers 需要的 key 保存 safetensors。
3. 写入 model_type = glm_moe_dsa 与 architectures。
```

阶段验证命令：

```bash
cd "$MINIMIND_ROOT"
python3 scripts/export_miniglm_hf.py \
  --ckpt out/miniglm_2b_a0_6b/stage4_agent_sft_1536.pth \
  --tokenizer tokenizer/miniglm-32k \
  --config configs/miniglm_2b_a0_6b.json \
  --out hf/mini-glm \
  --safe-serialization true
```

再检查：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from transformers import AutoConfig, AutoTokenizer

path = "hf/mini-glm"
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print(cfg.model_type)
print(cfg.vocab_size, len(tok))
PY
```

通过标准：

```text
model_type 是 glm_moe_dsa。
cfg.vocab_size == len(tokenizer)。
```

## 10. 第九步：再运行 Mini-GLM 目标训练命令

只有完成前面所有阶段后，下面的命令才应该可执行：

```bash
cd "$MINIMIND_ROOT"
test -f tokenizer/miniglm-32k/tokenizer.json
test -f tokenizer/miniglm-32k/tokenizer_config.json
test -f data/miniglm/pretrain_base/smoke_10m.jsonl

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

通过标准：

```text
三个 test -f 检查通过。
参数解析通过。
模型初始化成功。
loss 是有限数值。
out/miniglm_2b_a0_6b/stage0_smoke_1536*.pth 生成。
```

如果 `tokenizer/miniglm-32k/tokenizer.json` 不存在，先完成 [runbook/01-tokenizer-and-data.md](runbook/01-tokenizer-and-data.md)；不要临时把 `--tokenizer-path` 改成别的目录糊过去。

## 11. 你应该如何学习这条路线

不要一次性改完。建议按下面节奏：

```text
第 1 天:
  跑当前 MiniMind smoke test。
  加 tokenizer/vocab/position CLI。

第 2 天:
  加模型工厂。
  保证 minimind 路径不坏。

第 3-5 天:
  写 model_miniglm.py。
  用 tiny config 跑 forward。

第 6 天:
  接入训练脚本。
  跑 mini-glm smoke。

第 7 天:
  写导出脚本。
  Transformers/vLLM 验收。
```

每一天只接受一种结果：命令能跑通，或者失败原因写进报告。不要带着失败进入下一阶段。
