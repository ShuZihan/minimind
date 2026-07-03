# 01. Mini-GLM 工程化改造

[返回目录](../README.md)

这一章把当前 MiniMind 仓库改造成能识别 Mini-GLM 的工程结构。每一节都按“改什么、怎么验收”推进。不要跳到后续训练章；这里的目标只是把接口和模型路径接通。

先回到仓库根目录：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
```

## 1. 训练脚本支持 tokenizer 路径

修改文件：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_dpo.py
```

在 `argparse` 中加入：

```python
parser.add_argument(
    "--tokenizer-path", "--tokenizer_path",
    dest="tokenizer_path",
    type=str,
    default="../model",
    help="tokenizer 路径"
)
```

把：

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

验收：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_pretrain.py -h | grep tokenizer
python3 trainer/train_full_sft.py -h | grep tokenizer
python3 trainer/train_dpo.py -h | grep tokenizer
```

三个脚本都应显示 `--tokenizer-path`。

## 2. 训练脚本支持 vocab、position 和横线别名

继续修改同三个训练脚本。

新增：

```python
parser.add_argument("--vocab-size", "--vocab_size", dest="vocab_size", type=int, default=6400, help="词表大小")
parser.add_argument("--max-position-embeddings", "--max_position_embeddings", dest="max_position_embeddings", type=int, default=32768, help="最大位置编码长度")
```

把原来的 `--hidden_size`、`--num_hidden_layers` 替换为横线/下划线双别名：

```python
parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=768, help="隐藏层维度")
parser.add_argument("--num-hidden-layers", "--num_hidden_layers", dest="num_hidden_layers", type=int, default=8, help="隐藏层数量")
```

构造 `MiniMindConfig` 时带上新字段：

```python
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe),
    vocab_size=args.vocab_size,
    max_position_embeddings=args.max_position_embeddings,
)
```

验收：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_pretrain.py -h | grep vocab
python3 trainer/train_pretrain.py -h | grep max-position
python3 trainer/train_pretrain.py -h | grep hidden-size
```

再跑极小单进程解析测试：

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

这会真的开始训练一个小模型。只想验证参数解析时，可以看到启动后手动停止。

## 3. tokenizer 训练脚本支持 CLI

当前 `trainer/train_tokenizer.py` 默认是硬编码常量，需要改成可传参。

让 `get_texts` 同时支持预训练 JSONL 的 `text` 字段和 SFT JSONL 的 `conversations` 字段：

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

把文件底部改成：

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

验收：

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_tokenizer.py -h | grep -E 'input|vocab-size|output-dir'
```

## 4. 引入模型工厂

新建：

```text
model/model_factory.py
```

写入：

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

```python
from model.model_factory import build_model
```

并把 `init_model` 里的：

```python
model = MiniMindForCausalLM(lm_config)
```

替换为：

```python
model = build_model(lm_config)
```

训练脚本中把 `MiniMindConfig` 构造改为：

```python
from model.model_factory import build_config

parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default="minimind", choices=["minimind", "mini-glm"], help="模型实现名称")

lm_config = build_config(args)
```

验收：

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

应输出：

```text
minimind MiniMindForCausalLM
```

## 5. 加 Mini-GLM 配置

新建：

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

训练脚本还要加入 Mini-GLM 结构参数：

```python
parser.add_argument("--first-k-dense-replace", "--first_k_dense_replace", dest="first_k_dense_replace", type=int, default=3)
parser.add_argument("--n-routed-experts", "--n_routed_experts", dest="n_routed_experts", type=int, default=16)
parser.add_argument("--n-shared-experts", "--n_shared_experts", dest="n_shared_experts", type=int, default=1)
parser.add_argument("--num-experts-per-tok", "--num_experts_per_tok", dest="num_experts_per_tok", type=int, default=2)
parser.add_argument("--moe-intermediate-size", "--moe_intermediate_size", dest="moe_intermediate_size", type=int, default=768)
parser.add_argument("--dense-intermediate-size", "--dense_intermediate_size", dest="dense_intermediate_size", type=int, default=4096)
```

验收：

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

## 6. 实现 MiniGLMForCausalLM

这一步才是真正的模型实现。不能只写空壳类，否则训练能启动但不是 Mini-GLM。

需要实现：

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

建议顺序：

```text
1. 复用 MiniMind 的 RMSNorm、RoPE、CausalLM 输出结构。
2. 先实现 dense layer，跑通 tiny forward。
3. 再实现 MoE layer，跑通 tiny forward。
4. 最后加入 DSA/MLA attention。
```

验收：

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

## 7. checkpoint 保存完整 config

Mini-GLM 参数多，不能只靠文件名判断结构。修改训练保存逻辑：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/train_dpo.py
trainer/trainer_utils.py
```

要求：

```text
1. `.pth` 仍保存 state_dict。
2. 同目录保存 config.json 或 checkpoint meta。
3. resume checkpoint 保存完整 config。
4. 加载权重前校验 config 是否匹配当前模型。
```

验收：

```bash
cd "$MINIMIND_ROOT/trainer"
torchrun --nproc_per_node 1 train_pretrain.py \
  --model-name minimind \
  --tokenizer-path ../model \
  --data_path ../data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir ../out/miniglm_2b_a0_6b \
  --save_weight checkpoint_check \
  --hidden-size 128 \
  --num-hidden-layers 2 \
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

## 8. 实现 Hugging Face 导出脚本

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

导出脚本必须写入：

```text
model_type = glm_moe_dsa
architectures = ["GlmMoeDsaForCausalLM"]
```

验收：

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

## 9. 过关标准

```text
训练脚本能识别 --model-name mini-glm。
tokenizer 训练脚本能识别 --input / --vocab-size / --output-dir。
model_factory 能构造 minimind 和 mini-glm config。
MiniGLM tiny forward 可运行。
checkpoint 旁边有完整 config/meta。
HF 导出目录能被 Transformers 读取。
```

下一步：打开 [02-tokenizer-data-and-target-smoke.md](02-tokenizer-data-and-target-smoke.md)。
