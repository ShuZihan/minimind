# 02. Tokenizer、数据准备与 Mini-GLM Target Smoke

[返回目录](../README.md)

这一章开始使用上一章改造后的 Mini-GLM 工程接口。目标是训练 32K tokenizer，准备最小可跑预训练数据，然后启动一次 Mini-GLM target smoke。

先回到仓库根目录：

```bash
cd "${MINIMIND_ROOT:-$(git rev-parse --show-toplevel)}"
export MINIMIND_ROOT="$(pwd)"
```

前置检查：

```bash
cd "$MINIMIND_ROOT"
test -f dataset/pretrain_t2t_mini.jsonl
python3 trainer/train_tokenizer.py -h | grep -E 'input|vocab-size|output-dir'
python3 trainer/train_pretrain.py -h | grep model-name
```

## 1. 生成 tokenizer 训练语料

先从官方预训练数据切出一个最小语料。正式训练时可以替换为更干净、更覆盖代码场景的语料。

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
dst = Path("data/miniglm/tokenizer_corpus/train.jsonl")
target_bytes = 200 * 1024 * 1024

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

test -s data/miniglm/tokenizer_corpus/train.jsonl
head -n 3 data/miniglm/tokenizer_corpus/train.jsonl
```

## 2. 训练 32K tokenizer

```bash
cd "$MINIMIND_ROOT"
python3 trainer/train_tokenizer.py \
  --input data/miniglm/tokenizer_corpus/train.jsonl \
  --vocab-size 32768 \
  --output-dir tokenizer/miniglm-32k
```

验收：

```bash
cd "$MINIMIND_ROOT"
python3 - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer/miniglm-32k")
samples = [
    "你好，请解释 MoE 和 dense 模型的区别。",
    "def add(a, b):\n    return a + b\n",
    "diff --git a/src/a.py b/src/a.py\n@@ -1,3 +1,3 @@\n-pytest.fail()\n+assert True\n",
    "Traceback (most recent call last):\n  File \"tests/test_a.py\", line 10, in test_x\nAssertionError\n",
]

print("vocab", len(tok))
for s in samples:
    ids = tok.encode(s)
    print("-" * 80)
    print("chars", len(s), "tokens", len(ids), "chars/token", round(len(s) / max(len(ids), 1), 2))
    print(tok.decode(ids)[:200])
PY
```

通过标准：

```text
len(tok) == 32768
中文、英文、代码能正常 encode/decode。
diff/traceback 不被切得过碎。
chat_template 存在并能 apply_chat_template。
```

## 3. 准备 base pretraining 最小可跑文件

下一章会用：

```text
data/miniglm/pretrain_base/train_100m.jsonl
data/miniglm/pretrain_base/train.jsonl
data/miniglm/pretrain_base/valid.jsonl
```

先用官方 `pretrain_t2t_mini.jsonl` 建最小版本：

```bash
cd "$MINIMIND_ROOT"
mkdir -p data/miniglm/pretrain_base
ln -sfn ../../../dataset/pretrain_t2t_mini.jsonl data/miniglm/pretrain_base/train.jsonl
python3 - <<'PY'
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
train_100m = Path("data/miniglm/pretrain_base/train_100m.jsonl")
valid = Path("data/miniglm/pretrain_base/valid.jsonl")

if not src.exists():
    raise SystemExit(f"missing {src}")

def copy_bytes(dst, target_bytes, skip_bytes=0):
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        for line in fin:
            if skipped < skip_bytes:
                skipped += len(line)
                continue
            if not line.strip():
                continue
            fout.write(line)
            written += len(line)
            if written >= target_bytes:
                break
    print(dst, written, "bytes")

copy_bytes(train_100m, 100 * 1024 * 1024)
copy_bytes(valid, 10 * 1024 * 1024, skip_bytes=100 * 1024 * 1024)
PY

ls -lh data/miniglm/pretrain_base/train.jsonl \
       data/miniglm/pretrain_base/train_100m.jsonl \
       data/miniglm/pretrain_base/valid.jsonl
```

## 4. 建立数据审计记录

正式训练数据必须能追溯来源、许可证、仓库、commit 和 split。先建立文件：

```bash
cd "$MINIMIND_ROOT"
touch data/miniglm/audit/data_manifest.jsonl
touch data/miniglm/audit/swebench_blacklist.json
```

`data_manifest.jsonl` 每行建议记录：

```json
{
  "sample_id": "pretrain_base_00000001",
  "source": "the-stack-v2",
  "license": "mit",
  "repo": "owner/name",
  "commit": "abc123",
  "url": "https://example.com/owner/name",
  "split": "train",
  "sha256": "hex",
  "tags": ["code", "python", "pretrain"]
}
```

如果你要参与 SWE-bench 类测评，官方 SWE-bench / Lite / Verified / Multilingual 的 issue、PR、patch、test_patch 都不能进训练集。

## 5. 跑 Mini-GLM target smoke

前置检查：

```bash
cd "$MINIMIND_ROOT"
test -f tokenizer/miniglm-32k/tokenizer.json
test -f tokenizer/miniglm-32k/tokenizer_config.json
test -f data/miniglm/pretrain_base/smoke_10m.jsonl
python3 trainer/train_pretrain.py -h | grep model-name
python3 trainer/train_pretrain.py -h | grep n-routed-experts
```

启动 target smoke：

```bash
cd "$MINIMIND_ROOT"
torchrun --nproc_per_node 8 trainer/train_pretrain.py \
  --model-name mini-glm \
  --tokenizer-path tokenizer/miniglm-32k \
  --data_path data/miniglm/pretrain_base/smoke_10m.jsonl \
  --save_dir out/miniglm_2b_a0_6b \
  --save_weight stage0_miniglm_target_smoke \
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
参数解析通过。
模型初始化成功。
loss 是有限数值。
out/miniglm_2b_a0_6b/stage0_miniglm_target_smoke_*.pth 生成。
```

下一步：打开 [03-pretraining-and-long-context.md](03-pretraining-and-long-context.md)。
