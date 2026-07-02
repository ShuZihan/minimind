# 05. 导出 Hugging Face checkpoint 与 vLLM 推理

[返回执行手册](../02-training-runbook.md)

训练脚本保存的一般是仓库内部 `.pth`。vLLM 更适合读取 Hugging Face 目录：`config.json` 描述结构，`model.safetensors` 保存权重，tokenizer 文件负责分词和 chat template。

这一步的核心是把训练世界和推理世界接起来。

## 1. 目标目录

```text
hf/mini-glm/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
```

config 核心字段：

```json
{
  "architectures": ["GlmMoeDsaForCausalLM"],
  "model_type": "glm_moe_dsa",
  "dtype": "bfloat16",
  "vocab_size": 32768,
  "max_position_embeddings": 65536,
  "hidden_size": 1536,
  "num_hidden_layers": 30,
  "first_k_dense_replace": 3,
  "n_routed_experts": 16,
  "n_shared_experts": 1,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 768,
  "intermediate_size": 4096,
  "tie_word_embeddings": false
}
```

## 2. 导出

```bash
python3 scripts/export_miniglm_hf.py \
  --ckpt out/miniglm_2b_a0_6b/stage4_agent_sft_1536.pth \
  --tokenizer tokenizer/miniglm-32k \
  --config configs/miniglm_2b_a0_6b.json \
  --out hf/mini-glm \
  --safe-serialization true
```

这条命令把仓库内部 `.pth` 转成 Hugging Face 目录。`--ckpt` 是训练得到的权重，`--tokenizer` 复制 32K tokenizer，`--config` 写入 GLM-MoE-DSA 结构字段，`--safe-serialization true` 保存为 safetensors。

## 3. Transformers 加载检查

```bash
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

先用 Transformers 检查，比直接上 vLLM 更容易定位错误。`vocab_size` 不一致通常是 tokenizer/config 问题；模型实例化失败通常是 `model_type`、`architectures`、权重 key 或 shape 不匹配。

过关标准：

```text
cfg.model_type == glm_moe_dsa
cfg.vocab_size == len(tokenizer) == 32768
模型能实例化
```

## 4. vLLM 8K 启动

```bash
vllm serve hf/mini-glm \
  --served-model-name mini-glm \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --trust-remote-code \
  --port 8998
```

这条命令启动推理服务。`--served-model-name` 是 API 里看到的模型名；`--tensor-parallel-size 8` 用 8 卡张量并行；`--max-model-len 8192` 先限制在 8K，避免一开始就把显存和延迟问题放大。

检查服务：

```bash
curl http://localhost:8998/v1/models
```

这个请求只确认服务是否启动、模型名是否注册成功。

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

这个请求检查 OpenAI-compatible chat API、tokenizer、chat template 和基础生成是否正常。

## 5. 逐步开放长上下文

```bash
--max-model-len 16384
--max-model-len 32768
--max-model-len 65536
```

每次提升都记录：

```text
prefill latency
decode tokens/s
GPU memory
是否 OOM
是否能引用前文
```

## 6. 工具调用

```bash
vllm serve hf/mini-glm \
  --served-model-name mini-glm \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --chat-template-content-format string \
  --trust-remote-code \
  --port 8998
```

这些 parser 只负责解析输出，不会让模型自动学会工具调用。只有模型在 SFT 中见过稳定的 tool_call 格式，它们才有意义。

过关标准：

```text
基础 chat 能返回。
不乱码、不明显重复。
代码输出正常。
tool_call JSON 可解析。
64K 下能引用前文。
```
