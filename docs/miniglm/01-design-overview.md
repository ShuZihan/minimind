# Mini-GLM 设计总览：从规格到 vLLM

[返回目录](README.md)

这篇文档只回答一个问题：如何在当前仓库里做一个教学版 Mini-GLM，并最终用 vLLM 推理。

它不是 GLM-5.1 复刻。目标是保留 DSA、MLA-style attention、MoE、shared expert、GLM-compatible export 这些关键结构，以 2.1B-A0.6B 规格跑通训练、导出和推理闭环。逐步执行手册见 [02-training-runbook.md](02-training-runbook.md)。

---

## 1. 结论

第一版建议：

```text
Mini-GLM-5.1-2.1B-A0.6B
vocab_size = 32768
max_position_embeddings = 65536
```

核心配置：

```text
hidden_size = 1536
num_hidden_layers = 30
first_k_dense_replace = 3
n_routed_experts = 16
n_shared_experts = 1
num_experts_per_tok = 2
moe_intermediate_size = 768
dense_intermediate_size = 4096
tie_word_embeddings = false
dtype = bfloat16
```

几个关键判断：

1. 先做 2.1B-A0.6B，不直接做 6B。这个规格更适合验证结构、训练、导出和 vLLM 链路。
2. 词表用 32K。它能兼顾中文、英文、代码、diff、traceback 和工具调用格式。
3. 上下文上限设 64K，但训练长度从 2K/4K 开始，逐步到 8K、16K、32K，最后少量 64K。
4. 官方 SWE-bench / Lite / Verified / Multilingual 是评测集，不进入训练集。
5. vLLM 推理走 `glm_moe_dsa` 兼容 Hugging Face checkpoint（检查点权重），不是让 vLLM 单独认识自定义 `MiniGLM`。

---

## 2. 目标链路

最终链路应该是：

```text
训练语料
  -> 32K tokenizer
  -> base pretraining
  -> repo-level 长上下文训练
  -> issue-to-patch SFT
  -> agent trajectory SFT
  -> 测试反馈优化
  -> Hugging Face / safetensors 导出
  -> vLLM OpenAI-compatible API
```

模型要逐步学会：

```text
读 issue。
读多文件 repo context。
理解 traceback / pytest / 测试意图。
生成可 apply 的 unified diff。
按 read/search/edit/run_tests 流程工作。
在长上下文里引用前文。
```

---

## 3. 模型规格

### 3.1 为什么是 2.1B-A0.6B

2.1B-A0.6B 规格足够容纳 32K 词表、MoE、shared expert、DSA/MLA 等结构，也能在 8xH20 141G 上比较从容地试错。

粗略参数：

```text
body total params  ≈ 1.96B
body active params ≈ 0.63B
vocab 层参数       ≈ 0.10B
总参数            ≈ 2.07B
```

这里的 `A0.6B` 指每个 token 推理时激活的主体参数量，不把完整 vocab projection 按同一口径计入。

### 3.2 为什么不像 GLM 原版的 total/active 比

GLM-5.1 原版专家池很大，稀疏比例高。第一版 Mini-GLM 只有 `16E/top-2`，所以 total/active 比较低。

| 配置 | body total / active | 比例 |
|---|---:|---:|
| 16E/top-2 | 1.96B-A0.63B | 3.1 |
| 64E/top-2 | 6.55B-A0.63B | 10.4 |
| 128E/top-4 | 12.67B-A0.82B | 15.4 |
| 256E/top-8 | 24.91B-A1.21B | 20.6 |

结论：2.1B-A0.6B 版用于教学验证；6B 以上才更像“稀疏大专家池”的 MoE。

### 3.3 什么能缩，什么不能删

可以缩小：

```text
hidden_size
num_hidden_layers
expert 数量
expert 中间维度
长上下文训练比例
vocab size
```

不建议删：

```text
DSA
MLA-style attention 参数化
MoE routed experts
shared expert
GLM-compatible config/export 路线
```

如果去掉 DSA，就只能叫 GLM-inspired，不能叫 Mini-GLM-5.1-style。

---

## 4. 32K 词表与 64K 上下文

这两个数字控制不同东西：

```text
vocab_size = 32768:
  tokenizer 词表大小。
  决定 embedding / lm_head 的第一维。

max_position_embeddings = 65536:
  模型支持的最大位置数。
  不等于训练时一开始就用 64K。
```

推荐长度课程：

```text
2K / 4K:
  学语言、代码语法、局部 API 和函数结构。

8K / 16K:
  学文件级上下文、测试、局部多文件依赖。

32K:
  长 issue、长 traceback、issue-to-patch、agent trajectory。

64K:
  少量真实远距离依赖样本。
```

64K 样本应该包含真实依赖，例如：

```text
repo tree + 多文件源码 + tests + issue
前文 API/约束在末尾被测试
traceback 与后半段源码对应
agent trajectory 跨多轮 read/search/edit/run_tests
```

不要随机拼接短文本凑 64K。

---

## 5. 数据计划

Mini-GLM 数据分四层：

| 层级 | 作用 | 典型数据 |
|---|---|---|
| Base text/code | 学语言和代码基本形态 | 通用文本、The Stack、permissive repos |
| Repo context | 学多文件和长上下文 | repo tree、源码、测试、docs |
| Issue-to-patch | 学根据问题生成修改 | SWE-smith、自建 issue/PR |
| Agent trajectory | 学工具调用和测试闭环 | read/search/edit/run_tests 轨迹 |

推荐起点：

```text
60% code/repo continued pretraining
20% long-context repo understanding
15% issue-to-patch SFT
5% agent trajectory / test feedback
```

如果参加 SWE-bench，官方数据只做评测。不要训练：

```text
problem_statement
patch
test_patch
FAIL_TO_PASS
PASS_TO_PASS
issue_url
pr_url
base_commit
repo 在该 commit 附近的状态
```

必须维护：

```text
data_manifest.jsonl
swebench_blacklist.json
dedup_report.md
eval_protocol.md
```

---

## 6. 训练路线

第一轮先跑最小闭环，再放大。

| 阶段 | 目标 | 建议量级 |
|---|---|---:|
| 最小链路验证 | 训练、保存、导出、推理能跑 | 10M-100M tokens |
| base pretraining | 学语言和代码基础 | 1B-3B tokens 起步 |
| repo-level 长上下文 | 学多文件上下文 | 2B-10B tokens |
| issue-to-patch SFT | 学输出 unified diff | 50k-200k tasks |
| agent trajectory SFT | 学工具调用流程 | 10k-100k trajectories |
| 测试反馈优化 | 偏向能通过测试的修改 | 按采样成本决定 |

详细命令见 [02-training-runbook.md](02-training-runbook.md)。

---

## 7. 导出与 vLLM

训练阶段通常保存内部 `.pth`，vLLM 更适合加载 Hugging Face 目录：

```text
hf/mini-glm/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
```

导出要同时对齐：

```text
config.json:
  model_type = glm_moe_dsa
  architectures = ["GlmMoeDsaForCausalLM"]

tokenizer:
  len(tokenizer) == vocab_size == 32768

weights:
  state_dict key 和 shape 符合 GlmMoeDsaForCausalLM 期望
```

核心 config：

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

最小 vLLM 命令：

```bash
vllm serve hf/mini-glm \
  --served-model-name mini-glm \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --trust-remote-code \
  --port 8998
```

先验收 8K，再逐步开放 16K、32K、64K。

---

## 8. 8xH20 预算口径

普通 DDP 下，每张卡都有完整模型、梯度和优化器状态，8 张卡不是一个 1128G 大显存池。

AdamW 粗估：

```text
bf16 params: 2 bytes/param
bf16 grads: 2 bytes/param
fp32 master weights: 4 bytes/param
fp32 Adam m: 4 bytes/param
fp32 Adam v: 4 bytes/param
合计约 16 bytes/param
```

2.1B 参数约：

```text
2.1B * 16 bytes ≈ 33.6GB / GPU
```

再加 activation、MoE routing buffer、通信 buffer、padding 和长上下文开销。141G 单卡足够起步，但 32K/64K 仍要谨慎调 batch size 和 activation checkpointing。

训练时间用实测吞吐估：

```text
总时间 = 目标 tokens / 实测 tokens/s
```

---

## 9. 工程改造顺序

1. 模型工厂：`build_model(config)` 支持 `mini_glm / glm_moe_dsa`。
2. 模型实现：新增 `model/model_miniglm.py`。
3. 参数暴露：训练脚本支持结构参数、词表、最大位置数。
4. sequence packing：减少 padding 浪费，提高有效 tokens/s。
5. HF/vLLM 导出：新增 `scripts/export_miniglm_hf.py`。
6. 数据审计：manifest、SWE-bench 黑名单、去重报告、评测协议。

顺序原则：先能跑，再有效训练，再可推理，最后再放大能力。

---

## 10. 复习问题

1. 为什么第一版先做 2.1B-A0.6B？
2. `vocab_size = 32768` 和 `max_position_embeddings = 65536` 分别控制什么？
3. 为什么 64K 不能靠随机拼接短样本训练？
4. 为什么官方 SWE-bench 不能进训练集？
5. `.pth` 导出成 Hugging Face/safetensors 的目的是什么？
6. vLLM 识别模型结构时看哪些文件？

---

## 11. 参考资料

```text
GLM-5.1 config:
https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json

GLM-5 technical report:
https://arxiv.org/html/2602.15763

vLLM GLM recipe:
https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html

vLLM serve CLI:
https://docs.vllm.ai/en/stable/cli/serve/

SWE-bench datasets guide:
https://www.swebench.com/SWE-bench/guides/datasets/

SWE-smith:
https://github.com/SWE-bench/SWE-smith

The Stack v2:
https://huggingface.co/datasets/bigcode/the-stack-v2

Chinchilla scaling:
https://arxiv.org/pdf/2203.15556
```
