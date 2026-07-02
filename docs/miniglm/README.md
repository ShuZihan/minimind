# Mini-GLM 学习文档目录

这组文档只围绕 Mini-GLM 的设计、训练和 vLLM 推理。

建议阅读顺序：

1. [01-design-overview.md](01-design-overview.md)
   先看设计总览：模型规格、tokenizer、数据计划、训练阶段、导出和 vLLM 推理路线。

2. [02-training-runbook.md](02-training-runbook.md)
   再看执行手册索引：它把训练拆成若干阶段，告诉你今天该打开哪一篇。

3. 分阶段 runbook：
   - [00. 环境准备与最小链路验证](runbook/00-setup-and-smoke.md)
   - [01. Tokenizer 与数据治理](runbook/01-tokenizer-and-data.md)
   - [02. Base Pretraining](runbook/02-base-pretraining.md)
   - [03. Repo-level 长上下文训练](runbook/03-long-context.md)
   - [04. Patch SFT、Agent SFT 与偏好优化](runbook/04-sft-and-preference.md)
   - [05. 导出 Hugging Face checkpoint 与 vLLM 推理](runbook/05-export-and-vllm.md)
   - [06. 阶段报告、停止条件与参考资料](runbook/06-reports-and-stop-rules.md)

## 文档职责

```text
01-design-overview.md
  回答“为什么这么设计”。

02-training-runbook.md
  回答“训练应该按什么顺序推进”。

runbook/*.md
  回答“这个阶段今天应该怎么一步步做”。
```

## 当前目标

```text
Mini-GLM-5.1-2.1B-A0.6B
vocab_size = 32768
max_position_embeddings = 65536
export target = glm_moe_dsa-compatible Hugging Face checkpoint（检查点权重）
inference target = vLLM OpenAI-compatible API
```
