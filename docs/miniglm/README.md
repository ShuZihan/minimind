# Mini-GLM 学习文档目录

这组文档只围绕 Mini-GLM 的设计、训练和 vLLM 推理。

## 唯一执行主线

如果你要边做边学，只读 `runbook/` 下面这 5 篇，并严格按文件名顺序执行：

```text
00-preflight-and-current-smoke.md
01-engineering-migration.md
02-tokenizer-data-and-target-smoke.md
03-pretraining-and-long-context.md
04-sft-export-vllm-and-reports.md
```

入口：

[runbook/00-preflight-and-current-smoke.md](runbook/00-preflight-and-current-smoke.md)

每一章只指向下一章，不需要在文档之间来回跳。

## 背景材料

[01-design-overview.md](01-design-overview.md) 解释模型规格、tokenizer、数据计划、训练阶段、导出和 vLLM 推理路线。它是背景阅读，不是执行入口。

[02-training-runbook.md](02-training-runbook.md) 是 5 篇主线 runbook 的目录索引。它也不是另一套执行流程。

[00-engineering-migration.md](00-engineering-migration.md) 只保留为旧链接兼容页；真正的工程化步骤在 [runbook/01-engineering-migration.md](runbook/01-engineering-migration.md)。

## 当前目标

```text
Mini-GLM-5.1-2.1B-A0.6B
vocab_size = 32768
max_position_embeddings = 65536
export target = glm_moe_dsa-compatible Hugging Face checkpoint
inference target = vLLM OpenAI-compatible API
```
