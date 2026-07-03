# Mini-GLM 训练执行手册

[返回目录](README.md)

这份文件只是执行目录。真正要照着做的内容在 `runbook/` 下，按文件名顺序阅读即可。

## 执行顺序

1. [00. 环境准备与当前 MiniMind Smoke](runbook/00-preflight-and-current-smoke.md)
   验证服务器、依赖、官方数据、当前 MiniMind 训练链路和 checkpoint 保存。

2. [01. Mini-GLM 工程化改造](runbook/01-engineering-migration.md)
   改训练脚本 CLI、tokenizer CLI、模型工厂、MiniGLMConfig、MiniGLMForCausalLM、checkpoint config 和 HF 导出脚本。

3. [02. Tokenizer、数据准备与 Mini-GLM Target Smoke](runbook/02-tokenizer-data-and-target-smoke.md)
   训练 32K tokenizer，生成最小预训练数据，启动 Mini-GLM target smoke。

4. [03. Base Pretraining 与长上下文训练](runbook/03-pretraining-and-long-context.md)
   做 100M tokens 吞吐标定、base 2K/4K、repo 8K/16K/32K/64K。

5. [04. SFT、偏好优化、导出、vLLM 与报告](runbook/04-sft-export-vllm-and-reports.md)
   做 Patch SFT、Agent SFT、DPO、HF 导出、vLLM 推理验收和阶段报告。

## 读命令时抓四类参数

```text
输入：读哪个 tokenizer、数据集、checkpoint。
输出：保存到哪个目录、checkpoint 叫什么。
模型结构：hidden_size、layers、experts、vocab、position 上限。
训练控制：max_seq_len、batch、梯度累积、学习率、保存和日志频率。
```

不要把旧文档里的跳转路径当作执行路线。现在只有上面 5 篇主线 runbook。
