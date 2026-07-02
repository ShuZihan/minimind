# Mini-GLM 训练执行手册

[返回目录](README.md)

这份手册不再把所有命令塞进一个长文档。训练是分阶段推进的：每一阶段解决一个清晰的问题，跑完后留下报告，再决定是否进入下一阶段。

前置条件：

```text
必须先完成 00-engineering-migration.md。
否则，只有当前 MiniMind smoke test 可执行；Mini-GLM 目标训练命令不可执行。
```

目标配置：

```text
Mini-GLM-5.1-2.1B-A0.6B
vocab_size = 32768
max_position_embeddings = 65536
export target = glm_moe_dsa-compatible Hugging Face checkpoint（检查点权重）
inference target = vLLM OpenAI-compatible API
```

读命令时先抓四类参数：

```text
输入：读哪个 tokenizer、数据集、checkpoint。
输出：保存到哪个目录、checkpoint 叫什么。
模型结构：hidden_size、layers、experts、vocab、position 上限。
训练控制：max_seq_len、batch、梯度累积、学习率、保存和日志频率。
```

## 1. 阅读顺序

0. [00-engineering-migration.md](00-engineering-migration.md)
   先把当前 MiniMind 仓库改造成能识别 Mini-GLM 的工程结构。

1. [00. 环境准备与最小链路验证](runbook/00-setup-and-smoke.md)
   先证明训练链路能跑通：目录、环境、模型接口、8 卡 smoke test、checkpoint。

2. [01. Tokenizer 与数据治理](runbook/01-tokenizer-and-data.md)
   固定 32K tokenizer，建立数据 manifest 和 SWE-bench 黑名单。

3. [02. Base Pretraining](runbook/02-base-pretraining.md)
   先用 100M tokens 标定吞吐，再做 2K/4K base 预训练。

4. [03. Repo-level 长上下文训练](runbook/03-long-context.md)
   从 8K 到 64K 逐步拉长上下文，让模型学习跨文件、跨日志、跨工具调用定位信息。

5. [04. Patch SFT、Agent SFT 与偏好优化](runbook/04-sft-and-preference.md)
   训练 issue-to-patch、工具轨迹和基于测试反馈的偏好数据。

6. [05. 导出 Hugging Face checkpoint 与 vLLM 推理](runbook/05-export-and-vllm.md)
   把内部 `.pth` 导出成 Hugging Face 目录，并用 vLLM 做 8K 到 64K 推理验收。

7. [06. 阶段报告、停止条件与参考资料](runbook/06-reports-and-stop-rules.md)
   每阶段写报告，遇到 loss、长上下文、patch 或 vLLM 问题时知道何时停下来。

## 2. 阶段地图

```text
setup/smoke
  -> tokenizer + data audit
  -> base pretraining 2K/4K
  -> repo long-context 8K/16K/32K/64K
  -> issue-to-patch SFT
  -> agent trajectory SFT
  -> preference optimization
  -> HF export
  -> vLLM inference
```

这条路线的思想很简单：先让工程链路可靠，再让模型学会基础分布；先训短序列，再训长序列；先输出补丁，再学习工具流程；最后导出并用 vLLM 验收。

## 3. 每阶段怎么推进

每个阶段都按四件事推进：

```text
目标
输入
命令
过关标准
```

不要只看 loss。至少同时记录：

```text
tokens/s
max GPU memory
checkpoint 是否可恢复
held-out loss
生成样例
patch apply 成功率
长上下文引用能力
vLLM 加载和推理结果
```

## 4. 第一轮建议

第一轮不要直接冲完整规模。先跑一个能闭合的小循环：

```text
10M-100M tokens smoke test
32K tokenizer
1B tokens base 2K
100M-300M tokens repo 8K/16K
5k-20k issue-to-patch SFT
HF export
vLLM 8K serve
10 个手写小 repo bug 评估
```

如果这个闭环稳定，再扩大到：

```text
10B-30B base tokens
2B-10B repo long-context tokens
50k-200k issue-to-patch tasks
10k-100k agent trajectories
少量 64K 高质量样本
```

这不是保守，而是训练工程的基本节奏：先让小实验能解释，再让大实验值得花钱。
