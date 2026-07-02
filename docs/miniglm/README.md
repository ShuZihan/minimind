# Mini-GLM 学习文档目录

这组文档只围绕 Mini-GLM 的设计、训练和 vLLM 推理。

除非某一节特别说明，所有命令都从仓库根目录开始执行。第一次进入服务器后，先运行：

```bash
cd "$(git rev-parse --show-toplevel)"
export MINIMIND_ROOT="$(pwd)"
```

## 执行顺序

如果目的是“照着跑”，按这个顺序来：

```text
1. runbook/00-setup-and-smoke.md 的第 1-5 节：环境、依赖、官方数据、当前 MiniMind smoke test。
2. 00-engineering-migration.md 的第 2-9 节：补齐 Mini-GLM 所需工程接口。
3. runbook/01-tokenizer-and-data.md：训练 32K tokenizer，并生成 base pretraining 最小可跑文件。
4. 00-engineering-migration.md 的 Mini-GLM 目标 smoke 命令。
5. 从 runbook/02-base-pretraining.md 开始进入正式训练阶段。
```

不要把所有 Markdown 代码块当作一整段 shell 脚本顺序粘贴。带 `python` 的代码块是要写进源码的片段；带 `text/json/markdown` 的代码块是格式说明。

## 阅读顺序

如果目的是先理解设计，再执行，建议阅读顺序：

1. [00-engineering-migration.md](00-engineering-migration.md)
   先看工程化改造：当前 MiniMind 仓库还不能直接训练 Mini-GLM，必须先接入模型工厂、Mini-GLM 模型实现、训练参数和导出脚本。

2. [01-design-overview.md](01-design-overview.md)
   先看设计总览：模型规格、tokenizer、数据计划、训练阶段、导出和 vLLM 推理路线。

3. [02-training-runbook.md](02-training-runbook.md)
   再看执行手册索引：它把训练拆成若干阶段，告诉你今天该打开哪一篇。

4. 分阶段 runbook：
   - [00. 环境准备与最小链路验证](runbook/00-setup-and-smoke.md)
   - [01. Tokenizer 与数据治理](runbook/01-tokenizer-and-data.md)
   - [02. Base Pretraining](runbook/02-base-pretraining.md)
   - [03. Repo-level 长上下文训练](runbook/03-long-context.md)
   - [04. Patch SFT、Agent SFT 与偏好优化](runbook/04-sft-and-preference.md)
   - [05. 导出 Hugging Face checkpoint 与 vLLM 推理](runbook/05-export-and-vllm.md)
   - [06. 阶段报告、停止条件与参考资料](runbook/06-reports-and-stop-rules.md)

## 文档职责

```text
00-engineering-migration.md
  回答“当前 MiniMind 仓库要先怎么改，Mini-GLM 命令才会真的可执行”。

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
