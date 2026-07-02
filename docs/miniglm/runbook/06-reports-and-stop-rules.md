# 06. 阶段报告、停止条件与参考资料

[返回执行手册](../02-training-runbook.md)

训练模型最怕“感觉还行”。每个阶段都要留下配置、数据、结果和决策。这样下一次 loss 异常、显存爆掉、vLLM 加载失败时，你知道该回到哪里。

## 1. 每阶段报告模板

```markdown
# stage_name report

## Config

- model:
- tokenizer:
- max_seq_len:
- data:
- lr:
- batch:
- accumulation:

## Training

- processed tokens:
- tokens/s:
- loss start:
- loss end:
- aux_loss:
- max GPU memory:

## Evaluation

- held-out loss:
- generation samples:
- patch valid rate:
- apply success rate:
- tests pass rate:
- long-context result:

## Decision

- [ ] promote to next stage
- [ ] repeat same stage
- [ ] change data
- [ ] change hyperparameters
```

## 2. 什么时候停下来

不要继续放大的情况：

```text
loss 不降或周期性爆炸。
MoE aux_loss 异常大。
64K 训练不能引用远距离信息。
patch 格式有效率低于 50%。
vLLM 不能稳定加载 checkpoint。
训练数据 audit 不能证明 SWE-bench 无污染。
```

处理方式：

```text
回到上一个阶段。
缩小数据和长度。
复现实验。
只改一个变量。
写报告。
```

## 3. 第一轮学习路线

先跑最小闭环：

```text
1. 工程最小链路验证:
   10M-100M tokens, 2K。

2. tokenizer:
   32K，验证中文/英文/代码/diff/traceback。

3. base mini run:
   1B tokens, 2K。

4. repo mini run:
   100M-300M tokens, 8K/16K。

5. patch SFT:
   5k-20k issue-to-patch samples。

6. export:
   HF safetensors。

7. vLLM:
   8K serve。

8. manual eval:
   10 个自己写的小 repo bug。
```

再扩大：

```text
10B-30B base tokens
2B-10B repo long-context tokens
50k-200k issue-to-patch tasks
10k-100k agent trajectories
少量 64K 高质量样本
```

## 4. 参考资料

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
