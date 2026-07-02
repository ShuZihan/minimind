# Mini-GLM Engineering Migration Documentation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hand-holding engineering migration document that tells readers how to modify the current MiniMind repository before running Mini-GLM training commands.

**Architecture:** Keep training runbooks as target workflows. Add `docs/miniglm/00-engineering-migration.md` as the mandatory prerequisite document and update navigation so readers do not mistake target commands for current repository commands.

**Tech Stack:** Markdown docs, MiniMind PyTorch scripts, argparse CLI, model factory pattern, Hugging Face export.

---

### Task 1: Add Engineering Migration Document

**Files:**
- Create: `docs/miniglm/00-engineering-migration.md`

- [x] Explain that Mini-GLM is not implemented yet in the current repository.
- [x] List current supported `train_pretrain.py` arguments and unsupported Mini-GLM arguments.
- [x] Provide ordered phases: baseline smoke, tokenizer/config CLI, model factory, Mini-GLM model file, checkpoint/config, export, vLLM.
- [x] Put verification commands only after the code edits that make them runnable.
- [x] Mark current MiniMind smoke as runnable now and Mini-GLM training as runnable only after migration is complete.
- [x] Normalize runnable shell examples to use `MINIMIND_ROOT` and `python3`.
- [x] Document dashed/underscored CLI aliases so target commands do not fail on argument parsing.

### Task 2: Update Navigation

**Files:**
- Modify: `docs/miniglm/README.md`
- Modify: `docs/miniglm/02-training-runbook.md`
- Modify: `docs/miniglm/runbook/00-setup-and-smoke.md`

- [x] Make `00-engineering-migration.md` the first document in reading order.
- [x] Add a prerequisite warning at the top of training runbook.
- [x] Link smoke setup to engineering migration.

### Task 3: Verify Docs

**Files:**
- Check: `docs/miniglm/**/*.md`

- [x] Check Markdown code fences.
- [x] Check local links.
- [x] Check that `00-engineering-migration.md` exists and is linked.
- [x] Run `git diff --check`.
