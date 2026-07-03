# Mini-GLM Linear Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Mini-GLM training docs into a small, linear runbook sequence that can be read and executed in filename order without jumping between documents.

**Architecture:** Keep multiple files, but make `docs/miniglm/runbook/00..04` the only execution path. Existing scattered docs are consolidated into five chapters; old branch-style entry points are removed or reduced to directory/index context.

**Tech Stack:** Markdown documentation, MiniMind PyTorch scripts, bash command examples.

---

### Task 1: Create Five Linear Runbook Chapters

**Files:**
- Create: `docs/miniglm/runbook/00-preflight-and-current-smoke.md`
- Create: `docs/miniglm/runbook/01-engineering-migration.md`
- Create: `docs/miniglm/runbook/02-tokenizer-data-and-target-smoke.md`
- Create: `docs/miniglm/runbook/03-pretraining-and-long-context.md`
- Create: `docs/miniglm/runbook/04-sft-export-vllm-and-reports.md`

- [x] Move current environment/data/current smoke content into chapter 00.
- [x] Move engineering migration content into chapter 01.
- [x] Move tokenizer/data preparation and Mini-GLM target smoke into chapter 02.
- [x] Move base pretraining and long-context training into chapter 03.
- [x] Move SFT, preference, export, vLLM, reports, and stop rules into chapter 04.
- [x] Ensure each chapter ends with exactly one next-step pointer.

### Task 2: Remove Branching Execution Routes

**Files:**
- Delete or replace old runbook files:
  - `docs/miniglm/runbook/00-setup-and-smoke.md`
  - `docs/miniglm/runbook/01-tokenizer-and-data.md`
  - `docs/miniglm/runbook/02-base-pretraining.md`
  - `docs/miniglm/runbook/03-long-context.md`
  - `docs/miniglm/runbook/04-sft-and-preference.md`
  - `docs/miniglm/runbook/05-export-and-vllm.md`
  - `docs/miniglm/runbook/06-reports-and-stop-rules.md`
- Modify: `docs/miniglm/00-engineering-migration.md`

- [x] Remove old split runbook files to avoid duplicate execution paths.
- [x] Turn `docs/miniglm/00-engineering-migration.md` into a short reference page pointing to the linear runbook chapter.

### Task 3: Update Navigation

**Files:**
- Modify: `docs/miniglm/README.md`
- Modify: `docs/miniglm/02-training-runbook.md`

- [x] Make `runbook/00-preflight-and-current-smoke.md` the only execution entry.
- [x] List the five runbook chapters in order.
- [x] Remove conflicting "reading order" versus "execution order" branches.

### Task 4: Verify Documentation

**Files:**
- Check: `docs/miniglm/**/*.md`

- [x] Check Markdown code fences.
- [x] Check bash code block syntax.
- [x] Check Python heredoc syntax.
- [x] Check local links.
- [x] Check no old runbook filenames remain in Mini-GLM docs.
- [x] Run `git diff --check`.
