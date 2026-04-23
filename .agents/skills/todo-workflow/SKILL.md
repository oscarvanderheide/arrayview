---
name: todo-workflow
description: Use when working through TODO items, implementing batches of features or fixes, or when the user gives multiple tasks at once. Enforces commit-per-item, collateral updates, and cross-mode verification.
---

# TODO Workflow

## Overview

When working through multiple tasks (TODO items, feature requests, bug fixes given together), follow these rules for every item. The TODO list lives in `dev/TODO.md`.

## Per-Item Rules

Each finished item gets:
1. **Its own commit** — one item, one commit, no batching
2. **Updated tests** — add/update test coverage for new functionality
3. **Updated README** — if user-facing behavior changed
4. **UI audit** — run ui-consistency-audit skill to verify across all modes
5. **Invocation check** — use invocation-consistency skill if touching server/startup/display-opening
6. **VS Code check** — use vscode-simplebrowser skill if touching extension install, signal-file IPC, or auto-open logic. This breaks often.
7. **Lessons learned** — update `dev/lessons_learned.md` with anything important for future sessions

## Execution Rules

- **Plan first** — write a plan before starting implementation
- **Subagents** — spawn them for independent items to parallelize work
- **Branching** — work on `main` unless parallelizing; if branches needed, rebase (no merge commits)
- **Compact context** — clear/compact between items to stay sharp
- **Skills** — use and update relevant skills, especially ui-consistency-audit

## Design Philosophy

The app is feature-rich but minimal — no clutter. Users should discover features and think: *"wait... it already does that?!"* Every feature should feel like a hidden gift, not visual noise.

## When User Says They're Going to Sleep

Don't ask for confirmation. Make your own decisions on remaining items. They expect to be impressed when they wake up.
