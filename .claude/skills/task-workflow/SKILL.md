---
name: task-workflow
description: Enforce one-commit-per-TODO-item workflow and required collateral updates (README/help/tests/CHANGELOG) for arrayview feature/fix tasks.
---

# ArrayView Task Workflow Skill

## Purpose

Use this skill whenever implementing or fixing a TODO item. It enforces a disciplined workflow so each TODO becomes a single, self-contained commit (and ideally a single PR) that also updates documentation, help text, and tests.

## Rule

Every completed TODO item must:

- Be implemented in a single commit with a clear commit message (see Commit Message format below).
- Include or update automated tests that validate the feature/fix where possible.
- Update `README.md` or the in-app help overlay if usage or shortcuts changed.
- Update `tests/visual_smoke.py` for UI/layout changes (add a numbered scenario and screenshot capture).
- Add a short entry to `CHANGELOG.md` or `AGENTS.md` (if CHANGELOG.md doesn't exist, add to `AGENTS.md` under a "Changelog" or the Skills section).

If any of these steps cannot be completed (e.g., untestable UI dialog that requires human review), the implementer must document the reason in the PR description and add a smoke-test TODO row in `tests/visual_smoke.py` with `✗ (reason)`.

## Commit Message Format

- Use a concise, searchable prefix describing the TODO ID or short label, then a clear subject, then an optional body.
- Recommended: `todo: <short-label>: <one-line-summary>`
- Example: `todo: picker-two-column: add two-column compare picker and shape-filtering`

Include a bullet list in the commit body showing collateral updates, e.g.:

- Tests: `tests/test_picker.py` and `tests/visual_smoke.py#NN`
- Docs: `README.md` updated section "Compare Picker"
- Changelog: `AGENTS.md` / `CHANGELOG.md`

## Checklist (enforced by this skill)

For each TODO item, ensure the following before marking the task done:

- [ ] Code implements the feature/fix and passes `python -m mccabe`/lint checks (if configured).
- [ ] Unit tests or API tests added/updated in `tests/`.
- [ ] If UI changes, `tests/visual_smoke.py` updated with a numbered scenario and `_shot()` call.
- [ ] README or in-app help overlay (`#help-overlay` content in `src/arrayview/_viewer.html`) updated if usage changed.
- [ ] `CHANGELOG.md` or `AGENTS.md` updated with a one-line summary.
- [ ] Commit message follows the format above and lists affected files.
- [ ] **Tests MUST be run by the agent — NEVER ask the user to run them.**
  - After every code change, run the relevant test suite immediately:
    - Backend/API changes → `uv run pytest tests/test_api.py -q`
    - Viewer UI changes → `uv run pytest tests/test_interactions.py tests/test_mode_consistency.py -q`
    - CLI changes → `uv run pytest tests/test_cli.py -q`
  - If any test fails, fix it before considering the task done.
  - Do NOT end a task with "run this to verify" — run it yourself and report results.

## How to use

1. Create a branch for the TODO item: `git checkout -b todo/<short-label>`.
2. Implement the change and add tests and docs as required.
3. Run tests and smoke script locally until passing (or document failures).
4. Stage and commit only the files relevant to this TODO with a single commit message as defined above.
5. Push and open a PR.

## Special cases

- If a TODO necessarily spans multiple commits (e.g., large refactor), use a temporary feature branch with descriptive commits, then squash into a single commit before merging. The PR must include a description explaining why squashing is required and ensure tests/docs are included in the final squashed commit.

- If a fix must touch unrelated files (rare), keep those touches minimal and include an explanation in the commit body.

## Automation hints for maintainers

- Prefer adding a simple CI check that asserts commits touching `src/arrayview/` are accompanied by changes in `tests/` or `README.md`. This can be implemented as a lightweight script in `.github/workflows/`.

- Encourage using `git commit --no-verify -m "..."` only when CI is temporarily failing for reasons unrelated to the task; prefer fixing CI first.


## Red Flags — STOP

- "I'll do tests later" — must add tests or justify why not and add smoke test TODO entry.
- "I'll bundle multiple unrelated TODOs into one commit" — split them or squash locally, but final PR should present one item per commit semantics.
- "Docs unchanged although usage changed" — update README/help or explain in PR why docs remain unchanged.


# End of skill
