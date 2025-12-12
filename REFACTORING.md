# Refactoring Guidelines

This document defines the contract and discipline for all refactors in this codebase.

## Core Principles

1. **Tests are the scientific contract.**
   - Never modify tests unless explicitly requested.
   - If a test fails, the code is wrong—not the test.

2. **Small, reviewable diffs.**
   - One theme per PR (e.g., "unify output paths" or "rename X to Y").
   - Avoid mixing cleanup with new features.

3. **No new abstractions unless they delete more code than they add.**
   - Prefer explicit, readable code over clever indirection.
   - If a helper doesn't reduce duplication, don't add it.

4. **Behavior changes only when a test fails or a known bug is documented.**
   - Refactors must preserve semantics.
   - Small fixes are allowed if tests/plots confirm correctness.

## Outputs Policy

| Directory   | Git Status      | Purpose                                      |
|-------------|-----------------|----------------------------------------------|
| `outputs/`  | **Untracked**   | Generated artifacts (plots, pickles, logs)   |
| `results/`  | **Committed**   | Publication-ready results, curated outputs   |

- All generated files go under `outputs/<analysis_name>/`.
- Never commit raw analysis outputs; only curated results belong in `results/`.
- Every saved artifact should include metadata (seed, version, key params).

## PR Workflow

1. **Identify** the duplication or hotspot.
2. **Propose** a minimal change (explain current structure first).
3. **Make** the change in a small diff.
4. **Run** focused tests to verify behavior is unchanged.
5. **Stop** once the single theme is complete.

## Reproducibility

- The random seed is set via `seed_random()` in `src/model/config.py`.
- Do not introduce new sources of nondeterminism (time-based seeds, unordered sets, etc.).
- New code should accept an optional `rng` parameter; defaults preserve current behavior.

## See Also

- `.cursor/rules/` — Agent-facing rules that enforce these principles automatically.
- `docs/dev_style.md` — Local formatting/linting instructions.

