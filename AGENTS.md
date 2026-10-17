# Agent Guidelines

This repository is a research project started from poorin-lite, a template carrying Poorin's
reproducibility and experiment-tracking layer. `README.md` describes the workflow: artifacts,
versions, the launcher, run references, and evaluation.

## Development

```bash
# Tests (CPU)
uv run pytest

# Lint and format
uv run ruff check --fix . && uv run ruff format .

# Type checking: errors and warnings fail
uvx ty@0.0.84 check --error-on-warning
```

- Python >= 3.13.
- Dependencies may be upgraded freely. Each dependency change is its own commit, and its
  `uv.lock` diff is checked for versions that moved without being asked.
- Commit before running a calendar-versioned experiment: `--run` refuses to build one from
  uncommitted code. A dev run saves uncommitted code as a snapshot commit.
  `uv run python -m repro.experiment.runs show NAME@VERSION` shows the code behind a run.

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- Commit only when asked. Never push unless asked.
- Commit titles are imperative, at most 72 characters, with a `[scope]` prefix such as
  `[repro]` or `[project]`. The body states what changed and why.
- NEVER credit yourself in commit messages: no `Co-Authored-By` trailer, no "Generated with …"
  line, no emoji attribution, even if a tool default suggests one.

## Code Style

- All imports at the top of the file. No local imports except to break circular dependencies or guard optional deps. No `TYPE_CHECKING` guards — fix cycles structurally via protocols.
- Prefer top-level functions over classes when code does not mutate shared state. Reduce deep inheritance hierarchies.
- Use early returns to reduce nesting.
- Document public APIs with concise Google-style docstrings. Skip docstrings on trivial functions with clear names.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer logging over `print` (except in scripts and debugging).
- Resolve environment-dependent defaults once and fail fast on unknown inputs.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`); update code consistently.
- Prefer small concrete helpers over abstraction that adds indirection without reuse. Start simple; abstract only under real pressure.
- Delete dead code: unused parameters, stale options, old experiments.
- Top-level constants for magic strings/numbers.
- Separate computation from I/O (split compute from upload/write).
- Use context managers for resource lifecycle.

## Naming

- No `*_utils.py` — use descriptive names like `text_cleaning.py`.
- Function names should reflect return types (`probe_task` → `task_status`).
- No `_s` suffix for seconds (assumed in this codebase). No abbreviations like `exe` — use `exec` or full words.

## Types & Data Structures

- Dataclass/namedtuple over raw dicts. `StrEnum` over string keys.
- Use `Protocol` for decoupling; avoid hard-coupling to concrete types.
- Avoid `X | str` unions that require `isinstance` checks — pick one input type.
- Replace compound booleans encoding state with an enum.

## Configuration

- No `default_*` wrappers that obscure underlying mechanisms.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location.
- Prefer explicit constructor/config parameters over env vars.
- Composition over inheritance: embed sub-configs, don't subclass.
- A value that changes results belongs in the experiment's config, where `--set` reaches it and
  the fingerprint records it; a click option or runtime arg is for how a run executes.

## API Design

- Accept only what's necessary. Replace boolean flags with meaningful parameters (e.g., `num_workers: int` instead of `parallel: bool`).
- Use separate classes over boolean flags for variant behavior.
- Normalize inputs to a standard format once at the boundary, not throughout.

## Error Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter control flow.
- NEVER swallow exceptions unless specifically requested.
- Assert liberally; prefer `raise ValueError` over silent fallbacks.

## Deprecation

**NO BACKWARD COMPATIBILITY**: Update all call sites instead. Only add compatibility shims if the user explicitly requests it.

## Comments

- Write comments for module/class-level behavior or subtle logic. Do not restate the code.
- Delete stale comments immediately on discovery.
- Inline comments to clarify non-obvious boolean arguments.

## LLM-Generated Code Pitfalls

Watch for and eliminate these patterns in generated code:
- Over-protective try/except and defensive None checks
- Tautological and guard-mirroring tests (type exists, constant has value, nonpositive scalar
  raises without a regression or public compatibility contract)
- Verbose/redundant docstrings and `__all__` in `__init__.py`
- Boolean dispatch instead of separate classes
- Environment variables instead of explicit parameters

## Code Reuse

Before writing any utility function, helper, or data structure, search `repro/` and
`src/project/`, then `pyproject.toml` for available third-party packages. If a suitable
implementation exists, use it. Do not create parallel implementations.

`repro/` must not import from `src/` or `experiments/`; project code imports `repro`, never the
other way.

## Testing

- Test behavior, not implementation: a test names the change that would make it fail.
- No tautological tests that restate a constant or a type.
- Prefer real objects and small fakes over mocks; never assert on a mock's own behavior.
- No sleeps to wait for work; synchronize on events, barriers, or the result.
- Always fix tests you broke. Do not relax tolerances or hack around failures.
