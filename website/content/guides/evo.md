---
title: Evo foundation
description: Run reproducible coding tasks with frozen skills and independent verification.
---

Tau-Evo adds an opt-in experiment layer around Tau's existing CodingSession.
It selects reviewed skills, freezes their contents for one task, records tool use
and provider token usage, and runs a host-owned verifier after the solver finishes.
The portable agent loop and provider protocols remain reusable.

This is the foundation for Evo-Harness experiments. Automatic failure reflection,
skill compilation, held-out promotion gates, and source-code self-modification
are future phases. The included smoke skill is a human-authored fixture.

## Install and verify DeepSeek

From the repository root:

```bash
uv sync --dev --locked
uv run tau-evo --store .tau-evo smoke --model deepseek-flash --api-key-file API_KEY.txt
```

The credential file can contain a raw key or a single
`DEEPSEEK_API_KEY="..."` assignment. It is read only into memory. Alternatively,
set `DEEPSEEK_API_KEY`; the CLI removes it from its environment after reading it.
Do not place the credential file in the solver's task workspace.

The smoke command creates a temporary Python project, asks the real model to
read and fix a function with Tau tools, and checks the result in a separate
Python process. Exit code zero requires successful verification and actual tool
use. The temporary project is deleted; audit records remain under
`<store>/smoke/runs/<run-id>/`. Its fixture snapshot lives in a separate smoke
namespace and does not replace the main store's skills.

## Review and activate skills

```bash
uv run tau-evo --store .tau-evo status
uv run tau-evo --store .tau-evo publish examples/evo/skills.json --expected empty --reviewed
```

A snapshot replaces the complete skill set. Review the JSON before using
`--reviewed`. Future updates must pass the version returned by `status` as
`--expected`; stale writers fail rather than silently replacing another version.
The command reports the activated content hash.

```bash
uv run tau-evo --store .tau-evo rollback <previous-version> --expected <current-version>
```

`empty` is the built-in empty version. Rollback changes the current pointer and
retains older snapshots and run records. An interrupted publisher can leave
`writer.lock`; inspect whether its writer is still running before removing it.

## Run a coding task

Prepare a dedicated task workspace and a trusted verifier outside that workspace.
The prompt file contains the user task only. Use absolute paths where indicated.

```bash
uv run tau-evo --store .tau-evo run --model deepseek-flash --api-key-file API_KEY.txt --cwd <absolute-task-workspace> --task-id example-001 --prompt-file <task-prompt.txt> --verify python <absolute-verifier.py>
```

`--verify` must be last; all remaining arguments belong to the verifier. It runs
in the task workspace without shell interpolation. Exit zero means passed,
nonzero means failed, and spawn errors or verifier timeout mean unknown. A model
response saying “tests passed” cannot set this result.

The solver has `read`, `write`, and `edit` by default. Add `--allow-shell` to
enable Tau's `bash` tool. The workspace and audit store must be separate.
Filesystem tools, extensions, and subprocesses retain the user's OS privileges;
use a container or another OS isolation mechanism for untrusted tasks. A working
directory and a separate verifier process do not enforce filesystem isolation.

`--max-turns` (default 12) and `--timeout` (default 180 seconds) bound each
experiment. The wall-clock deadline includes session preparation and verification;
resource cleanup follows it. Cancellation and timeout leave a final run record.
The runner accepts a caller-owned provider and verifier in Python, so experiments
are not coupled to the DeepSeek CLI.

## Selection and provenance

Selection is deterministic: matching trigger substrings rank by count, then
stable skill ID; general skills can match any task. The limit is three skills
and 8,000 UTF-8 bytes of rendered skill bodies by default. This is an exact byte
budget, not a claim about model tokenization. Use `--max-skills` and
`--max-bytes` to change it, or zero to run a no-skill baseline.

Each task receives a fresh CodingSession with ambient skill discovery and
filesystem extension discovery disabled. Reviewed guidance is injected into the
system prompt; the original task text remains intact. Each record contains:

- Exact selected skill bodies, snapshot version, and selection budgets.
- Prompt, effective system prompt, and tool-schema hashes.
- Model/provider identity, Tau version, task ID, time and turn limits.
- Execution status, tool-call count, token usage and external verification.
- A durable Tau session with a `tau.evo` custom entry linking the run.

The local `system.txt` and `session.jsonl` contain full task context and tool
outputs. Treat them as private experiment data; only publish reviewed summaries.
Credentials are never added by the runner, but arbitrary task text or tool output
can contain sensitive content. `.tau-evo/`, `API_KEY.txt`, and `.env` files are
ignored by this repository. Provider cost fields may be zero when prices are
unavailable; this does not mean the request was free.

The initial store uses a single-writer lock and atomically replaces files.
An interrupted run marked `running` must be inspected, not automatically replayed:
tool calls can already have produced side effects.

## Optional extension for normal Tau sessions

Set `TAU_EVO_HOME` to the absolute path of a reviewed skill store, then run:

```bash
uv run tau -e src/tau_coding/extensions/evo_harness
```

The extension wraps the first ordinary interactive prompt with selected guidance
and writes `tau.evo.selection` and `tau.evo.settled` session entries. Explicit
slash commands/skills, extension-originated prompts, and steering/follow-up
messages are left to Tau. Settled is recorded as unverified.

Use a fresh session for each independent task. The extension freezes its first
selection for that runtime generation; a resource reload starts a new generation
but does not erase guidance already in conversation history. Use the experiment
runner for clean comparisons and external verification.

## Deterministic tests

```bash
uv run pytest -q tests/test_evo.py
uv run ruff check .
uv run mypy --follow-imports=silent src/tau_coding/evo src/tau_coding/extensions/evo_harness
```

The Evo CI workflow runs offline tests on both Windows and Linux. Live model
verification is explicitly invoked and never consumes API credentials in CI.

