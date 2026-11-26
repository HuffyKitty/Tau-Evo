# Evo foundation implementation

This phase provides P1's execution and evidence boundary plus manually reviewed
snapshot activation. It is not the automatic P2/P3 evolution controller.

The implementation uses a fresh `CodingSession` per experiment and the existing
`prepare_coding_session` adoption boundary. No edits to `tau_agent`, `tau_ai`,
or `session.py` were necessary. An optional ordinary extension exercises Tau's
input and settled hooks; the experiment runner uses explicit system prompt
assembly to preserve user input and avoid previous-task context leakage.

Source map:

- `tau_coding/evo/models.py`: schema v1, immutable snapshots, run/feedback contracts.
- `store.py`: content-addressed snapshots, stale-writer checks, atomic activation/rollback.
- `selection.py`: deterministic matching and exact UTF-8 byte budgets.
- `runner.py`: frozen CodingSession, persisted provenance, bounded execution and cleanup.
- `verifier.py`: host-owned subprocess verification; solver output never decides success.
- `cli.py`: reviewed snapshot management, DeepSeek runner and real coding smoke.
- `extensions/evo_harness/extension.py`: opt-in frontend integration.
- `tests/test_evo.py`: offline state, selection, execution, verifier and extension regressions.

Read the [usage guide](../website/content/guides/evo.md) for commands and boundaries.
The original [design](rsi-improvement-plan.md) remains the roadmap; automatic
reflection/compilation and held-out promotion require later implementation.

## Verification on 2026-11-25

DeepSeek's official models endpoint authenticated successfully and returned
`deepseek-flash` and `deepseek-v4-pro`. The real `deepseek-flash` smoke run
`244833ca7aa94378aa81b58933f0bb91` completed four model turns and three tool calls.
The independent verifier passed fractional, negative, singleton and empty-input
checks. Provider-reported total usage was 7,844 tokens. This demonstrates model,
tools, skill injection, persistence, and external verification integration;
it does not demonstrate improvement from learned skills.

The human-authored fixture snapshot was
`30faa4f1c2857dd0bd5b82142949d40c200b754f5ce138b26559b1eb064fafc8`.
Raw transcripts and credentials are local ignored artifacts. No API key or local
user path is included in this report.

## Regression validation and integration

The initial implementation's [complete Linux CI run](https://github.com/HuffyKitty/Tau-Evo/actions/runs/45639210553)
passed 2,260 tests with three skips, full Ruff checks, formatting, mypy, and the
website type/build/test checks. The [Evo matrix](https://github.com/HuffyKitty/Tau-Evo/actions/runs/45639210297)
passed on both Windows and Linux. A final audit regression preserves provider
usage when a later tool is interrupted; the local Evo suite now has 15 passing tests.

On the development Windows machine, the full upstream test run encountered
existing failures and stopped advancing at 92%; it was terminated after an
extended stall. The previous Windows baseline and known platform problems remain
documented in the design plan. This incomplete run is not reported as passing.
Local website type checking and build passed; its route test still assumes POSIX
path separators. The same website checks pass in Linux CI.

The upstream documentation deployment and PyPI publication workflows now run only
for the original upstream repository. This fork retains ordinary build/test CI
and does not automatically publish the upstream package name or site.

