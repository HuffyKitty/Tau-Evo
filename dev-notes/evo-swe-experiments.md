# Evo SWE-bench controller

This phase adds official grading and automatic experimental skill learning around
the existing frozen Tau runner. `tau_agent` and `tau_ai` stay unchanged.
The [usage guide](../website/content/guides/evo-swe.md) contains reproducible commands.

## Design

- `benchmark_cli.py`: opt-in dataset preparation, DeepSeek configuration, preflight and commands.
- `provenance.py`: solver AST, experiment manifest, and grader image checksums.
- `swe.py`: isolated solver shell, patch extraction and unchanged official Linux grading CLI.
- `experiment.py`: sequential streams, per-batch snapshots, checkpoints, rescoring and paired summaries.
- `learning.py`: bounded evidence extraction, Reflect and Compile with explicit JSON contracts.
- `evidence.py`: credential-pattern redaction and bounded text for feedback and learning inputs.

SWE-bench 5.x's canonical dataset includes the official image, evaluation script,
parser and evaluation type. The older dataset schema lacks these fields. We pin
`swebench==5.0.2`, whose aggregate report is named `tau-evo.<run-id>.json` in the
working directory. We parse this released version's schema, not a moving Git branch.
The grader Docker image is built once and pinned by image ID for both arms.

The solver and the official test runner use separate containers. The solver sees
the issue and repository only. Reflect sees bounded trajectory, failed test names
and counts, and host-owned evidence hashes; it never sees reference or test patches.
Compile sees only existing skills and lessons from the just-completed batch. Each
change must cite a current failed candidate; MERGE/REVISE may also cite evidence
already attached to that target skill. Unknown or future IDs remain invalid.
Selection for a task is recorded as
full skill contents, so proving injection does not depend on a mutable library.

Learning remains outside the portable agent core, matching Pi's separation of the
brain and its environment. It uses the same provider abstraction and a tool-free
one-turn harness rather than a separate model-client implementation.

## Restart and failure semantics

One experiment lock serializes stream writers. A completed task is never solved
twice on resume. A completed compiler candidate can be activated idempotently after
a crash; compare-and-swap prevents replacing an unexpected active library. An
interrupted task with unknown tool side effects requires inspection. A scorer-only
failure can be retried using the original patch without another solver call.
An ambiguous official test timeout can additionally run a same-image, same-limit
reference-patch control. A passing control diagnoses the candidate timeout as a
negative result; a failed or unknown control leaves it unknown. Both official
evidence hashes are kept, and the reference code never enters learning context.
Completed official reports survive a later driver shutdown timeout.
Grading calibration is allowed only before the first learned batch completes.
It rescores every saved patch from both arms, preserves the original manifest and
reports, records a separate grading policy, and blocks solver continuation until
regrading finishes. Solver model calls, patches and budgets are retained.

Invalid compilation is atomic: no partial skill set becomes active. Experimental
activation is distinct from production `publish(reviewed=True)`; it does not forge
human approval. There is no automatic promotion into the user's normal skill store.
Malformed reflection/compilation responses stop the batch and record completed
provider usage. Learning calls are not retried implicitly.

## Verification and limits

Offline tests cover evidence grounding, revision merges, library limits, production
store refusal, failure-to-next-batch injection, no learning from success/unknown,
resume without replay, manifest checksums, official report paths, Linux grading
invocation, and scorer-only retry. Live model and Docker runs remain opt-in.

The first Windows-native official scorer preflight failed because its generated
shell script used CRLF. The same reference patch passed after moving the unchanged
scorer into Linux. This is infrastructure validation, not a model result.

An initial two-task diagnostic run exhausted turns without patches while searching
for upstream answers. Network and host-artifact isolation blocked those searches.
It was stopped, retained locally as a pilot, and excluded from the formal arms.
Before restarting both arms, the common solver prompt was fixed to require current
repository diagnosis and prohibit upstream/gold/hidden-artifact searches. Task
selection, model, and budgets were not retuned based on pass rates.

Only solver limits are matched. Learning has additional explicitly counted calls;
there is no claim of equal total compute or of established RSI benefit. This phase
has no independent held-out promotion gate, repeated-seed study, broad repository
coverage, or source-code self-modification. Those remain prerequisites for stronger
claims and later scaling decisions.
