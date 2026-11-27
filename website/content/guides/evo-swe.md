---
title: SWE-bench skill experiments
description: Compare a frozen Tau solver with failure-driven skills using official grading.
---

This opt-in pipeline implements the Evo-Harness sequence: select, inject,
execute, reflect, compile, update. The solver model and implementation stay
fixed. Only the external skill library evolves between batches. This is a
method adaptation using Tau and DeepSeek, not a reproduction of the paper's
original model, tool stack, or reported scores.

## Prepare a frozen sample

Start Docker with Linux containers, then run from the repository root:

```bash
uv sync --dev --extra benchmark --locked
docker build -t tau-evo-grader:5.0.2 -f src/tau_coding/evo/grader.Dockerfile src/tau_coding/evo
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 prepare --repo sympy/sympy --limit 10 --batch-size 2 --model deepseek-flash
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 preflight
```

`prepare` downloads `SWE-bench/SWE-bench_Lite` at a recorded Hugging Face
revision. It selects the first ten lexicographically sorted IDs in one repository,
before outcomes are observed. The manifest freezes the task data checksum, order,
model, batch size, and execution limits. A single-repository sample makes local
pipeline checks practical; it is not representative of SWE-bench Lite overall.
Pass `prepare --revision <dataset-commit-sha>` to recreate a recorded dataset revision.

The first `preflight` or `run` also pins the solver implementation hash, manifest
hash, and Linux grader image ID. Subsequent calls refuse changes. Use a new
experiment directory after changing the solver or budgets. The other arm reuses
each task's recorded registry image digest, avoiding mutable-tag drift.

`preflight` evaluates the first task's reference patch with the official scorer.
It consumes no model tokens and is never counted as a solver result. The reference
patch and test patch remain in the host-side grading data; the solver and reflector
receive neither. Images can take several minutes to download and require several
GB each. Tasks run sequentially; newly pulled task images are removed after use.
Existing images and unrelated containers are retained.

## Run both arms

```bash
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 run --arm frozen --api-key-file API_KEY.txt
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 run --arm evolving --api-key-file API_KEY.txt
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 summary
```

Alternatively set `DEEPSEEK_API_KEY`. The key is read into the host process and
never mounted in either task or grading containers. Both arms use the same
problem statements, model, task order, tool, and per-task solver limits:

| Limit | Default |
| --- | --- |
| Model turns | 20 |
| Solver wall time | 300 seconds |
| Output tokens per model call | 4,096 |
| One shell command | 60 seconds |
| Official evaluation | 180 seconds |
| Selected skills | At most 3, within 8,000 rendered UTF-8 bytes |
| Compiled library | At most 6 skills, within 16,000 body bytes |

Sampling uses the provider defaults in both arms. Model aliases and stochastic
sampling prevent a claim of bit-for-bit model reproducibility. Token usage includes
provider-reported cached tokens; reported cost zero can mean pricing is unavailable.
The limits above are equal **solver budgets**. Reflect and Compile make extra model
calls, measured separately; this is not a total-compute-matched comparison.
Use the reported learning overhead when planning a later cost-matched experiment.

The frozen arm always uses the empty snapshot. The evolving arm begins empty,
reflects on externally failed tasks, and compiles their lessons after each batch.
Tasks within one batch all receive the snapshot captured at the batch start.
Neither successes nor infrastructure errors generate lessons. The last batch
does not learn, since this experiment has no subsequent tasks.

Compile supports ADD, MERGE, REVISE, and SKIP. Its JSON must cite failed run IDs
provided by the host. Unknown evidence, conflicting actions, invalid IDs, or
oversized libraries reject the entire candidate. A candidate is activated only
in that arm's `experimental-skills` store with a compare-and-swap version check.
This is experimental activation, not human review or held-out validation. Normal
`tau-evo publish` retains its explicit reviewed-publication requirement.

## Isolation and official scoring

The solver has one shell tool inside a fresh task container. It has no host
mounts, no Docker socket, and no network. It runs with two CPUs, a 3 GB memory
limit, and a process limit. The host extracts a patch, then a separate trusted
Linux grader invokes the unchanged `swebench==5.0.2` evaluation CLI. This avoids
Windows CRLF corruption of the official Linux evaluation scripts.

The grader needs the Docker socket to create official evaluation containers;
it is trusted infrastructure, not a tool available to the model. It mounts only
its grading directory. Reference patches, grading scripts and test patches are
kept out of the solver container. Solver instructions prohibit searching for
upstream solutions, hidden tests, or benchmark artifacts. Container isolation
does not address possible benchmark contamination in model training data.

Official JSON reports decide pass/fail. Empty patches are recorded as failures.
Solver timeout or turn-limit exhaustion can still leave a partial patch, which
is evaluated normally. Missing reports, scorer process errors and classified
infrastructure failures stop the stream with an unknown verdict, without learning.

## Inspect and resume

The experiment directory contains:

- `experiment.json`, `tasks.json`, `protocol.json`: frozen inputs and implementation identity.
- `<arm>/tasks/<id>/`: image digest, model patch, official reports, final run and any lesson.
- `<arm>/experimental-skills/runs/`: complete Tau sessions and selected skill contents.
- `<arm>/batches/<n>/`: input snapshot, compiler audit, candidate and completion marker.
- `summary.json`: paired results, selected skill IDs, solver time/tokens and learning tokens.

Repeating `run` skips completed batches and task results. An interrupted task
without a result is not silently replayed. Inspect its artifacts and process
state first. A stale `stream.lock` likewise requires inspection before removal.
The lock serializes both arms. Malformed learning responses stop before activation;
there are no hidden model retries or free repair calls.

When only grading failed, preserve the original solver patch and retry scoring:

```bash
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 rescore --arm frozen --task-id <instance-id>
uv run --extra benchmark tau-evo-swe --experiment .tau-evo/swe-lite-10 run --arm frozen --api-key-file API_KEY.txt
```

`rescore` accepts only unknown verdicts, uses the pinned image and saved patch,
and consumes no model tokens. It keeps prior grading logs in place.
Full local artifacts may contain private task text, model reasoning, or logs;
review summaries before publication. The `.tau-evo` directory is ignored by Git.

## Interpret a small experiment

A failure producing a cited skill that is injected into the next batch proves
the data path works. A pass-rate difference across ten tasks does not establish
RSI effectiveness. Inspect paired improvements and regressions, skill use and
learning overhead. Before expansion, add multiple repositories, repeated seeds
or runs, an independent held-out gate, and an explicitly cost-matched control.
