# Experiment Run Board

A dashboard on claude.ai that shows how an experiment is going: one card per arm (status,
answers in, failures, cases per hour, prompt and prefill throughput, expected finish,
accuracy), answers completed over time, prompt throughput per 15 minutes, accuracy by
group once judging ends, health checks, and the arms' settings.

Template: https://claude.ai/artifact/4dMc9EAPDdfDCSwCfvuUGc (first used for
`artifacts/lme500-layerwise-20261113`).

The page reads only four files, so it draws any experiment written in this schema.
Nothing on the page names an experiment, an arm or a metric that the files don't supply.

## The four files

`snapshot.py SPEC.json OUT_DIR [--unblind]` writes them. Each is `{"rows": [...], "meta": [...]}`;
`meta` carries the script's own text so a reader can see how the numbers were made.

| File | One row per | Columns |
| --- | --- | --- |
| `runs.json` | arm | `experiment, title, arm, label, status` (running, queued, done, stopped), `total, completed, errors, started_at, last_finished_at, snapshot_at, eta_at, elapsed_hours, cases_per_hour, prompt_tok_s, output_tok_s, step_prefill_tok_s, mean_output_tokens, output_cap_hits, graded, judged_questions, accuracy` |
| `cases.json` | finished or failed request, per arm | `arm, id, group, status, started_at, ended_at, seconds, prompt_tokens, output_tokens, finish, correct` (0, 1 or null) |
| `checks.json` | check, per arm | `arm, check, value, target, state` (ok, warn, bad, info) |
| `settings.json` | setting, per arm | `arm, setting, value` |

Times are ISO 8601 in UTC. `title` and `snapshot_at` are read from the first `runs` row.

## A new experiment

1. Write a spec (see the docstring of `snapshot.py`; `artifacts/lme500-layerwise-20261113/board.json`
   is an example). Arms point at directories of `cases/<id>/result.json` records with
   `status, started, ended, prompt_tokens, usage.completion_tokens, finish_reason`. An arm
   with `kvmem_capture` also gets Query/pressure decision checks from its server capture.
   A harness with another layout writes the four files itself instead.
2. `python benchmarks/board/snapshot.py SPEC.json OUT_DIR`.
3. Ask Claude for a run board from the template: it copies the template's page and four
   dataset entries into a new dashboard and points them at the uploaded files.

## Refreshing

The files are copies, so the board changes only when a new snapshot is uploaded: run
`snapshot.py` again, upload the four files to the dashboard, and point each dataset at
its new file (Claude does all three when asked to refresh the board). The page marks a
snapshot older than two hours as stale.

## Grading

Accuracy stays empty until judging ends. `--unblind` adds each case's grade from the
grading directory (`key.json` from the blind A/B sheet, `judgments.jsonl`); before that only
the number of judged questions is shown, so the board never unblinds the judge.
