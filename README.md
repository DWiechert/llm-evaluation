# llm-evaluation

A small local LLM evaluation harness for models served by [Ollama](https://ollama.com),
scored with rule-based/code checkers (no LLM-as-judge) and tracked in
[MLflow](https://mlflow.org), with a flat CSV/SQLite export for your own analysis.

## Requirements

- [Ollama](https://ollama.com) running locally (`http://localhost:11434`) with at least
  one model pulled (`ollama pull <model>`)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

## Usage

```bash
uv run mlflow_eval.py --models ornith:9b llama3.1:8b qwen2.5-coder:7b
```

Then browse results:

```bash
uv run mlflow_ui.py
# open http://localhost:5000 and browse the "local-llm-eval" experiment
```

(`mlflow_ui.py` just wraps `mlflow ui --backend-store-uri sqlite:///results/mlflow.db`
— plain `mlflow ui` defaults to a tracking DB at the repo root instead, which this
project doesn't use.)

Everything lands under `./results/` — nothing is written to the repo root:
MLflow's own tracking DB (`mlflow.db`) and artifact store (`mlartifacts/`), plus a
flat CSV, SQLite export, and a system-info snapshot (CPU/GPU/RAM/Ollama version, for
comparing runs across machines).

### Options

| Flag | Default | Description |
|---|---|---|
| `--models` | *(required)* | One or more Ollama model tags to evaluate |
| `--experiment` | `local-llm-eval` | MLflow experiment name |
| `--out` | `results` | Output directory for CSV/SQLite export |
| `--all` | | Run every category (default when no category flag is given) |
| `--basic` | | Run only `basic_questions` |
| `--tools` | | Run only `tool_usage` |
| `--coding` | | Run only `coding` |
| `--finance` | | Run only `finance` |
| `--reasoning` | | Run only `reasoning` |
| `--instructions` | | Run only `instruction_following` |
| `--design` | | Run only `design` |

Category flags are additive, so `--basic --tools` runs both. Omitting all of them
is equivalent to `--all`.

### Reading the MLflow UI

All models/categories from one script invocation share a single MLflow run (so the
Run ID/name — a short run_id hex string — stays constant instead of changing per
model). To tell results apart:

- **Run list**: each run has `models` and `categories` tags summarizing what that
  invocation covered — enable them via the Columns picker if they're hidden.
- **Traces/Evaluation table**: every individual sample is tagged with `model` and
  `category` — enable those columns to filter/sort/group per-model results instead
  of only seeing aggregate scores.

## What's scored, and how

Every category is scored with deterministic, rule-based checkers — no LLM judge. See
`eval_dataset.py` for the full dataset and `mlflow_eval.py` for the scorer implementations.

| Category | How it's scored |
|---|---|
| `basic_questions` | Exact/substring match against a known-correct short answer |
| `tool_usage` | Structural check: right tool name + required arg keys present (does not validate argument *values*) |
| `coding` | The model's generated code is **actually executed** against test cases in a subprocess — real pass/fail |
| `finance` | Numeric answer extracted from the response and checked against a value computed with plain Python |
| `reasoning` | Same numeric-extraction approach as `finance` — these are riddles with one unambiguous computed answer |
| `instruction_following` | Concrete, machine-verifiable constraints (sentence/bullet counts, word limits, forbidden words, required JSON keys) |
| `design` | **No deterministic scorer exists** for open-ended system design quality. Included for side-by-side manual review only — the attached score is a keyword-coverage checklist, not a quality grade |

Every row also captures timing/throughput metrics pulled directly from Ollama's own
response fields (not estimated): wall-clock latency, decode tokens/sec, prefill
tokens/sec, and a best-effort VRAM delta via `nvidia-smi`.

## Files

- `mlflow_eval.py` — main entrypoint: runs the dataset against Ollama, scores it,
  logs to MLflow, exports CSV/SQLite
- `eval_dataset.py` — the eval cases (prompts, tool schemas, expected answers/test
  cases) consumed by `mlflow_eval.py`
- `mlflow_ui.py` — launches `mlflow ui` pointed at `results/mlflow.db`
