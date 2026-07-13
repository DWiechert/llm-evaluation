# Contributing

## Adding a question to an existing category

Each category is a plain Python list in its own file under `src/datasets/`
(e.g. `basic_questions.py`, `finance.py`). Append a new entry following the
shape of the existing ones in that file:

```python
{
    "inputs": {"id": "basic_questions_4", "category": "basic_questions", "messages": [
        {"role": "user", "content": "..."}
    ]},
    "expectations": {"expected_substring": "..."},
},
```

- `id` must be unique within the file — it's used to identify the row in
  MLflow traces and results exports.
- `expectations` is checker-specific — copy the shape from a neighboring
  entry in the same file, or see the matching `check_*` function in
  `src/mlflow_eval.py` for what keys it reads (e.g. `check_numeric_answer`
  for `finance`/`reasoning`, `check_instruction_following` for
  `instruction_following`).
- `tool_usage` entries also need a `"tools"` key (see `TOOLS` in
  `tool_usage.py`).

That's it — no other file needs to change. Adding/editing entries changes
that category's content hash (`CATEGORY_HASHES` in `src/datasets/__init__.py`,
stamped into exported results per issue #8) — that's expected, it's what lets
old and new runs be told apart, not a bug to work around.

## Adding a new category

This touches several files — miss one and the category can silently run but
never show up somewhere (or vice versa):

1. **Create `src/datasets/<name>.py`** with a `<NAME>_DATASET` list, same
   `inputs`/`expectations` shape as above.
2. **Register it in `src/datasets/__init__.py`**: import the dataset, add it
   to the `EVAL_DATASET` concatenation, and add it to the `CATEGORY_DATASETS`
   dict (this is what generates its content hash).
3. **Write a checker in `src/mlflow_eval.py`**: a plain
   `check_<name>(content, expectations) -> (result, rationale)` function
   (`result` is `True`/`False` or a `0.0`-`1.0` score — see existing
   `check_*` functions), plus an MLflow-facing wrapper:
   ```python
   @scorer
   def <name>_correct(outputs, expectations) -> Feedback:
       result, rationale = check_<name>(_content_of(outputs), expectations)
       return Feedback(value=result, rationale=rationale)
   ```
4. **Register the category** in all three dicts near the bottom of
   `mlflow_eval.py`: `CATEGORY_SCORERS`, `CATEGORY_CHECKERS`, and
   `CATEGORY_FLAGS` (the CLI `--<flag>` option is generated automatically
   from `CATEGORY_FLAGS` — no argparse changes needed).
5. **Add unit tests** in `tests/test_scorers.py` for the new `check_<name>`
   function, mirroring an existing `TestCheck*` class — no Ollama/MLflow
   needed for these.
6. **Update the README**: the category table under "What's scored, and how",
   and the list of valid `categories` values under "Named eval-suite
   configs".

Once steps 1-4 are done, the category is automatically usable via `--all`,
its own `--<flag>`, and `categories: all` or `categories: [<name>]` in a
`configs/*.yaml` suite — no extra plumbing needed there.
