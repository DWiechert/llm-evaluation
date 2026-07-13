"""
datasets — Evaluation data for mlflow_eval.py

Each category lives in its own module (see e.g. `finance.py`, `coding.py`) and is
combined here into EVAL_DATASET. Every category is scored with rule-based/code
scorers only (no LLM judge, per your preference). That means:

  - basic_questions: exact/substring match against a known-correct short answer
  - tool_usage: checks the model picked the right tool + required args (structural)
  - coding: ACTUALLY RUNS the generated code against test cases — real pass/fail
  - finance: extracts the model's numeric answer and checks it against a value
             computed with plain Python, not guessed
  - reasoning: same numeric-extraction approach as finance — these are riddles
             with one unambiguous computed answer, not open-ended judgment calls
  - instruction_following: checks concrete, machine-verifiable constraints from
             the prompt (sentence/bullet counts, word limits, forbidden words,
             required JSON keys) — not prose quality
  - design: NO deterministic scorer exists for open-ended system design quality.
             These are included for side-by-side manual review only. The scorer
             attached to this category is a coverage heuristic (does it mention
             expected concerns), not a quality judgment — treat it as a checklist,
             not a grade.
"""

import hashlib
import json

from .basic_questions import BASIC_DATASET
from .coding import CODING_DATASET
from .design import DESIGN_DATASET
from .finance import FINANCE_DATASET
from .instruction_following import INSTRUCTION_DATASET
from .reasoning import REASONING_DATASET
from .tool_usage import TOOL_DATASET

EVAL_DATASET = BASIC_DATASET + TOOL_DATASET + CODING_DATASET + FINANCE_DATASET + REASONING_DATASET + INSTRUCTION_DATASET + DESIGN_DATASET

# ---- Version hashes, stamped into exported results so old/new rows scored ----
# against different ground truth never look silently comparable (issue #8).
CATEGORY_DATASETS = {
    "basic_questions": BASIC_DATASET,
    "tool_usage": TOOL_DATASET,
    "coding": CODING_DATASET,
    "finance": FINANCE_DATASET,
    "reasoning": REASONING_DATASET,
    "instruction_following": INSTRUCTION_DATASET,
    "design": DESIGN_DATASET,
}


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


DATASET_HASH = _hash(EVAL_DATASET)
CATEGORY_HASHES = {category: _hash(dataset) for category, dataset in CATEGORY_DATASETS.items()}
