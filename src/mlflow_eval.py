#!/usr/bin/env python3
"""
mlflow_eval.py — Evaluate local Ollama models with mlflow.genai.evaluate()
using rule-based/code scorers only (no LLM-as-judge), plus a flat CSV/SQLite
export for your own analysis.

Requires (run this on your machine with Ollama, not in a sandbox):
    uv sync

Usage:
    uv run src/mlflow_eval.py --models qwen3.5:9b llama3.1:8b qwen2.5-coder:7b

    # Run only specific categories (flags are additive; default is --all):
    uv run src/mlflow_eval.py --models qwen3.5:9b --basic --tools

Then view results:
    uv run src/mlflow_ui.py
    # open http://localhost:5000 and browse the "local-llm-eval" experiment
    # everything (CSV/SQLite export, MLflow's own tracking DB + artifacts)
    # lives under ./results/ — nothing is written to the repo root

What's captured and how honest each number is:
  - Runtime/latency: real, timed around each Ollama call in this script AND
    separately captured by MLflow's own trace timing (both should roughly agree).
  - Tokens & tok/s: pulled directly from Ollama's own response fields
    (eval_count, prompt_eval_count, eval_duration, prompt_eval_duration) —
    not estimated.
  - VRAM: best-effort via `nvidia-smi`, sampled immediately before/after each
    call. This is a rough delta, not a precise per-model allocation — other
    processes on your GPU will pollute it, and Ollama may keep a model loaded
    between calls, so "before" isn't always a clean zero baseline.
  - Model metadata (parameter_size, quantization_level): pulled directly from
    Ollama's /api/show endpoint, cached per model.
  - Cost: intentionally not included — these are local models with no metered
    per-token price, so a cost column would just be zero/meaningless.
  - basic_questions, finance, reasoning: deterministic checks against values
    computed with plain Python (see eval_dataset.py).
  - tool_usage: structural check only (right tool name + required arg keys
    present) — does not validate that argument VALUES are correct.
  - coding: ACTUALLY EXECUTES the model's generated code against test cases
    in a subprocess. Runs arbitrary model-generated code on your machine —
    fine for a personal benchmark, not for untrusted use.
  - instruction_following: checks concrete, machine-verifiable constraints
    (sentence/bullet counts, word limits, forbidden words, required JSON
    keys) extracted from the prompt — not a prose-quality judgment.
  - design: no scorer can judge open-ended design quality. The attached
    "score" is a keyword-checklist heuristic only — read the actual
    responses in the MLflow UI trace view or the exported CSV.
"""

import argparse
import csv
import json
import platform
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
import yaml

try:
    import mlflow
    from mlflow.entities import Feedback
    from mlflow.genai import scorer
except ImportError:
    print("mlflow is not installed. Run `uv sync`, then invoke this script with `uv run src/mlflow_eval.py ...`.")
    sys.exit(1)

from eval_dataset import CATEGORY_HASHES, DATASET_HASH, EVAL_DATASET

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_SHOW_URL = "http://localhost:11434/api/show"

# Deterministic sampling for every model/category so comparisons reflect model
# quality rather than differing Modelfile defaults (see issue #15).
SAMPLING_OPTIONS = {"temperature": 0, "seed": 42}

# Populated as the script runs; used both for the MLflow-side scorers and for
# the flat CSV/SQLite export at the end.
RUN_LOG = []
_MODEL_META_CACHE = {}


# --------------------------------------------------------------------------
# Model metadata (best-effort; never fatal if it fails)
# --------------------------------------------------------------------------
def get_model_metadata(model_name):
    if model_name in _MODEL_META_CACHE:
        return _MODEL_META_CACHE[model_name]
    meta = {"parameter_size": None, "quantization_level": None, "family": None}
    try:
        resp = requests.post(OLLAMA_SHOW_URL, json={"model": model_name}, timeout=15)
        resp.raise_for_status()
        details = resp.json().get("details", {})
        meta["parameter_size"] = details.get("parameter_size")
        meta["quantization_level"] = details.get("quantization_level")
        meta["family"] = details.get("family")
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  (could not fetch metadata for {model_name}: {e})")
    _MODEL_META_CACHE[model_name] = meta
    return meta


def get_vram_used_mb():
    """Best-effort VRAM query via nvidia-smi. Returns None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def unload_ollama_model(model_name):
    """Best-effort `ollama stop` so a model doesn't sit in VRAM (and skew the
    next model's vram_before baseline) once we're done evaluating it."""
    try:
        subprocess.run(["ollama", "stop", model_name], capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


# --------------------------------------------------------------------------
# System info — portable, stdlib-only where possible, so anyone can run this
# and their results are comparable to yours. Every field is best-effort:
# on failure it's None, never a guess.
# --------------------------------------------------------------------------
def _get_cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machine.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        elif system == "Windows":
            out = subprocess.run(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired, IndexError):
        pass
    # Fallback: less pretty but always available
    return platform.processor() or None


def _get_ram_total_gb():
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 1)
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return round(int(out.stdout.strip()) / (1024 ** 3), 1)
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def _get_gpu_info():
    """NVIDIA only, via nvidia-smi. AMD/Intel GPUs aren't detected here —
    there's no single stdlib-free tool that covers all three without an
    extra install, and we're prioritizing zero-install portability."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, vram_mb, driver = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            return {"gpu_name": name, "gpu_vram_total_mb": int(vram_mb), "gpu_driver_version": driver}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return {"gpu_name": None, "gpu_vram_total_mb": None, "gpu_driver_version": None}


def _get_ollama_version():
    try:
        resp = requests.get("http://localhost:11434/api/version", timeout=5)
        resp.raise_for_status()
        return resp.json().get("version")
    except (requests.exceptions.RequestException, ValueError):
        return None


def collect_system_info(run_id):
    gpu = _get_gpu_info()
    info = {
        "run_id": run_id,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine_arch": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": _get_cpu_model(),
        "cpu_logical_cores": __import__("os").cpu_count(),
        "ram_total_gb": _get_ram_total_gb(),
        "ollama_version": _get_ollama_version(),
    }
    info.update(gpu)
    return info


# --------------------------------------------------------------------------
# predict_fn factory — one closure per model so mlflow.genai.evaluate() can
# call the same signature for every row regardless of model under test.
# --------------------------------------------------------------------------
def make_predict_fn(model_name, run_id):
    meta = get_model_metadata(model_name)

    def predict_fn(id, category, messages, tools=None):
        # mlflow.genai.evaluate() calls predict_fn once outside a traced span
        # (its "test with the first sample" pre-check) before the real, traced
        # evaluation loop — skip tagging then to avoid a spurious warning.
        if mlflow.get_current_active_span() is not None:
            mlflow.update_current_trace(tags={
                "model": model_name, "category": category,
                "category_hash": CATEGORY_HASHES.get(category),
            })
        payload = {"model": model_name, "messages": messages, "stream": False, "options": SAMPLING_OPTIONS}
        if tools:
            payload["tools"] = tools

        vram_before = get_vram_used_mb()
        wall_start = time.time()
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            RUN_LOG.append({
                "run_id": run_id, "model": model_name, "id": id, "category": category,
                "category_hash": CATEGORY_HASHES.get(category), "error": str(e),
            })
            return {"content": "", "tool_calls": None}
        wall_seconds = round(time.time() - wall_start, 3)
        vram_after = get_vram_used_mb()

        msg = data.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        total_duration_ns = data.get("total_duration", 0)

        decode_tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else None
        prefill_tps = (
            prompt_eval_count / (prompt_eval_duration_ns / 1e9)
            if prompt_eval_duration_ns else None
        )

        RUN_LOG.append({
            "run_id": run_id,
            "model": model_name,
            "id": id,
            "category": category,
            "category_hash": CATEGORY_HASHES.get(category),
            "parameter_size": meta["parameter_size"],
            "quantization_level": meta["quantization_level"],
            "wall_seconds": wall_seconds,
            "total_duration_s": round(total_duration_ns / 1e9, 3) if total_duration_ns else None,
            "output_tokens": eval_count,
            "input_tokens": prompt_eval_count,
            "decode_tokens_per_sec": round(decode_tps, 1) if decode_tps else None,
            "prefill_tokens_per_sec": round(prefill_tps, 1) if prefill_tps else None,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_delta_mb": (vram_after - vram_before) if (vram_before is not None and vram_after is not None) else None,
            "output_content": content,
            "output_tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "error": None,
        })

        return {"content": content, "tool_calls": tool_calls}

    return predict_fn


def _content_of(outputs):
    if isinstance(outputs, dict):
        return outputs.get("content", "") or ""
    return str(outputs)


# --------------------------------------------------------------------------
# Shared scoring logic — used both by the @scorer functions below (for the
# MLflow UI/traces) and by build_export_rows() (for the CSV/SQLite export),
# so the two never disagree with each other.
# --------------------------------------------------------------------------
def check_basic_answer(content, expectations):
    expected = expectations["expected_substring"]
    ok = expected.lower() in content.lower()
    return ok, f"Looked for '{expected}' in response. Found: {ok}"


def check_tool_call(tool_calls, expectations):
    if not tool_calls:
        return False, "Model did not make a tool call."
    call = tool_calls[0]
    fn = call.get("function", call)
    name = fn.get("name")
    raw_args = fn.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    else:
        args = raw_args or {}
    name_match = name == expectations["expected_tool"]
    missing = [a for a in expectations["required_args"] if a not in args]
    ok = name_match and not missing
    return ok, f"Called '{name}' (expected '{expectations['expected_tool']}'). Missing required args: {missing or 'none'}."


def _extract_fenced(text):
    """Strip a markdown code fence (```python, ```json, or bare ```) if present."""
    match = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def _run_code_tests(code, function_name, test_cases, timeout=10):
    harness = f"""
{code}

import json
_tests = {test_cases!r}
_results = []
for _t in _tests:
    try:
        _r = {function_name}(*_t["args"])
        _results.append(_r == _t["expected"])
    except Exception as _e:
        _results.append(type(_e).__name__ == _t["expected"])
print(json.dumps(_results))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0, len(test_cases), "timed out"

    if proc.returncode != 0:
        return 0, len(test_cases), proc.stderr.strip()[-500:]

    try:
        results = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return 0, len(test_cases), "could not parse test output"

    return sum(results), len(test_cases), None


def check_coding(content, expectations):
    code = _extract_fenced(content)
    passed, total, error = _run_code_tests(code, expectations["function_name"], expectations["test_cases"])
    if error:
        return 0.0, f"Execution error: {error}"
    return passed / total, f"{passed}/{total} test cases passed"


def check_numeric_answer(content, expectations):
    """Used by both 'finance' and 'reasoning' categories — both are riddles/
    problems with one unambiguous number computed by plain Python, just
    extracted from free-text prose rather than a structured field."""
    raw_numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", content)
    numbers = []
    for n in raw_numbers:
        try:
            numbers.append(float(n.replace(",", "")))
        except ValueError:
            continue
    if not numbers:
        return False, "No number found in response."
    target = expectations["expected_number"]
    tol = expectations.get("tolerance", 1.0)
    closest = min(numbers, key=lambda x: abs(x - target))
    ok = abs(closest - target) <= tol
    return ok, f"Closest number in response: {closest}. Expected {target} (±{tol})."


def check_instruction_following(content, expectations):
    """Runs whichever concrete, machine-verifiable checks are present in
    `expectations` (sentence structure, bullet formatting, JSON schema) and
    reports the fraction that passed. Not a prose-quality judgment."""
    checks = []

    if "sentence_count" in expectations or "required_substrings_by_sentence" in expectations or "cta_keywords" in expectations:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content.strip()) if s.strip()]
        if "sentence_count" in expectations:
            checks.append((f"sentence count == {expectations['sentence_count']}", len(sentences) == expectations["sentence_count"]))
        for idx_str, required in expectations.get("required_substrings_by_sentence", {}).items():
            idx = int(idx_str)
            sentence = sentences[idx] if idx < len(sentences) else ""
            hit = any(r.lower() in sentence.lower() for r in required)
            checks.append((f"sentence {idx} mentions one of {required}", hit))
        if "cta_keywords" in expectations:
            idx = expectations.get("cta_sentence_index", len(sentences) - 1)
            sentence = sentences[idx] if 0 <= idx < len(sentences) else ""
            hit = any(k.lower() in sentence.lower() for k in expectations["cta_keywords"])
            checks.append((f"sentence {idx} reads as a call to action", hit))

    for forbidden in expectations.get("forbidden_substrings", []):
        checks.append((f"does not contain '{forbidden}'", forbidden.lower() not in content.lower()))

    if "bullet_count" in expectations or "max_words_per_bullet" in expectations:
        bullets = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", ln) for ln in content.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", ln)]
        if "bullet_count" in expectations:
            checks.append((f"bullet count == {expectations['bullet_count']}", len(bullets) == expectations["bullet_count"]))
        if "max_words_per_bullet" in expectations:
            limit = expectations["max_words_per_bullet"]
            over = [b for b in bullets if len(b.split()) > limit]
            checks.append((f"all bullets <= {limit} words", not over))

    if "json_required_keys" in expectations:
        try:
            parsed = json.loads(_extract_fenced(content).strip())
        except (json.JSONDecodeError, ValueError):
            parsed = None
        checks.append(("response is a valid JSON object", isinstance(parsed, dict)))
        if isinstance(parsed, dict):
            for key in expectations["json_required_keys"]:
                checks.append((f"JSON has key '{key}'", key in parsed))
            if "sentiment_options" in expectations and "sentiment" in parsed:
                allowed = [o.lower() for o in expectations["sentiment_options"]]
                checks.append(("'sentiment' is an allowed value", str(parsed.get("sentiment", "")).lower() in allowed))
            if "reason_max_words" in expectations and "reason" in parsed:
                limit = expectations["reason_max_words"]
                checks.append((f"'reason' <= {limit} words", len(str(parsed.get("reason", "")).split()) <= limit))

    if not checks:
        return None, "No instruction-following checks defined for this expectation set."

    passed = sum(1 for _, ok in checks if ok)
    rationale = "; ".join(f"{'PASS' if ok else 'FAIL'}: {desc}" for desc, ok in checks)
    return passed / len(checks), rationale


def check_design(content, expectations):
    concepts = expectations["expected_concepts"]
    content_lower = content.lower()
    hits = [c for c in concepts if c.lower() in content_lower]
    coverage = len(hits) / len(concepts)
    return coverage, (
        f"HEURISTIC ONLY, not a quality score. Keyword checklist hit "
        f"{len(hits)}/{len(concepts)}: {hits}. Read the actual response for quality."
    )


# --------------------------------------------------------------------------
# @scorer wrappers for mlflow.genai.evaluate() (drives the MLflow UI/traces)
# --------------------------------------------------------------------------
@scorer
def basic_answer_correct(outputs, expectations) -> Feedback:
    ok, rationale = check_basic_answer(_content_of(outputs), expectations)
    return Feedback(value=ok, rationale=rationale)


@scorer
def tool_call_correct(outputs, expectations) -> Feedback:
    tool_calls = outputs.get("tool_calls") if isinstance(outputs, dict) else None
    ok, rationale = check_tool_call(tool_calls, expectations)
    return Feedback(value=ok, rationale=rationale)


@scorer
def coding_tests_pass(outputs, expectations) -> Feedback:
    value, rationale = check_coding(_content_of(outputs), expectations)
    return Feedback(value=value, rationale=rationale)


@scorer
def numeric_answer_correct(outputs, expectations) -> Feedback:
    ok, rationale = check_numeric_answer(_content_of(outputs), expectations)
    return Feedback(value=ok, rationale=rationale)


@scorer
def instruction_following_correct(outputs, expectations) -> Feedback:
    value, rationale = check_instruction_following(_content_of(outputs), expectations)
    return Feedback(value=value, rationale=rationale)


@scorer
def design_concept_coverage(outputs, expectations) -> Feedback:
    value, rationale = check_design(_content_of(outputs), expectations)
    return Feedback(value=value, rationale=rationale)


CATEGORY_SCORERS = {
    "basic_questions": [basic_answer_correct],
    "tool_usage": [tool_call_correct],
    "coding": [coding_tests_pass],
    "finance": [numeric_answer_correct],
    "reasoning": [numeric_answer_correct],
    "instruction_following": [instruction_following_correct],
    "design": [design_concept_coverage],
}

CATEGORY_CHECKERS = {
    "basic_questions": lambda content, tool_calls, exp: check_basic_answer(content, exp),
    "tool_usage": lambda content, tool_calls, exp: check_tool_call(tool_calls, exp),
    "coding": lambda content, tool_calls, exp: check_coding(content, exp),
    "finance": lambda content, tool_calls, exp: check_numeric_answer(content, exp),
    "reasoning": lambda content, tool_calls, exp: check_numeric_answer(content, exp),
    "instruction_following": lambda content, tool_calls, exp: check_instruction_following(content, exp),
    "design": lambda content, tool_calls, exp: check_design(content, exp),
}

# Maps a CLI flag (e.g. --basic) to the category name used in eval_dataset.py.
CATEGORY_FLAGS = {
    "basic": "basic_questions",
    "tools": "tool_usage",
    "coding": "coding",
    "finance": "finance",
    "reasoning": "reasoning",
    "instructions": "instruction_following",
    "design": "design",
}


# --------------------------------------------------------------------------
# Export: build flat rows from RUN_LOG + EVAL_DATASET expectations, using the
# SAME checker functions as the MLflow scorers above, so the CSV/SQLite
# numbers always match what you'd see in the MLflow UI.
# --------------------------------------------------------------------------
def build_export_rows():
    expectations_by_id = {row["inputs"]["id"]: row["expectations"] for row in EVAL_DATASET}
    export_rows = []
    for entry in RUN_LOG:
        row = dict(entry)
        if entry.get("error"):
            row["score"] = None
            row["score_rationale"] = f"Request failed: {entry['error']}"
            export_rows.append(row)
            continue

        expectations = expectations_by_id.get(entry["id"], {})
        content = entry.pop("output_content", "")
        tool_calls_json = entry.get("output_tool_calls")
        tool_calls = json.loads(tool_calls_json) if tool_calls_json else None
        checker = CATEGORY_CHECKERS.get(entry["category"])
        if checker and expectations:
            score, rationale = checker(content, tool_calls, expectations)
        else:
            score, rationale = None, "No checker/expectations available"
        row["output_content"] = content
        row["score"] = score
        row["score_rationale"] = rationale
        export_rows.append(row)
    return export_rows


def export_csv(rows, path):
    if not rows:
        print("No rows to export.")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")


def export_sqlite(rows, path):
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    conn = sqlite3.connect(path)
    cols_sql = ", ".join(f'"{c}" TEXT' for c in fieldnames)
    conn.execute(f'CREATE TABLE IF NOT EXISTS eval_results ({cols_sql})')
    placeholders = ", ".join("?" for _ in fieldnames)
    col_list = ", ".join(f'"{c}"' for c in fieldnames)
    for row in rows:
        values = [json.dumps(row[c]) if isinstance(row.get(c), (dict, list)) else row.get(c) for c in fieldnames]
        conn.execute(f'INSERT INTO eval_results ({col_list}) VALUES ({placeholders})', values)
    conn.commit()
    conn.close()
    print(f"SQLite: {path} (table: eval_results)")


def configure_mlflow(experiment_name, out_dir):
    """Point MLflow's tracking DB and artifact store at out_dir instead of
    letting it default to ./mlflow.db and ./mlruns/ at the repo root."""
    mlflow.set_tracking_uri(f"sqlite:///{out_dir}/mlflow.db")
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        artifact_location = (out_dir / "mlartifacts").resolve().as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def export_system_info_json(info, path):
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"System info: {path}")


def export_system_info_sqlite(info, path):
    conn = sqlite3.connect(path)
    cols_sql = ", ".join(f'"{c}" TEXT' for c in info.keys())
    conn.execute(f'CREATE TABLE IF NOT EXISTS system_info ({cols_sql}, PRIMARY KEY ("run_id"))')
    placeholders = ", ".join("?" for _ in info)
    col_list = ", ".join(f'"{c}"' for c in info.keys())
    conn.execute(f'INSERT OR REPLACE INTO system_info ({col_list}) VALUES ({placeholders})', list(info.values()))
    conn.commit()
    conn.close()
    print(f"System info also written to {path} (table: system_info, join on run_id)")


# --------------------------------------------------------------------------
# Named eval-suite configs (configs/*.yaml) — an alternative to retyping
# long --models/--category invocations. CLI flags, if given, override the
# config's values rather than merging with them.
# --------------------------------------------------------------------------
def load_eval_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)

    models = config.get("models")
    if not models:
        raise ValueError(f"{path}: 'models' must be a non-empty list")

    categories = config.get("categories", "all")
    if categories == "all":
        categories = list(CATEGORY_SCORERS.keys())
    else:
        unknown = [c for c in categories if c not in CATEGORY_SCORERS]
        if unknown:
            raise ValueError(f"{path}: unknown categories {unknown} (valid: {list(CATEGORY_SCORERS.keys())})")

    return {"models": models, "categories": categories}


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="Path to a configs/*.yaml eval suite (models + categories). CLI flags below override its values.")
    ap.add_argument("--models", nargs="+", help="Ollama model tags to evaluate (required unless --config is given)")
    ap.add_argument("--experiment", default="local-llm-eval", help="MLflow experiment name")
    ap.add_argument("--out", default="results", help="Output directory for CSV/SQLite export")
    ap.add_argument("--all", action="store_true", help="Run every category (default if no category flag is given)")
    for flag, category in CATEGORY_FLAGS.items():
        ap.add_argument(f"--{flag}", action="store_true", help=f"Run only the '{category}' category (repeatable alongside other category flags)")
    return ap


def resolve_run_config(args, ap):
    """Merge --config with CLI flags: any --models/category flag given on the
    CLI overrides that part of the config rather than merging with it."""
    cli_categories = [category for flag, category in CATEGORY_FLAGS.items() if getattr(args, flag)]

    if args.config:
        config = load_eval_config(args.config)
        models = args.models or config["models"]
        if args.all:
            selected_categories = list(CATEGORY_SCORERS.keys())
        elif cli_categories:
            selected_categories = cli_categories
        else:
            selected_categories = config["categories"]
    else:
        if not args.models:
            ap.error("--models is required unless --config is given")
        models = args.models
        selected_categories = cli_categories
        if args.all or not selected_categories:
            selected_categories = list(CATEGORY_SCORERS.keys())

    return models, selected_categories


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    models, selected_categories = resolve_run_config(args, ap)

    run_id = uuid.uuid4().hex[:12]
    print(f"Run ID: {run_id}")
    print(f"Categories: {', '.join(selected_categories)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_mlflow(args.experiment, out_dir)

    # A single run for the whole process, so the MLflow Run ID stays uniform
    # across every model/category instead of changing per iteration. Each
    # individual model/category is still distinguishable via the "model" and
    # "category" trace tags set in predict_fn (visible as Traces columns).
    with mlflow.start_run(run_name=run_id):
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("models", ", ".join(models))
        mlflow.set_tag("categories", ", ".join(selected_categories))

        mlflow.set_tag("dataset_hash", DATASET_HASH)

        for model in models:
            predict_fn = make_predict_fn(model, run_id)
            for category, scorers in CATEGORY_SCORERS.items():
                if category not in selected_categories:
                    continue
                subset = [row for row in EVAL_DATASET if row["inputs"]["category"] == category]
                if not subset:
                    continue
                print(f"\n=== {model} | {category} ({len(subset)} cases) ===")
                mlflow.genai.evaluate(
                    data=subset,
                    predict_fn=predict_fn,
                    scorers=scorers,
                )
            print(f"Unloading {model} from Ollama...")
            unload_ollama_model(model)

    export_rows = build_export_rows()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_csv(export_rows, out_dir / f"eval_results_{timestamp}.csv")
    db_path = out_dir / f"eval_results_{timestamp}.db"
    export_sqlite(export_rows, db_path)

    system_info = collect_system_info(run_id)
    export_system_info_json(system_info, out_dir / f"system_info_{timestamp}.json")
    export_system_info_sqlite(system_info, db_path)

    print("\nDone. Run `uv run src/mlflow_ui.py` and open http://localhost:5000 to browse results in detail.")
    print(f"Every row in eval_results is tagged run_id={run_id} — join against system_info to compare across machines.")


if __name__ == "__main__":
    main()
