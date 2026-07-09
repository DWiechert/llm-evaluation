"""Tests for the --config eval-suite loading/override mechanic (issue #7)."""

import pytest

from mlflow_eval import CATEGORY_SCORERS, build_arg_parser, load_eval_config, resolve_run_config

ALL_CATEGORIES = list(CATEGORY_SCORERS.keys())


def write_config(tmp_path, models, categories):
    path = tmp_path / "suite.yaml"
    lines = ["models:"] + [f"  - {m}" for m in models]
    if categories == "all":
        lines.append("categories: all")
    else:
        lines.append("categories:")
        lines += [f"  - {c}" for c in categories]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# load_eval_config
# --------------------------------------------------------------------------
def test_load_eval_config_explicit_categories(tmp_path):
    path = write_config(tmp_path, ["m1", "m2"], ["basic_questions", "coding"])
    config = load_eval_config(path)
    assert config == {"models": ["m1", "m2"], "categories": ["basic_questions", "coding"]}


def test_load_eval_config_all_categories(tmp_path):
    path = write_config(tmp_path, ["m1"], "all")
    config = load_eval_config(path)
    assert config["categories"] == ALL_CATEGORIES


def test_load_eval_config_rejects_unknown_category(tmp_path):
    path = write_config(tmp_path, ["m1"], ["not_a_real_category"])
    with pytest.raises(ValueError, match="unknown categories"):
        load_eval_config(path)


def test_load_eval_config_rejects_empty_models(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text("models: []\ncategories: all\n")
    with pytest.raises(ValueError, match="'models' must be a non-empty list"):
        load_eval_config(path)


# --------------------------------------------------------------------------
# resolve_run_config — the CLI/config override mechanic
# --------------------------------------------------------------------------
def test_config_alone_uses_config_models_and_categories(tmp_path):
    path = write_config(tmp_path, ["m1"], ["basic_questions", "tool_usage"])
    ap = build_arg_parser()
    args = ap.parse_args(["--config", str(path)])
    models, categories = resolve_run_config(args, ap)
    assert models == ["m1"]
    assert categories == ["basic_questions", "tool_usage"]


def test_cli_models_override_config_models(tmp_path):
    path = write_config(tmp_path, ["config-model"], "all")
    ap = build_arg_parser()
    args = ap.parse_args(["--config", str(path), "--models", "cli-model"])
    models, _ = resolve_run_config(args, ap)
    assert models == ["cli-model"]


def test_cli_category_flag_overrides_config_categories(tmp_path):
    path = write_config(tmp_path, ["m1"], ["basic_questions", "tool_usage"])
    ap = build_arg_parser()
    args = ap.parse_args(["--config", str(path), "--coding"])
    _, categories = resolve_run_config(args, ap)
    assert categories == ["coding"]


def test_cli_all_flag_overrides_config_categories(tmp_path):
    path = write_config(tmp_path, ["m1"], ["basic_questions"])
    ap = build_arg_parser()
    args = ap.parse_args(["--config", str(path), "--all"])
    _, categories = resolve_run_config(args, ap)
    assert categories == ALL_CATEGORIES


def test_models_required_without_config():
    ap = build_arg_parser()
    args = ap.parse_args([])
    with pytest.raises(SystemExit):
        resolve_run_config(args, ap)


def test_no_config_no_category_flags_defaults_to_all():
    ap = build_arg_parser()
    args = ap.parse_args(["--models", "m1"])
    models, categories = resolve_run_config(args, ap)
    assert models == ["m1"]
    assert categories == ALL_CATEGORIES


def test_no_config_category_flags_are_additive():
    ap = build_arg_parser()
    args = ap.parse_args(["--models", "m1", "--basic", "--tools"])
    _, categories = resolve_run_config(args, ap)
    assert categories == ["basic_questions", "tool_usage"]
