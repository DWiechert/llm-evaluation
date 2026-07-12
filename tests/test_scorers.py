"""Tests for the rule-based scorer functions (issue #6).

These test the plain (content, expectations) -> (result, rationale) functions
directly, not the @scorer-wrapped MLflow versions — no Ollama/MLflow required.
"""

from mlflow_eval import (
    check_basic_answer,
    check_coding,
    check_design,
    check_instruction_following,
    check_numeric_answer,
    check_tool_call,
)


class TestCheckBasicAnswer:
    def test_match(self):
        ok, _ = check_basic_answer("The capital of France is Paris.", {"expected_substring": "Paris"})
        assert ok is True

    def test_case_insensitive(self):
        ok, _ = check_basic_answer("the capital is paris", {"expected_substring": "Paris"})
        assert ok is True

    def test_no_match(self):
        ok, _ = check_basic_answer("The capital of France is Lyon.", {"expected_substring": "Paris"})
        assert ok is False


class TestCheckToolCall:
    def test_no_calls(self):
        ok, rationale = check_tool_call([], {"expected_tool": "get_stock_price", "required_args": ["symbol"]})
        assert ok is False
        assert "did not make a tool call" in rationale.lower()

    def test_none(self):
        ok, _ = check_tool_call(None, {"expected_tool": "x", "required_args": []})
        assert ok is False

    def test_correct(self):
        tool_calls = [{"function": {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}}]
        ok, _ = check_tool_call(tool_calls, {"expected_tool": "get_stock_price", "required_args": ["symbol"]})
        assert ok is True

    def test_wrong_name(self):
        tool_calls = [{"function": {"name": "convert_currency", "arguments": {}}}]
        ok, _ = check_tool_call(tool_calls, {"expected_tool": "get_stock_price", "required_args": []})
        assert ok is False

    def test_missing_required_arg(self):
        tool_calls = [{"function": {"name": "get_stock_price", "arguments": {}}}]
        ok, rationale = check_tool_call(tool_calls, {"expected_tool": "get_stock_price", "required_args": ["symbol"]})
        assert ok is False
        assert "symbol" in rationale

    def test_arguments_as_json_string(self):
        tool_calls = [{"function": {"name": "get_stock_price", "arguments": '{"symbol": "AAPL"}'}}]
        ok, _ = check_tool_call(tool_calls, {"expected_tool": "get_stock_price", "required_args": ["symbol"]})
        assert ok is True

    def test_arguments_as_invalid_json_string(self):
        tool_calls = [{"function": {"name": "get_stock_price", "arguments": "not json"}}]
        ok, _ = check_tool_call(tool_calls, {"expected_tool": "get_stock_price", "required_args": ["symbol"]})
        assert ok is False


class TestCheckCoding:
    def test_all_pass(self):
        content = "```python\ndef add(a, b):\n    return a + b\n```"
        expectations = {"function_name": "add", "test_cases": [{"args": [1, 2], "expected": 3}, {"args": [2, 2], "expected": 4}]}
        score, rationale = check_coding(content, expectations)
        assert score == 1.0
        assert "2/2" in rationale

    def test_partial_pass(self):
        content = "```python\ndef add(a, b):\n    return a + b + 1\n```"
        expectations = {"function_name": "add", "test_cases": [{"args": [1, 2], "expected": 3}, {"args": [0, 0], "expected": 1}]}
        score, _ = check_coding(content, expectations)
        assert score == 0.5

    def test_no_fence(self):
        content = "def add(a, b):\n    return a + b\n"
        expectations = {"function_name": "add", "test_cases": [{"args": [1, 2], "expected": 3}]}
        score, _ = check_coding(content, expectations)
        assert score == 1.0

    def test_exception_matches_expected(self):
        content = "```python\ndef divide(a, b):\n    return a / b\n```"
        expectations = {"function_name": "divide", "test_cases": [{"args": [1, 0], "expected": "ZeroDivisionError"}]}
        score, _ = check_coding(content, expectations)
        assert score == 1.0

    def test_syntax_error_returns_zero(self):
        content = "```python\ndef add(a, b:\n    return a + b\n```"
        expectations = {"function_name": "add", "test_cases": [{"args": [1, 2], "expected": 3}]}
        score, rationale = check_coding(content, expectations)
        assert score == 0.0
        assert "Execution error" in rationale


class TestCheckNumericAnswer:
    def test_exact(self):
        ok, _ = check_numeric_answer("The answer is 42.", {"expected_number": 42, "tolerance": 1})
        assert ok is True

    def test_outside_tolerance(self):
        ok, _ = check_numeric_answer("The answer is 50.", {"expected_number": 42, "tolerance": 1})
        assert ok is False

    def test_picks_closest_number(self):
        ok, rationale = check_numeric_answer(
            "First I considered 10, but the real answer is 42.", {"expected_number": 42, "tolerance": 1}
        )
        assert ok is True
        assert "42" in rationale

    def test_no_number(self):
        ok, rationale = check_numeric_answer("I'm not sure.", {"expected_number": 42, "tolerance": 1})
        assert ok is False
        assert "No number found" in rationale

    def test_comma_formatted(self):
        ok, _ = check_numeric_answer("The total is 1,234.5 dollars.", {"expected_number": 1234.5, "tolerance": 0.01})
        assert ok is True

    def test_default_tolerance(self):
        ok, _ = check_numeric_answer("42.4", {"expected_number": 42})
        assert ok is True


class TestCheckInstructionFollowing:
    def test_no_checks_defined(self):
        result, _ = check_instruction_following("hello", {})
        assert result is None

    def test_sentence_count(self):
        score, _ = check_instruction_following("One. Two. Three.", {"sentence_count": 3})
        assert score == 1.0

    def test_sentence_count_fail(self):
        score, _ = check_instruction_following("One. Two.", {"sentence_count": 3})
        assert score == 0.0

    def test_required_substrings_by_sentence(self):
        content = "Intro sentence. This mentions Python. Closing."
        score, _ = check_instruction_following(content, {"required_substrings_by_sentence": {"1": ["python", "java"]}})
        assert score == 1.0

    def test_cta_keywords(self):
        score, _ = check_instruction_following("Here is info. Buy now!", {"cta_keywords": ["buy now", "sign up"]})
        assert score == 1.0

    def test_forbidden_substrings(self):
        score, _ = check_instruction_following("This is a great product.", {"forbidden_substrings": ["great", "amazing"]})
        assert score == 0.5

    def test_bullet_count_and_length(self):
        content = "- one\n- two\n- three"
        score, _ = check_instruction_following(content, {"bullet_count": 3, "max_words_per_bullet": 1})
        assert score == 1.0

    def test_json_required_keys(self):
        content = '```json\n{"sentiment": "positive", "reason": "great"}\n```'
        expectations = {
            "json_required_keys": ["sentiment", "reason"],
            "sentiment_options": ["positive", "negative", "neutral"],
            "reason_max_words": 5,
        }
        score, _ = check_instruction_following(content, expectations)
        assert score == 1.0

    def test_invalid_json(self):
        score, rationale = check_instruction_following("not json at all", {"json_required_keys": ["sentiment"]})
        assert score == 0.0
        assert "valid JSON" in rationale


class TestCheckDesign:
    def test_full_coverage(self):
        score, _ = check_design(
            "We need to consider scalability, security, and latency.",
            {"expected_concepts": ["scalability", "security", "latency"]},
        )
        assert score == 1.0

    def test_partial_coverage(self):
        score, _ = check_design("We need scalability.", {"expected_concepts": ["scalability", "security"]})
        assert score == 0.5

    def test_zero_coverage(self):
        score, _ = check_design("Not relevant.", {"expected_concepts": ["scalability"]})
        assert score == 0.0
