"""Deterministic pre-checks run by the Evaluator before any LLM call."""

import pytest

from agentic_testing.agents.evaluator import deterministic_precheck


def make_result(**overrides):
    result = {
        "id": "TC-X",
        "actual": "Le PhoneUltra 5G est disponible à 899.99 DT.",
        "status_code": 200,
        "latency_ms": 120,
        "error": None,
        "expected": {"max_latency_ms": 15000},
        "failure_criteria": {"fail_if_status_gte": 400, "fail_if_empty_response": True},
    }
    result.update(overrides)
    return result


def test_clean_result_is_left_to_the_llm():
    assert deterministic_precheck(make_result()) == (None, None)


@pytest.mark.parametrize(
    ("error", "expected_verdict"),
    [
        ("SKIP: mobile channel disabled", "WARN"),
        ("APPIUM_UNAVAILABLE", "ERROR"),
        ("APPIUM_ERROR:NoSuchElementException", "FAIL"),
        ("APPIUM_ERROR:TimeoutException", "FAIL"),
        ("CONNECTION_ERROR - agent down", "ERROR"),
    ],
)
def test_execution_errors_short_circuit(error, expected_verdict):
    verdict, reason = deterministic_precheck(make_result(error=error))
    assert verdict == expected_verdict
    assert reason


def test_server_error_fails():
    verdict, reason = deterministic_precheck(make_result(status_code=500))
    assert verdict == "FAIL"
    assert "500" in reason


def test_latency_above_threshold_fails():
    verdict, reason = deterministic_precheck(make_result(latency_ms=15001))
    assert verdict == "FAIL"
    assert "15001" in reason


def test_latency_at_threshold_is_left_to_the_llm():
    assert deterministic_precheck(make_result(latency_ms=15000)) == (None, None)


def test_empty_response_fails_when_forbidden():
    verdict, _ = deterministic_precheck(make_result(actual="   "))
    assert verdict == "FAIL"


def test_required_keyword_is_case_insensitive():
    result = make_result(expected={"max_latency_ms": 15000, "contains": ["PHONEULTRA"]})
    assert deterministic_precheck(result) == (None, None)


def test_missing_required_keyword_fails():
    result = make_result(expected={"max_latency_ms": 15000, "contains_all": ["899.99", "stock"]})
    verdict, reason = deterministic_precheck(result)
    assert verdict == "FAIL"
    assert "stock" in reason


def test_contains_any_needs_at_least_one_match():
    ok = make_result(expected={"max_latency_ms": 15000, "contains_any": ["rupture", "disponible"]})
    ko = make_result(expected={"max_latency_ms": 15000, "contains_any": ["rupture", "épuisé"]})
    assert deterministic_precheck(ok) == (None, None)
    assert deterministic_precheck(ko)[0] == "FAIL"


def test_forbidden_keyword_fails():
    result = make_result(actual="<script>alert(1)</script>", expected={"not_contains": ["<script>"]})
    verdict, reason = deterministic_precheck(result)
    assert verdict == "FAIL"
    assert "<script>" in reason
