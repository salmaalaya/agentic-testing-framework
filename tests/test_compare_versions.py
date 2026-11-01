"""Regression detection between two versions of the target agent."""

from scripts.compare_versions import diff_results


def case(verdict="PASS", score=4.0, hallucination=False, latency=100, canal="api"):
    return {
        "verdict": verdict,
        "score_moyen": score,
        "hallucination": hallucination,
        "latency_ms": latency,
        "canal": canal,
    }


def test_identical_runs_have_no_regression():
    old = {"TC-1": case(), "TC-2": case(verdict="FAIL", score=1.0)}
    diff = diff_results(old, dict(old))
    assert diff["regressions"] == []
    assert diff["improvements"] == []
    assert diff["unchanged"] == 2


def test_pass_to_fail_is_a_regression():
    diff = diff_results({"TC-1": case()}, {"TC-1": case(verdict="FAIL")})
    assert [r["id"] for r in diff["regressions"]] == ["TC-1"]
    assert "PASS→FAIL" in diff["regressions"][0]["reasons"][0]


def test_new_hallucination_is_a_regression():
    diff = diff_results({"TC-1": case()}, {"TC-1": case(hallucination=True)})
    assert any("hallucination" in reason for reason in diff["regressions"][0]["reasons"])


def test_score_drop_threshold_is_half_a_point():
    small = diff_results({"TC-1": case(score=4.0)}, {"TC-1": case(score=3.6)})
    big = diff_results({"TC-1": case(score=4.0)}, {"TC-1": case(score=3.5)})
    assert small["regressions"] == []
    assert len(big["regressions"]) == 1


def test_latency_increase_over_50_percent():
    ok = diff_results({"TC-1": case(latency=100)}, {"TC-1": case(latency=150)})
    ko = diff_results({"TC-1": case(latency=100)}, {"TC-1": case(latency=151)})
    assert ok["regressions"] == []
    assert "latence" in ko["regressions"][0]["reasons"][0]


def test_fail_to_pass_is_reported_as_improvement():
    diff = diff_results({"TC-1": case(verdict="FAIL", score=1.0)}, {"TC-1": case()})
    assert diff["improvements"] == [{"id": "TC-1", "detail": "verdict FAIL→PASS"}]
    assert diff["regressions"] == []


def test_added_and_removed_tests_are_listed():
    diff = diff_results({"TC-1": case(), "TC-2": case()}, {"TC-1": case(), "TC-3": case()})
    assert diff["new_only"] == ["TC-3"]
    assert diff["missing"] == ["TC-2"]
    assert diff["common_ids"] == ["TC-1"]
