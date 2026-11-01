"""score_result() with a stubbed LLM: the verdict is recomputed from the scores, never trusted."""

import json
from types import SimpleNamespace

import pytest

from agentic_testing.agents import evaluator


class FakeCompletions:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        content = self._contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*contents):
        completions = FakeCompletions(contents)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(evaluator, "_ollama_client", client)
        return completions

    return install


BASE_RESULT = {
    "id": "TC-002",
    "name": "Prix produit",
    "input": "Quel est le prix du PhoneUltra 5G ?",
    "actual": "Le PhoneUltra 5G coûte 899.99 DT.",
    "status_code": 200,
    "latency_ms": 10,
    "expected": {"max_latency_ms": 15000, "reference_response": "Le PhoneUltra 5G coûte 899.99 DT."},
    "failure_criteria": {},
}


def llm_json(pertinence, exactitude, coherence, hallucination, verdict="PASS"):
    return json.dumps(
        {
            "pertinence": pertinence,
            "exactitude": exactitude,
            "coherence": coherence,
            "hallucination": hallucination,
            "verdict": verdict,
            "reason": "stub",
        }
    )


def test_hallucination_forces_fail_even_if_llm_says_pass(fake_llm):
    fake_llm(llm_json(5, 5, 5, True, verdict="PASS"))
    scored = json.loads(evaluator.score_result(json.dumps(BASE_RESULT)))
    assert scored["verdict"] == "FAIL"
    assert scored["score_moyen"] == 5.0


def test_low_average_fails_even_if_llm_says_pass(fake_llm):
    fake_llm(llm_json(2, 3, 3, False, verdict="PASS"))
    scored = json.loads(evaluator.score_result(json.dumps(BASE_RESULT)))
    assert scored["score_moyen"] == pytest.approx(2.67)
    assert scored["verdict"] == "FAIL"


def test_json_wrapped_in_text_is_extracted(fake_llm):
    fake_llm("Voici mon évaluation : " + llm_json(4, 4, 4, False) + " Merci")
    scored = json.loads(evaluator.score_result(json.dumps(BASE_RESULT)))
    assert scored["verdict"] == "PASS"
    assert scored["bleu_score"] == 1.0


def test_invalid_json_is_retried_then_neutral_warn(fake_llm):
    completions = fake_llm("pas du json", "toujours pas")
    scored = json.loads(evaluator.score_result(json.dumps(BASE_RESULT)))
    assert completions.calls == 2
    assert scored["verdict"] == "WARN"
    assert scored["score_moyen"] == 3.0


def test_precheck_failure_skips_the_llm(fake_llm):
    completions = fake_llm()
    scored = json.loads(evaluator.score_result(json.dumps({**BASE_RESULT, "status_code": 503})))
    assert completions.calls == 0
    assert scored["verdict"] == "FAIL"
