"""Static checks on the versioned test corpus (no network, no LLM)."""

import json

import pytest

from agentic_testing.config import CORPUS_DIR

CORPORA = {"test_cases.json": "api", "web_test_cases.json": "web", "mobile_test_cases.json": "mobile"}


def load(name):
    return json.loads((CORPUS_DIR / name).read_text(encoding="utf-8"))["test_cases"]


@pytest.mark.parametrize("name", CORPORA)
def test_every_case_has_the_required_fields(name):
    for case in load(name):
        for field in ("id", "name", "canal", "category", "input", "expected", "failure_criteria"):
            assert field in case, f"{case.get('id')} misses {field}"
        assert case["canal"] == CORPORA[name]
        assert case["category"] in {"nominal", "edge_case", "adversarial"}


def test_ids_are_unique_across_channels():
    ids = [case["id"] for name in CORPORA for case in load(name)]
    assert len(ids) == len(set(ids))


def test_corpus_has_73_scenarios():
    assert sum(len(load(name)) for name in CORPORA) == 73


def test_every_case_has_a_reference_response_for_bleu_rouge():
    for name in CORPORA:
        for case in load(name):
            assert case["expected"].get("reference_response"), case["id"]
