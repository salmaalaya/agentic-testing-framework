"""BLEU / ROUGE-L informative metrics."""

import pytest

from agentic_testing.evaluation.metrics import compute_nlp_metrics


def test_identical_answer_scores_one():
    text = "Le LaptopPro X1 coûte 1299.99 DT et 15 unités sont en stock."
    assert compute_nlp_metrics(text, text) == {"bleu_score": 1.0, "rouge_l_score": 1.0}


@pytest.mark.parametrize(
    ("actual", "reference"),
    [("réponse", None), ("réponse", ""), ("", "référence"), ("  ", "x")],
)
def test_missing_inputs_return_none(actual, reference):
    assert compute_nlp_metrics(actual, reference) == {"bleu_score": None, "rouge_l_score": None}


def test_partial_overlap_is_between_zero_and_one():
    scores = compute_nlp_metrics(
        "Le PhoneUltra 5G coûte 899.99 DT.",
        "Le PhoneUltra 5G est disponible à 899.99 DT avec 42 unités en stock.",
    )
    assert 0 < scores["bleu_score"] < 1
    assert 0 < scores["rouge_l_score"] < 1


def test_unrelated_answer_scores_lower_than_close_answer():
    reference = "La commande #1234 est en cours de livraison."
    close = compute_nlp_metrics("Votre commande #1234 est en cours de livraison.", reference)
    unrelated = compute_nlp_metrics("Il fait beau aujourd'hui.", reference)
    assert unrelated["rouge_l_score"] < close["rouge_l_score"]
    assert unrelated["bleu_score"] < close["bleu_score"]
