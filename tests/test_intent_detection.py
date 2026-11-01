"""Keyword intent detection of the target app: whole-word matching, not substrings."""

import pytest

from target_app.app import _is_fast_path_eligible, _keyword_intent


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("Bonjour", "greeting"),
        ("Où en est ma commande #1234 ?", "order_tracking"),
        ("Mon casque est cassé, je veux un remboursement", "complaint"),
        ("Quel temps fait-il ? météo", "out_of_domain"),
        ("Quel est le prix du laptop ?", "product_query"),
        ("merci", "farewell"),
        ("", "empty_input"),
        ("   ", "empty_input"),
        ("xyzzy", "unknown"),
    ],
)
def test_detects_expected_intent(message, intent):
    assert _keyword_intent(message)[0] == intent


def test_sav_inside_savoir_is_not_a_complaint():
    # Regression: substring matching used to find "sav" (after-sales service) inside "savoir".
    assert _keyword_intent("je veux savoir les prix")[0] == "product_query"


def test_hi_inside_chinois_is_not_a_greeting():
    assert _keyword_intent("vous vendez du matériel chinois ?")[0] != "greeting"


def test_confidence_is_within_documented_range():
    _, confidence = _keyword_intent("Bonjour")
    assert 0.72 <= confidence <= 0.88


def test_specific_product_question_uses_fast_path():
    assert _is_fast_path_eligible("Prix du PhoneUltra 5G ?", "product_query", 0.8)


def test_long_sarcastic_generic_question_goes_to_llm():
    message = "Super, encore un produit génial qui est cassé dès le premier jour, bravo vraiment !!"
    assert not _is_fast_path_eligible(message, "product_query", 0.8)
