"""
evaluation/metrics.py — Métriques informatives de l'Evaluator

  - compute_nlp_metrics() : BLEU (nltk) + ROUGE-L (rouge-score) vs réponse de référence
  - compute_nli_score()   : score d'entailment NLI (transformers, optionnel)

Aucune de ces métriques ne modifie le verdict PASS/FAIL.
"""

import os

# ─── Métriques NLP : BLEU / ROUGE-L vs réponse de référence ───────────────────
# Purement INFORMATIF (exigence du projet, section Métriques) — ne modifie
# jamais le verdict PASS/FAIL, qui reste basé sur hallucination + score_moyen
# + status_code. N'apparaît que pour les cas dont expected.reference_response
# est renseigné (champ optionnel dans le corpus).
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from agentic_testing.config import ensure_project_root_on_path

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
_bleu_smoothing = SmoothingFunction().method1


def compute_nlp_metrics(actual: str, reference: str | None) -> dict:
    """
    Calcule BLEU (nltk) et ROUGE-L (rouge-score) entre la réponse obtenue
    (actual) et une réponse de référence idéale (reference). Tokenisation
    simple (lower + split) — évite une dépendance au téléchargement des
    données nltk (punkt), non nécessaire pour un score approximatif informatif.

    Retourne {"bleu_score": None, "rouge_l_score": None} si reference est
    absent/vide ou en cas d'erreur — jamais d'exception propagée.
    """
    if not reference or not actual or not actual.strip():
        return {"bleu_score": None, "rouge_l_score": None}
    try:
        actual_tokens = actual.lower().split()
        reference_tokens = reference.lower().split()
        bleu = sentence_bleu([reference_tokens], actual_tokens, smoothing_function=_bleu_smoothing)
        rouge_l = _rouge_scorer.score(reference, actual)["rougeL"].fmeasure
        return {"bleu_score": round(bleu, 3), "rouge_l_score": round(rouge_l, 3)}
    except Exception:
        return {"bleu_score": None, "rouge_l_score": None}


# ─── Score NLI (entailment) : renfort de la détection d'hallucination ─────────
# EN PLUS du jugement LLM (llama3.2) existant sur "hallucination" — ne le
# remplace pas. Modèle chargé paresseusement (1er appel) et mis en cache ;
# dégrade proprement (nli_entailment_score=None) si torch/transformers ne
# sont pas installés OU si le modèle ne peut pas se charger sur cette
# machine (ex: Visual C++ Redistributable manquant sous Windows — cause
# réelle rencontrée en dev, cf. requirements.txt).
NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

_nli_state = {"loaded": False, "tokenizer": None, "model": None, "labels": None}


def _get_nli_model():
    """Charge le modèle NLI au premier appel ; renvoie (None, None, None) si indisponible."""
    if _nli_state["loaded"]:
        return _nli_state["tokenizer"], _nli_state["model"], _nli_state["labels"]

    _nli_state["loaded"] = True  # on ne retente pas à chaque appel si ça échoue
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        model.eval()
        # id2label typique XNLI : {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        labels = {i: str(lbl).lower() for i, lbl in model.config.id2label.items()}
        _nli_state["tokenizer"] = tokenizer
        _nli_state["model"] = model
        _nli_state["labels"] = labels
    except Exception as e:
        print(f"[Evaluator] Score NLI désactivé (modèle indisponible) : {e}")

    return _nli_state["tokenizer"], _nli_state["model"], _nli_state["labels"]


def compute_nli_score(hypothesis: str) -> float | None:
    """
    Calcule un score d'entailment (0.0-1.0) entre le contexte factuel connu
    (catalogue produits + commandes, cf. target_app/knowledge_base.py) et la
    réponse de l'agent testé (hypothesis). Un score bas signale que la
    réponse contredit ou n'est pas soutenue par les faits connus — un signal
    de renfort pour la détection d'hallucination, EN PLUS du jugement LLM.

    Retourne None si le modèle n'est pas disponible sur cette machine
    (dépendance optionnelle — le reste du pipeline continue de fonctionner).
    """
    if not hypothesis or not hypothesis.strip():
        return None

    tokenizer, model, labels = _get_nli_model()
    if tokenizer is None or model is None:
        return None

    try:
        import torch

        ensure_project_root_on_path()
        from target_app.knowledge_base import build_factual_context

        premise = build_factual_context()
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        entailment_idx = next((i for i, name in labels.items() if "entail" in name), None)
        if entailment_idx is None:
            return None
        return round(probs[entailment_idx].item(), 3)
    except Exception as e:
        print(f"[Evaluator] Erreur calcul score NLI (ignorée) : {e}")
        return None
