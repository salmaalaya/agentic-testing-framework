"""
src/agentic_testing/agents/evaluator.py — Agent Swarm #3 : Evaluator

Rôle : Pour chaque résultat d'exécution, scorer la réponse de l'agent
       sur 4 dimensions et rendre un verdict PASS/FAIL.

Modèle : llama3.2 (meilleur raisonnement pour le scoring nuancé)
Handoff : → reporter_agent (après scoring de tous les résultats)

Ce fichier expose :
  - evaluator_agent   : l'objet Agent Swarm prêt à l'emploi
  - score_result()    : appelle le LLM pour scorer un résultat individuel
  - deterministic_precheck() : règles rapides avant le scoring LLM
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from swarm import Agent

from agentic_testing.config import PROJECT_ROOT

# Métriques BLEU / ROUGE-L / NLI — réexportées pour les appelants existants
from agentic_testing.evaluation.metrics import NLI_MODEL_NAME, compute_nli_score, compute_nlp_metrics  # noqa: F401

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
_ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)

MODEL_EVALUATOR = os.getenv("MODEL_EVALUATOR", "llama3.2")
MIN_SCORE_PASS = float(os.getenv("MIN_SCORE_PASS", "3.0"))
TIMEOUT_APPIUM_S = int(os.getenv("MAX_LATENCY_MS", "30000")) // 1000  # converti en secondes

# ─── Chemins ───────────────────────────────────────────────────────────────────
ROOT = PROJECT_ROOT
RESULTS_PATH = ROOT / "reports" / "execution_results.json"
SCORES_PATH = ROOT / "reports" / "scored_results.json"


# ─── Pré-vérification déterministe (sans LLM) ─────────────────────────────────
def deterministic_precheck(result: dict) -> tuple[str | None, str | None]:
    """
    Vérifie les règles bloquantes AVANT d'appeler le LLM.
    Fonctionne pour les canaux api ET web.
    Retourne (verdict, reason) si un critère bloquant est déclenché,
    ou (None, None) si le LLM doit scorer.

    Règles bloquantes (dans l'ordre de priorité) :
    1. Erreur de connexion / Playwright           → ERROR immédiat
    2. HTTP >= seuil                              → FAIL immédiat
    3. Latence > max                              → FAIL immédiat
    4. Réponse vide                               → FAIL immédiat
    5. Mot obligatoire absent (contains/all)      → FAIL immédiat
    6. Aucun mot alternatif présent (contains_any)→ FAIL immédiat
    7. Mot interdit présent (not_contains)        → FAIL immédiat (sécurité)
    8. SKIP canal mobile                          → WARN
    """
    fc = result.get("failure_criteria", {})
    expected = result.get("expected", {})
    actual = (result.get("actual") or "").lower()
    status = result.get("status_code", 0)
    latency = result.get("latency_ms", 0)
    error = result.get("error") or ""

    # 1. Erreur de connexion / Playwright / Appium / SKIP
    if error:
        if "SKIP" in error:
            return "WARN", error
        if "APPIUM_UNAVAILABLE" in error:
            return "ERROR", "Appium server ou émulateur Android non démarré"
        if "APPIUM_ERROR:NoSuchElementException" in error:
            return "FAIL", "Élément UI Android introuvable — sélecteur invalide ou app non chargée"
        if "APPIUM_ERROR:TimeoutException" in error:
            return "FAIL", f"Timeout Appium — réponse app trop lente (>{TIMEOUT_APPIUM_S}s)"
        if "APPIUM_ERROR:" in error:
            return "FAIL", f"Erreur Appium : {error[13:100]}"
        if "LOGIN_FAILED" in error:
            return "FAIL", f"Flux login échoué : {error[13:100]}"
        return "ERROR", error

    # 2. Status code HTTP bloquant
    fail_if_gte = fc.get("fail_if_status_gte", 400)
    if status >= fail_if_gte:
        return "FAIL", f"HTTP {status} >= {fail_if_gte}"

    # 3. Latence dépassée
    max_latency = expected.get("max_latency_ms", 10000)
    if latency > max_latency:
        return "FAIL", f"Latence {latency}ms dépasse le max {max_latency}ms"

    # 4. Réponse vide
    if fc.get("fail_if_empty_response") and not actual.strip():
        return "FAIL", "Réponse vide inattendue"

    # 5. Mots obligatoires AND
    for kw in expected.get("contains", []):
        if kw.lower() not in actual:
            return "FAIL", f"Mot obligatoire absent : '{kw}'"
    for kw in expected.get("contains_all", []):
        if kw.lower() not in actual:
            return "FAIL", f"Mot obligatoire absent (contains_all) : '{kw}'"

    # 6. Mots alternatifs OR
    any_kws = expected.get("contains_any", [])
    if any_kws and not any(kw.lower() in actual for kw in any_kws):
        return "FAIL", f"Aucun des mots attendus trouvé : {any_kws}"

    # 7. Mots interdits — sécurité XSS / hallucination
    for kw in expected.get("not_contains", []):
        if kw.lower() in actual:
            return "FAIL", f"Mot interdit présent : '{kw}' (sécurité)"

    return None, None  # → LLM prend le relais


# ─── Fonction outil : scorer TOUS les résultats Web ──────────────────────────
def score_all_web_results(compute_nli: bool = False) -> str:
    """
    Charge reports/web_execution_results.json, score chaque résultat,
    sauvegarde dans reports/web_scored_results.json.
    Même logique que score_all_results() mais pour le canal web.

    compute_nli : si True, calcule aussi le score NLI (entailment) par cas —
    coût mesuré ~2.7s/cas sur CPU (modèle mDeBERTa-v3-base-mnli-xnli), soit
    largement > 30% de la durée totale d'un run typique. Désactivé par
    défaut (flag --nli de run_pipeline.py) — voir README section NLI.
    """
    web_results_path = ROOT / "reports" / "web_execution_results.json"
    web_scores_path = ROOT / "reports" / "web_scored_results.json"

    if not web_results_path.exists():
        return json.dumps(
            {
                "status": "ERROR",
                "message": "reports/web_execution_results.json introuvable. L'Executor Web doit tourner d'abord.",
            }
        )

    with open(web_results_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return json.dumps({"status": "ERROR", "message": "Aucun résultat web à scorer."})

    scored = []
    for result in results:
        scored_str = score_result(json.dumps(result), compute_nli=compute_nli)
        scored.append(json.loads(scored_str))

    total = len(scored)
    passed = sum(1 for s in scored if s.get("verdict") == "PASS")
    failed = sum(1 for s in scored if s.get("verdict") == "FAIL")
    warned = sum(1 for s in scored if s.get("verdict") == "WARN")
    errors = sum(1 for s in scored if s.get("verdict") == "ERROR")
    hallucs = sum(1 for s in scored if s.get("hallucination") is True)
    rate = round(passed / total * 100, 1) if total else 0

    avg_scores = {
        "pertinence": round(sum(s.get("pertinence", 0) for s in scored) / total, 2) if total else 0,
        "exactitude": round(sum(s.get("exactitude", 0) for s in scored) / total, 2) if total else 0,
        "coherence": round(sum(s.get("coherence", 0) for s in scored) / total, 2) if total else 0,
        "score_moyen": round(sum(s.get("score_moyen", 0) for s in scored) / total, 2) if total else 0,
    }

    output = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "errors": errors,
            "success_rate_pct": rate,
            "hallucinations_detected": hallucs,
            "canal": "web",
            "avg_scores": avg_scores,
        },
        "scored_results": scored,
    }

    web_scores_path.parent.mkdir(exist_ok=True)
    with open(web_scores_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return json.dumps(
        {
            "status": "OK",
            "total_scored": total,
            "success_rate_pct": rate,
            "passed": passed,
            "failed": failed,
            "hallucinations_detected": hallucs,
            "saved_to": web_scores_path.name,
            "message": f"{total} résultats web scorés. Taux: {rate}%. Handoff vers Reporter.",
        },
        ensure_ascii=False,
    )


# ─── Fonction outil : scorer TOUS les résultats Mobile ───────────────────────
def score_all_mobile_results(compute_nli: bool = False) -> str:
    """
    Charge reports/mobile_execution_results.json, score chaque résultat,
    sauvegarde dans reports/mobile_scored_results.json.

    Préchecks spécifiques Appium en plus des règles génériques :
    - APPIUM_UNAVAILABLE → ERROR (infrastructure)
    - APPIUM_ERROR:NoSuchElementException → FAIL (élément UI manquant)
    - APPIUM_ERROR:TimeoutException → FAIL (réponse trop lente)
    - login_success attendu mais absent → FAIL

    compute_nli : voir score_all_web_results() — désactivé par défaut.
    """
    mobile_results_path = ROOT / "reports" / "mobile_execution_results.json"
    mobile_scores_path = ROOT / "reports" / "mobile_scored_results.json"

    if not mobile_results_path.exists():
        return json.dumps(
            {
                "status": "ERROR",
                "message": "reports/mobile_execution_results.json introuvable. L'Executor Mobile doit tourner d'abord.",
            }
        )

    with open(mobile_results_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return json.dumps({"status": "ERROR", "message": "Aucun résultat mobile à scorer."})

    scored = []
    for result in results:
        # Enrichit l'error avec le canal pour le precheck
        enriched = {**result, "canal": "mobile"}
        scored_str = score_result(json.dumps(enriched), compute_nli=compute_nli)
        scored.append(json.loads(scored_str))

    total = len(scored)
    passed = sum(1 for s in scored if s.get("verdict") == "PASS")
    failed = sum(1 for s in scored if s.get("verdict") == "FAIL")
    warned = sum(1 for s in scored if s.get("verdict") == "WARN")
    errors = sum(1 for s in scored if s.get("verdict") == "ERROR")
    hallucs = sum(1 for s in scored if s.get("hallucination") is True)
    rate = round(passed / total * 100, 1) if total else 0

    avg_scores = {
        "pertinence": round(sum(s.get("pertinence", 0) for s in scored) / total, 2) if total else 0,
        "exactitude": round(sum(s.get("exactitude", 0) for s in scored) / total, 2) if total else 0,
        "coherence": round(sum(s.get("coherence", 0) for s in scored) / total, 2) if total else 0,
        "score_moyen": round(sum(s.get("score_moyen", 0) for s in scored) / total, 2) if total else 0,
    }

    output = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "errors": errors,
            "success_rate_pct": rate,
            "hallucinations_detected": hallucs,
            "canal": "mobile",
            "avg_scores": avg_scores,
        },
        "scored_results": scored,
    }

    mobile_scores_path.parent.mkdir(exist_ok=True)
    with open(mobile_scores_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return json.dumps(
        {
            "status": "OK",
            "total_scored": total,
            "success_rate_pct": rate,
            "passed": passed,
            "failed": failed,
            "hallucinations_detected": hallucs,
            "saved_to": mobile_scores_path.name,
            "message": f"{total} résultats mobile scorés. Taux: {rate}%.",
        },
        ensure_ascii=False,
    )


# ─── Fonction outil : scorer TOUS les résultats d'un coup ────────────────────
def score_all_results(compute_nli: bool = False) -> str:
    """
    Charge, score et sauvegarde TOUS les résultats en une seule fois.
    C'est l'outil PRINCIPAL de l'Evaluator — un seul appel suffit.

    compute_nli : voir score_all_web_results() — désactivé par défaut.

    Retourne un résumé JSON avec les statistiques globales.
    """
    # Chargement
    raw = load_execution_results()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # Message d'erreur

    results = data.get("results", [])
    if not results:
        return "ERREUR : aucun résultat à scorer dans execution_results.json."

    # Scoring de chaque résultat
    scored = []
    for result in results:
        scored_str = score_result(json.dumps(result), compute_nli=compute_nli)
        scored.append(json.loads(scored_str))

    # Calcul des statistiques globales
    total = len(scored)
    passed = sum(1 for s in scored if s.get("verdict") == "PASS")
    failed = sum(1 for s in scored if s.get("verdict") == "FAIL")
    warned = sum(1 for s in scored if s.get("verdict") == "WARN")
    errors = sum(1 for s in scored if s.get("verdict") == "ERROR")
    hallucs = sum(1 for s in scored if s.get("hallucination") is True)
    rate = round(passed / total * 100, 1) if total > 0 else 0

    avg_scores = {
        "pertinence": round(sum(s.get("pertinence", 0) for s in scored) / total, 2) if total else 0,
        "exactitude": round(sum(s.get("exactitude", 0) for s in scored) / total, 2) if total else 0,
        "coherence": round(sum(s.get("coherence", 0) for s in scored) / total, 2) if total else 0,
        "score_moyen": round(sum(s.get("score_moyen", 0) for s in scored) / total, 2) if total else 0,
    }

    output = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "errors": errors,
            "success_rate_pct": rate,
            "hallucinations_detected": hallucs,
            "avg_scores": avg_scores,
        },
        "scored_results": scored,
    }

    SCORES_PATH.parent.mkdir(exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return json.dumps(
        {
            "status": "OK",
            "total_scored": total,
            "success_rate_pct": rate,
            "passed": passed,
            "failed": failed,
            "hallucinations_detected": hallucs,
            "saved_to": SCORES_PATH.name,
            "message": f"{total} résultats scorés. Taux: {rate}%. Handoff vers Reporter.",
        },
        ensure_ascii=False,
    )


def load_execution_results() -> str:
    """
    Charge les résultats produits par l'Executor depuis reports/execution_results.json.
    Retourne leur contenu JSON sous forme de string.
    """
    if not RESULTS_PATH.exists():
        return "ERREUR : reports/execution_results.json introuvable. L'Executor doit tourner d'abord."
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    return json.dumps({"total": len(results), "results": results}, ensure_ascii=False)


# ─── Fonction outil : scorer UN résultat ─────────────────────────────────────
def score_result(result_json: str, compute_nli: bool = False) -> str:
    """
    Score un résultat individuel sur 4 dimensions via LLM (llama3.2).

    Applique d'abord les règles déterministes (sans LLM).
    Si aucune règle bloquante n'est déclenchée → appelle llama3.2 pour scorer.

    compute_nli : si True, calcule aussi nli_entailment_score (coût mesuré
    ~2.7s/cas sur CPU — voir README section NLI). Désactivé par défaut :
    BLEU/ROUGE-L restent toujours calculés (quasi-instantanés), seul le
    score NLI est derrière ce flag.

    Retourne un JSON avec :
    {
      "id": "TC-001",
      "name": "...",
      "canal": "api",
      "category": "nominal",
      "input": "...",
      "actual": "...",
      "pertinence": 4,       (1-5)
      "exactitude": 5,       (1-5)
      "coherence": 4,        (1-5)
      "hallucination": false,
      "score_moyen": 4.33,
      "verdict": "PASS",
      "reason": "...",
      "latency_ms": 12.5,
      "status_code": 200
    }
    """
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON invalide : {e}", "id": "?", "verdict": "ERROR"})

    tc_id = result.get("id", "?")
    tc_name = result.get("name", "?")
    canal = result.get("canal", "api")
    category = result.get("category", "?")
    actual = result.get("actual", "")
    expected = result.get("expected", {})

    # ── Métriques informatives : BLEU/ROUGE-L vs référence + score NLI ─────────
    # Calculées une seule fois, injectées dans TOUTES les branches de retour
    # ci-dessous (precheck, scoring LLM, fallback WARN/ERROR) — jamais
    # utilisées pour déterminer verdict/score_moyen. Le score NLI est le seul
    # des deux à être coûteux (~2.7s/cas) — opt-in via compute_nli.
    nlp_metrics = compute_nlp_metrics(actual, expected.get("reference_response"))
    nli_score = compute_nli_score(actual) if compute_nli else None

    # ── Étape 1 : pré-vérification déterministe ────────────────────────────────
    pre_verdict, pre_reason = deterministic_precheck(result)
    if pre_verdict is not None:
        return json.dumps(
            {
                "id": tc_id,
                "name": tc_name,
                "canal": canal,
                "category": category,
                "input": result.get("input", ""),
                "actual": actual,
                "pertinence": 0 if pre_verdict in ("FAIL", "ERROR") else 3,
                "exactitude": 0 if pre_verdict in ("FAIL", "ERROR") else 3,
                "coherence": 0 if pre_verdict in ("FAIL", "ERROR") else 3,
                "hallucination": False,
                "score_moyen": 0.0 if pre_verdict in ("FAIL", "ERROR") else 3.0,
                "verdict": pre_verdict,
                "reason": pre_reason,
                "latency_ms": result.get("latency_ms", 0),
                "status_code": result.get("status_code", 0),
                "notes": result.get("notes", ""),
                **nlp_metrics,
                "nli_entailment_score": nli_score,
            },
            ensure_ascii=False,
        )

    # ── Étape 2 : scoring LLM ─────────────────────────────────────────────────
    # Prépare le contexte pour le LLM
    expected_desc = []
    if expected.get("contains"):
        expected_desc.append(f"Doit contenir TOUS ces mots : {expected['contains']}")
    if expected.get("contains_all"):
        expected_desc.append(f"Doit contenir TOUS ces mots : {expected['contains_all']}")
    if expected.get("contains_any"):
        expected_desc.append(f"Doit contenir AU MOINS UN de ces mots : {expected['contains_any']}")
    if expected.get("not_contains"):
        expected_desc.append(f"Ne doit PAS contenir : {expected['not_contains']}")
    if expected.get("intent"):
        expected_desc.append(f"Intent attendu : {expected['intent']}")

    hallucination_risk = result.get("failure_criteria", {}).get("hallucination_risk", "medium")

    scoring_prompt = f"""Tu es un expert QA senior en agents conversationnels.
Évalue cette interaction selon les 4 critères suivants.

=== INTERACTION À ÉVALUER ===
ID du test    : {tc_id}
Catégorie     : {category}
Question user : {result.get("input", "")}
Réponse agent : {actual}
Intent détecté: {result.get("intent_actual", "?")} (confiance: {result.get("confidence", 0)})

=== CRITÈRES ATTENDUS ===
{chr(10).join(expected_desc) if expected_desc else "Vérification générale de la qualité"}
Risque hallucination : {hallucination_risk}

=== CRITÈRES DE SCORING ===
1. pertinence (1-5) : La réponse répond-elle directement à la question posée ?
   1=hors-sujet, 3=partiellement, 5=parfaitement ciblée

2. exactitude (1-5) : Les informations factuelles sont-elles correctes ?
   1=informations fausses, 3=partiellement correct, 5=exactitude parfaite
   ATTENTION : Si l'agent invente un prix, un stock ou un statut de commande → 1

3. coherence (1-5) : La réponse est-elle cohérente et bien formulée ?
   1=incohérente, 3=acceptable, 5=parfaitement structurée

4. hallucination (true/false) : L'agent a-t-il inventé des informations non présentes dans sa base ?
   true = FAIL immédiat (peu importe les autres scores)

=== RÈGLES DE VERDICT ===
- PASS si : hallucination=false ET score_moyen >= {MIN_SCORE_PASS}
- FAIL si : hallucination=true OU score_moyen < {MIN_SCORE_PASS}

=== RÉPONSE ATTENDUE ===
Retourne UNIQUEMENT ce JSON (sans texte autour, sans markdown) :
{{
  "pertinence": <entier 1-5>,
  "exactitude": <entier 1-5>,
  "coherence": <entier 1-5>,
  "hallucination": <true ou false>,
  "verdict": "<PASS ou FAIL>",
  "reason": "<explication concise en français, max 100 caractères>"
}}"""

    # Jusqu'à MAX_SCORE_ATTEMPTS tentatives : llama3.2 retourne parfois du
    # JSON invalide (texte parasite, JSON tronqué...) — un simple retry
    # suffit la plupart du temps, même logique que le retry du Generator.
    MAX_SCORE_ATTEMPTS = 2
    last_json_error = None

    for _attempt in range(1, MAX_SCORE_ATTEMPTS + 1):
        try:
            llm_response = _ollama_client.chat.completions.create(
                model=MODEL_EVALUATOR,
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.1,  # Faible pour des scores reproductibles
            )
            raw = llm_response.choices[0].message.content.strip()

            # ── Extraction robuste du JSON ───────────────────────────────────
            # llama3.2 ajoute parfois du texte avant/après le JSON.
            # On cherche le bloc { ... } avec regex pour extraire uniquement le JSON.

            # 1. Essai direct (réponse propre)
            json_str = raw

            # 2. Nettoie les balises ```json ... ```
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                end = len(lines) - 1
                while end > 0 and lines[end].strip() in ("```", ""):
                    end -= 1
                json_str = "\n".join(lines[1 : end + 1])

            # 3. Extrait le premier bloc JSON { ... } dans le texte (cas llama3.2)
            if not json_str.strip().startswith("{"):
                match = re.search(r"\{[^{}]*\}", json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)

            scores = json.loads(json_str)

            pertinence = int(scores.get("pertinence", 3))
            exactitude = int(scores.get("exactitude", 3))
            coherence = int(scores.get("coherence", 3))
            hallucination = bool(scores.get("hallucination", False))
            score_moyen = round((pertinence + exactitude + coherence) / 3, 2)

            # Le LLM peut se tromper sur le verdict — on le recalcule nous-mêmes
            # pour garantir la cohérence avec nos règles
            if hallucination or score_moyen < MIN_SCORE_PASS:
                verdict = "FAIL"
            else:
                verdict = "PASS"

            reason = scores.get("reason", "Scoré par LLM")

            return json.dumps(
                {
                    "id": tc_id,
                    "name": tc_name,
                    "canal": canal,
                    "category": category,
                    "input": result.get("input", ""),
                    "actual": actual,
                    "pertinence": pertinence,
                    "exactitude": exactitude,
                    "coherence": coherence,
                    "hallucination": hallucination,
                    "score_moyen": score_moyen,
                    "verdict": verdict,
                    "reason": reason,
                    "latency_ms": result.get("latency_ms", 0),
                    "status_code": result.get("status_code", 0),
                    "notes": result.get("notes", ""),
                    **nlp_metrics,
                    "nli_entailment_score": nli_score,
                },
                ensure_ascii=False,
            )

        except json.JSONDecodeError as e:
            last_json_error = e
            continue  # retente (si tentatives restantes), sinon tombe en WARN ci-dessous

        except Exception as e:
            return json.dumps(
                {
                    "id": tc_id,
                    "name": tc_name,
                    "canal": canal,
                    "category": category,
                    "input": result.get("input", ""),
                    "actual": actual,
                    "pertinence": 0,
                    "exactitude": 0,
                    "coherence": 0,
                    "hallucination": False,
                    "score_moyen": 0.0,
                    "verdict": "ERROR",
                    "reason": f"Erreur LLM : {str(e)[:80]}",
                    "latency_ms": result.get("latency_ms", 0),
                    "status_code": result.get("status_code", 0),
                    "notes": result.get("notes", ""),
                    **nlp_metrics,
                    "nli_entailment_score": nli_score,
                },
                ensure_ascii=False,
            )

    # Toutes les tentatives ont échoué à produire du JSON valide — score
    # neutre WARN plutôt qu'un FAIL/ERROR trompeur.
    return json.dumps(
        {
            "id": tc_id,
            "name": tc_name,
            "canal": canal,
            "category": category,
            "input": result.get("input", ""),
            "actual": actual,
            "pertinence": 3,
            "exactitude": 3,
            "coherence": 3,
            "hallucination": False,
            "score_moyen": 3.0,
            "verdict": "WARN",
            "reason": f"LLM n'a pas retourné du JSON valide après {MAX_SCORE_ATTEMPTS} tentatives — score neutre appliqué ({last_json_error})",
            "latency_ms": result.get("latency_ms", 0),
            "status_code": result.get("status_code", 0),
            "notes": result.get("notes", ""),
            **nlp_metrics,
            "nli_entailment_score": nli_score,
        },
        ensure_ascii=False,
    )


# ─── Fonction outil : sauvegarder les scores ─────────────────────────────────
def save_scored_results(scored_results_json: str) -> str:
    """
    Sauvegarde tous les résultats scorés dans reports/scored_results.json.

    Paramètre : scored_results_json — JSON string avec {"scored_results": [...]}
    Retourne  : confirmation avec statistiques (taux de réussite, PASS/FAIL/WARN)
    """
    try:
        cleaned = scored_results_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        data = json.loads(cleaned)

        if isinstance(data, list):
            scored = data
        elif isinstance(data, dict) and "scored_results" in data:
            scored = data["scored_results"]
        else:
            return "ERREUR : format inattendu. Clé 'scored_results' manquante."

        # Calcule les statistiques globales
        total = len(scored)
        passed = sum(1 for s in scored if s.get("verdict") == "PASS")
        failed = sum(1 for s in scored if s.get("verdict") == "FAIL")
        warned = sum(1 for s in scored if s.get("verdict") == "WARN")
        errors = sum(1 for s in scored if s.get("verdict") == "ERROR")
        hallucs = sum(1 for s in scored if s.get("hallucination") is True)
        rate = round(passed / total * 100, 1) if total > 0 else 0

        avg_scores = {
            "pertinence": round(sum(s.get("pertinence", 0) for s in scored) / total, 2) if total else 0,
            "exactitude": round(sum(s.get("exactitude", 0) for s in scored) / total, 2) if total else 0,
            "coherence": round(sum(s.get("coherence", 0) for s in scored) / total, 2) if total else 0,
            "score_moyen": round(sum(s.get("score_moyen", 0) for s in scored) / total, 2) if total else 0,
        }

        output = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warned": warned,
                "errors": errors,
                "success_rate_pct": rate,
                "hallucinations_detected": hallucs,
                "avg_scores": avg_scores,
            },
            "scored_results": scored,
        }

        SCORES_PATH.parent.mkdir(exist_ok=True)
        with open(SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return (
            f"OK — {total} résultats scorés sauvegardés dans {SCORES_PATH.name}.\n"
            f"Taux de réussite : {rate}% ({passed}/{total})\n"
            f"PASS:{passed}  FAIL:{failed}  WARN:{warned}  ERROR:{errors}\n"
            f"Hallucinations détectées : {hallucs}\n"
            f"Scores moyens : pertinence={avg_scores['pertinence']} "
            f"exactitude={avg_scores['exactitude']} "
            f"coherence={avg_scores['coherence']}\n"
            f"Tu peux maintenant faire le handoff vers le Reporter."
        )
    except Exception as e:
        return f"ERREUR lors de la sauvegarde : {e}"


# ─── Handoff ──────────────────────────────────────────────────────────────────
def handoff_to_reporter():
    """Handoff vers le Reporter Agent une fois tous les scores calculés."""
    from agentic_testing.agents.reporter import reporter_agent

    return reporter_agent


EVALUATOR_INSTRUCTIONS = """
Tu es un expert QA senior en agents conversationnels.
Ton rôle est de scorer tous les résultats d'exécution.

PROCÉDURE EN 2 ÉTAPES SEULEMENT :

1. Appelle score_all_results() — cet outil fait TOUT en un seul appel :
   - Il charge tous les résultats de l'Executor
   - Il score chaque résultat (pertinence, exactitude, cohérence, hallucination)
   - Il sauvegarde les scores automatiquement

2. Quand score_all_results retourne "status": "OK", fais immédiatement le handoff
   vers le Reporter en appelant handoff_to_reporter().

C'EST TOUT. Ne boucle pas sur les résultats individuellement.
Un seul appel à score_all_results suffit.
"""

evaluator_agent = Agent(
    name="Evaluator",
    model=MODEL_EVALUATOR,
    instructions=EVALUATOR_INSTRUCTIONS,
    functions=[
        score_all_results,
        score_all_web_results,
        score_all_mobile_results,
        load_execution_results,
        score_result,
        save_scored_results,
        handoff_to_reporter,
    ],
)
