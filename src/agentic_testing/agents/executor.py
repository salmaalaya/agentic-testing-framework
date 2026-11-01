"""
src/agentic_testing/agents/executor.py — Agent Swarm #2 : Executor

Rôle : Charger les cas de test (manuels ou générés), les exécuter sur l'agent
       cible via le bon canal (api / web / mobile), mesurer la latence et
       capturer les réponses brutes.

Modèle : mistral (phi3 ne supporte pas les tool calls Ollama)
Handoff : → evaluator_agent (après exécution de tous les cas)

Canaux supportés :
  - api    → HTTP POST /chat  (httpx, le plus rapide)
  - web    → Playwright (Browser Use, DOM automation)
  - mobile → Appium (code-based, S5-S6 phase 2)

Ce fichier expose :
  - executor_agent     : l'objet Agent Swarm prêt à l'emploi
  - execute_all_tests(): exécute tous les cas d'un corpus en un appel
  - execute_test()     : exécute UN cas de test (route par canal)
"""

import json
import os
import time

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from swarm import Agent

from agentic_testing.config import PROJECT_ROOT

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
_ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)

MODEL_EXECUTOR = os.getenv("MODEL_EXECUTOR", "mistral")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000/chat")
AGENT_BASE_URL = AGENT_API_URL.rsplit("/chat", 1)[0]  # http://localhost:8000

# ─── Chemins ───────────────────────────────────────────────────────────────────
ROOT = PROJECT_ROOT
CORPUS_GENERATED_PATH = ROOT / "corpus" / "generated_test_cases.json"
CORPUS_MANUAL_PATH = ROOT / "corpus" / "test_cases.json"
CORPUS_WEB_PATH = ROOT / "corpus" / "web_test_cases.json"
CORPUS_MOBILE_PATH = ROOT / "corpus" / "mobile_test_cases.json"
RESULTS_PATH = ROOT / "reports" / "execution_results.json"
WEB_RESULTS_PATH = ROOT / "reports" / "web_execution_results.json"
MOBILE_RESULTS_PATH = ROOT / "reports" / "mobile_execution_results.json"

# ─── Client HTTP persistant (module-level, réutilisé entre les appels) ────────
# Un seul client = une seule connexion TCP = pas d'overhead Windows par requête
_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(
            base_url=AGENT_BASE_URL,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
        )
        # Warmup : absorbe le coût TCP du premier appel (Windows ~2s)
        try:
            _http_client.get("/health")
        except Exception:
            pass  # L'agent n'est peut-être pas encore démarré, on continue
    return _http_client


# ─── Fonction interne : charger les cas de test ───────────────────────────────
def load_test_cases(source: str = "manual") -> str:
    """
    Charge les cas de test depuis le corpus.
    source : "manual" | "generated" | "both" | "web"
    Utilisée par execute_all_tests().
    """
    cases = []
    if source == "web":
        if CORPUS_WEB_PATH.exists():
            with open(CORPUS_WEB_PATH, encoding="utf-8") as f:
                cases.extend(json.load(f).get("test_cases", []))
        else:
            return "ERREUR : corpus/web_test_cases.json introuvable."
    elif source == "mobile":
        if CORPUS_MOBILE_PATH.exists():
            with open(CORPUS_MOBILE_PATH, encoding="utf-8") as f:
                cases.extend(json.load(f).get("test_cases", []))
        else:
            return "ERREUR : corpus/mobile_test_cases.json introuvable."
    else:
        if source in ("generated", "both"):
            if CORPUS_GENERATED_PATH.exists():
                with open(CORPUS_GENERATED_PATH, encoding="utf-8") as f:
                    cases.extend(json.load(f).get("test_cases", []))
            elif source == "generated":
                return "ERREUR : corpus/generated_test_cases.json introuvable."
        if source in ("manual", "both"):
            if CORPUS_MANUAL_PATH.exists():
                with open(CORPUS_MANUAL_PATH, encoding="utf-8") as f:
                    cases.extend(json.load(f).get("test_cases", []))
    if not cases:
        return "ERREUR : aucun cas de test trouvé."
    return json.dumps({"total": len(cases), "test_cases": cases}, ensure_ascii=False)


# ─── Fonction outil : exécuter TOUS les cas de test d'un coup ─────────────────
def execute_all_tests(source: str = "manual") -> str:
    """
    Charge ET exécute TOUS les cas de test en une seule fois.
    Sauvegarde les résultats dans reports/execution_results.json (api)
    ou reports/web_execution_results.json (web).

    C'est l'outil PRINCIPAL de l'Executor — un seul appel suffit.

    Paramètre source :
      - "manual"    → cas API manuels du corpus (23 scénarios)
      - "generated" → cas du Generator
      - "both"      → fusion manual + generated
      - "web"       → cas Web Playwright du corpus (15 scénarios)
    """
    # Pour le canal web, on délègue entièrement à executor_web
    if source == "web":
        from agentic_testing.executors.web import execute_all_web_tests

        return execute_all_web_tests()

    # Pour le canal mobile, on délègue à executor_mobile
    if source == "mobile":
        from agentic_testing.executors.mobile import execute_all_mobile_tests

        return execute_all_mobile_tests()

    # Chargement des cas API
    raw = load_test_cases(source=source)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    cases = data.get("test_cases", [])
    if not cases:
        return "ERREUR : aucun cas de test trouvé."

    results = []
    errors_count = 0
    for tc in cases:
        result_str = execute_test(json.dumps(tc))
        result = json.loads(result_str)
        results.append(result)
        if result.get("error"):
            errors_count += 1

    out_path = RESULTS_PATH
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    return json.dumps(
        {
            "status": "OK",
            "total_executed": len(results),
            "errors": errors_count,
            "saved_to": out_path.name,
            "message": f"{len(results)} cas exécutés, {errors_count} erreur(s). Handoff vers Evaluator.",
        },
        ensure_ascii=False,
    )


# ─── Fonction outil : exécuter UN cas de test ─────────────────────────────────
def execute_test(test_case_json: str) -> str:
    """
    Envoie UN cas de test à l'agent cible via le canal approprié.

    Canal "api"    → HTTP POST /chat (httpx, synchrone)
    Canal "web"    → Playwright (browser use, DOM + screenshot)
    Canal "mobile" → Appium (non implémenté, retourne SKIP)

    Paramètre : test_case_json — JSON string d'un seul cas de test
    Retourne  : JSON string avec le résultat (actual, latency_ms, status_code, error)
    """
    try:
        tc = json.loads(test_case_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON invalide : {e}", "id": "?", "actual": ""})

    canal = tc.get("canal", "api")

    # ── Routing par canal ─────────────────────────────────────────────────────
    if canal == "web":
        from agentic_testing.executors.web import execute_web_test

        return json.dumps(execute_web_test(tc), ensure_ascii=False)

    if canal == "mobile":
        from agentic_testing.executors.mobile import execute_mobile_test

        return json.dumps(execute_mobile_test(tc), ensure_ascii=False)

    # ── Canal API (défaut) ───────────────────────────────────────────────────

    sid = f"swarm_{tc.get('id', 'unknown')}"
    client = _get_client()

    # Gestion session expirée (TC-015 et équivalents)
    if tc.get("session_state") == "expired":
        try:
            client.delete(f"/session/{sid}")
        except Exception:
            pass

    # Construction du payload
    payload: dict = {
        "message": tc.get("input", ""),
        "session_id": sid,
    }
    if tc.get("context"):
        payload["context"] = tc["context"]

    start = time.perf_counter()
    try:
        response = client.post("/chat", json=payload)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if response.status_code == 200:
            body = response.json()
            result = {
                "id": tc.get("id", "?"),
                "name": tc.get("name", "?"),
                "canal": tc.get("canal", "api"),
                "category": tc.get("category", "?"),
                "input": tc.get("input", ""),
                "expected": tc.get("expected", {}),
                "failure_criteria": tc.get("failure_criteria", {}),
                "notes": tc.get("notes", ""),
                "actual": body.get("response", ""),
                "intent_actual": body.get("intent", "?"),
                "confidence": body.get("confidence", 0),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "error": None,
            }
        else:
            result = {
                "id": tc.get("id", "?"),
                "name": tc.get("name", "?"),
                "canal": tc.get("canal", "api"),
                "category": tc.get("category", "?"),
                "input": tc.get("input", ""),
                "expected": tc.get("expected", {}),
                "failure_criteria": tc.get("failure_criteria", {}),
                "notes": tc.get("notes", ""),
                "actual": "",
                "intent_actual": "?",
                "confidence": 0,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}",
            }

        return json.dumps(result, ensure_ascii=False)

    except httpx.ConnectError:
        return json.dumps(
            {
                "id": tc.get("id", "?"),
                "name": tc.get("name", "?"),
                "canal": tc.get("canal", "api"),
                "category": tc.get("category", "?"),
                "input": tc.get("input", ""),
                "expected": tc.get("expected", {}),
                "failure_criteria": tc.get("failure_criteria", {}),
                "notes": tc.get("notes", ""),
                "actual": "",
                "intent_actual": "?",
                "confidence": 0,
                "status_code": 0,
                "latency_ms": 0,
                "error": "CONNECTION_ERROR — agent non démarré ? Lance : uvicorn target_app.app:app --reload --port 8000",
            },
            ensure_ascii=False,
        )

    except httpx.TimeoutException:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return json.dumps(
            {
                "id": tc.get("id", "?"),
                "name": tc.get("name", "?"),
                "canal": tc.get("canal", "api"),
                "category": tc.get("category", "?"),
                "input": tc.get("input", ""),
                "expected": tc.get("expected", {}),
                "failure_criteria": tc.get("failure_criteria", {}),
                "notes": tc.get("notes", ""),
                "actual": "",
                "intent_actual": "?",
                "confidence": 0,
                "status_code": 0,
                "latency_ms": latency_ms,
                "error": f"TIMEOUT après {latency_ms}ms",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps(
            {
                "id": tc.get("id", "?"),
                "name": tc.get("name", "?"),
                "canal": tc.get("canal", "api"),
                "category": tc.get("category", "?"),
                "input": tc.get("input", ""),
                "expected": tc.get("expected", {}),
                "failure_criteria": tc.get("failure_criteria", {}),
                "notes": tc.get("notes", ""),
                "actual": "",
                "intent_actual": "?",
                "confidence": 0,
                "status_code": 0,
                "latency_ms": 0,
                "error": str(e),
            },
            ensure_ascii=False,
        )


# ─── Fonction outil : sauvegarder tous les résultats d'exécution ──────────────
def save_execution_results(results_json: str) -> str:
    """
    Sauvegarde la liste complète des résultats d'exécution.

    Paramètre : results_json — JSON string avec {"results": [...]}
    Retourne  : confirmation + statistiques basiques
    """
    try:
        cleaned = results_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        data = json.loads(cleaned)

        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            return "ERREUR : format inattendu. Clé 'results' manquante."

        RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)

        errors = [r for r in results if r.get("error")]
        return (
            f"OK — {len(results)} résultats sauvegardés dans {RESULTS_PATH.name}. "
            f"{len(errors)} erreur(s) de connexion détectée(s). "
            f"Tu peux maintenant faire le handoff vers l'Evaluator."
        )
    except Exception as e:
        return f"ERREUR lors de la sauvegarde : {e}"


# ─── Handoff ──────────────────────────────────────────────────────────────────
def handoff_to_evaluator():
    """Handoff vers l'Evaluator Agent une fois tous les tests exécutés."""
    from agentic_testing.agents.evaluator import evaluator_agent

    return evaluator_agent


EXECUTOR_INSTRUCTIONS = """
Tu es un ingénieur QA automation spécialisé dans les tests d'API REST.
Ton rôle est d'exécuter tous les cas de test sur l'agent cible et de collecter les résultats.

PROCÉDURE EN 2 ÉTAPES SEULEMENT :

1. Appelle execute_all_tests(source="manual") — cet outil fait TOUT en un seul appel :
   - Il charge les cas de test manuels du corpus (corpus/test_cases.json)
   - Il les exécute tous sur l'agent cible
   - Il sauvegarde les résultats automatiquement

2. Quand execute_all_tests retourne "status": "OK", fais immédiatement le handoff
   vers l'Evaluator en appelant handoff_to_evaluator().

C'EST TOUT. Ne fais pas autre chose. Ne boucle pas sur les cas individuellement.
Un seul appel à execute_all_tests suffit.
"""

executor_agent = Agent(
    name="Executor",
    model=MODEL_EXECUTOR,
    instructions=EXECUTOR_INSTRUCTIONS,
    functions=[execute_all_tests, execute_test, load_test_cases, save_execution_results, handoff_to_evaluator],
)
