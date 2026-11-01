"""
src/agentic_testing/executors/web.py — Exécution Playwright (Browser Use) — Canal Web S5-S6

Rôle : Pour chaque cas de test canal="web", ouvrir l'interface chat dans
       Chromium headless, envoyer le message via le formulaire, attendre
       la réponse du bot, capturer un screenshot, et retourner le résultat.

Approche CODE-BASED (pas tool-based) :
  L'Executor génère du code Playwright qui interagit directement avec le DOM.
  1 seul contexte de navigateur partagé → pas de relancement entre les tests.

Sélecteurs CSS stables (définis dans chat.html) :
  #chat-input       → champ de saisie
  #send-button      → bouton d'envoi
  .bot-message      → wrapper d'un message bot
  .message-bubble   → texte de la bulle
  #typing-indicator → spinner d'attente (doit disparaître)

Usage autonome (hors pipeline Swarm) :
    python -m agentic_testing.executors.web
"""

import json
import os
import time

import httpx
from dotenv import load_dotenv

from agentic_testing.config import CORPUS_DIR, REPORTS_DIR, SCREENSHOTS_DIR, relative_to_root

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
AGENT_WEB_URL = os.getenv("AGENT_WEB_URL", "http://localhost:8000")
WEB_CORPUS_PATH = CORPUS_DIR / "web_test_cases.json"
WEB_RESULTS_PATH = REPORTS_DIR / "web_execution_results.json"

# Timeouts Playwright (ms)
TIMEOUT_NAVIGATION = 20_000  # Chargement de la page
TIMEOUT_BOT_REPLY = 30_000  # Attente de la réponse bot (Ollama LLM peut prendre ~15s)
TIMEOUT_SPINNER_OFF = 25_000  # Attente que le spinner disparaisse


# ─── Contexte Playwright partagé ──────────────────────────────────────────────
# On garde un seul navigateur + contexte pour tous les tests.
# Cela économise ~800ms par test (pas de relancement chromium).
_browser = None
_context = None
_page = None


def _get_playwright_page():
    """
    Retourne la page Playwright active, en créant le navigateur si nécessaire.
    Le navigateur est headless par défaut (CI-friendly).
    Mettre PLAYWRIGHT_HEADLESS=false dans .env pour le voir pendant les tests.
    """
    global _browser, _context, _page
    from playwright.sync_api import sync_playwright

    headless = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() != "false"

    if _browser is None:
        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(headless=headless)
        _context = _browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="fr-FR",
        )
        _page = _context.new_page()
        # Charge la page initiale une seule fois (warmup)
        _page.goto(AGENT_WEB_URL, timeout=TIMEOUT_NAVIGATION, wait_until="networkidle")

    return _page


def close_browser():
    """Ferme proprement le navigateur — appeler en fin de suite de tests."""
    global _browser, _context, _page
    if _browser:
        _browser.close()
        _browser = _context = _page = None


# ─── Exécution d'un cas de test Web ───────────────────────────────────────────


def execute_web_test(test_case: dict) -> dict:
    """
    Exécute un cas de test via Playwright sur l'interface chat.

    Workflow :
      1. Recharge la page (état propre)
      2. Compte les bulles bot existantes
      3. Tape le message dans #chat-input
      4. Clique sur #send-button
      5. Attend qu'une nouvelle bulle .bot-message apparaisse
      6. Vérifie que le spinner #typing-indicator a disparu
      7. Lit le texte de la dernière bulle .message-bubble
      8. Prend un screenshot

    Retourne un dict au même format que execute_test() dans executor.py.
    """
    tc_id = test_case.get("id", "?")
    tc_name = test_case.get("name", "?")
    tc_input = test_case.get("input", "")

    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    screenshot_path = SCREENSHOTS_DIR / test_case.get("expected", {}).get("screenshot", f"{tc_id}.png")

    base_result = {
        "id": tc_id,
        "name": tc_name,
        "canal": "web",
        "category": test_case.get("category", "?"),
        "input": tc_input,
        "expected": test_case.get("expected", {}),
        "failure_criteria": test_case.get("failure_criteria", {}),
        "notes": test_case.get("notes", ""),
        "api_mirror": test_case.get("api_mirror", ""),
        "screenshot": relative_to_root(screenshot_path),
        "actual": "",
        "intent_actual": "?",
        "confidence": 0,
        "status_code": 200,
        "latency_ms": 0,
        "error": None,
    }

    try:
        page = _get_playwright_page()

        # ── 1. Recharge la page pour repartir d'un état propre ─────────────────
        page.reload(timeout=TIMEOUT_NAVIGATION, wait_until="networkidle")

        # ── 2. Compte les messages bot existants (le message de bienvenue) ──────
        initial_bot_count = page.locator(".bot-message").count()

        # ── 3. Tape le message (même vide pour TC-004 style) ────────────────────
        start = time.perf_counter()
        chat_input = page.locator("#chat-input")
        chat_input.click()
        if tc_input:
            # fill() efface d'abord, puis remplit — plus fiable que type()
            chat_input.fill(tc_input)
        # Pour input vide : on laisse le champ vide et on clique envoyer quand même

        # ── 4. Clique sur le bouton envoyer ────────────────────────────────────
        page.locator("#send-button").click()

        # ── 5. Attend une nouvelle bulle bot ────────────────────────────────────
        page.wait_for_function(
            f"document.querySelectorAll('.bot-message').length > {initial_bot_count}",
            timeout=TIMEOUT_BOT_REPLY,
        )

        # ── 6. Attend que le spinner disparaisse ────────────────────────────────
        page.wait_for_selector(
            "#typing-indicator.hidden",
            timeout=TIMEOUT_SPINNER_OFF,
            state="attached",
        )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # ── 7. Lit le texte de la dernière bulle bot ────────────────────────────
        last_bubble = page.locator(".bot-message .message-bubble").last
        actual_text = last_bubble.inner_text()

        # Attributs data enrichis (intent, confidence depuis data-* attrs)
        last_bot_msg = page.locator(".bot-message").last
        intent_actual = last_bot_msg.get_attribute("data-intent") or "?"
        confidence_raw = last_bot_msg.get_attribute("data-confidence") or "0"
        try:
            confidence = float(confidence_raw)
        except ValueError:
            confidence = 0.0

        # ── 8. Screenshot ────────────────────────────────────────────────────────
        page.screenshot(path=str(screenshot_path))

        return {
            **base_result,
            "actual": actual_text,
            "intent_actual": intent_actual,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        # Screenshot de debug en cas d'échec
        try:
            page = _get_playwright_page()
            error_ss = SCREENSHOTS_DIR / f"{tc_id}_ERROR.png"
            page.screenshot(path=str(error_ss))
        except Exception:
            pass

        return {
            **base_result,
            "error": f"PLAYWRIGHT_ERROR: {type(e).__name__}: {str(e)[:200]}",
        }


# ─── Exécution de tous les cas Web ────────────────────────────────────────────


def execute_all_web_tests() -> str:
    """
    Charge corpus/web_test_cases.json, exécute tous les cas via Playwright,
    sauvegarde les résultats dans reports/web_execution_results.json.

    Retourne un résumé JSON (status, total, errors).
    """
    if not WEB_CORPUS_PATH.exists():
        return json.dumps(
            {
                "status": "ERROR",
                "message": f"Corpus web introuvable : {WEB_CORPUS_PATH}",
            }
        )

    with open(WEB_CORPUS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    if not cases:
        return json.dumps({"status": "ERROR", "message": "Aucun cas web trouvé."})

    # Vérifie que l'agent web est disponible avant de lancer Chromium
    try:
        r = httpx.get(f"{AGENT_WEB_URL}/health", timeout=5.0)
        if r.status_code >= 500:
            return json.dumps(
                {
                    "status": "ERROR",
                    "message": f"Agent web retourne HTTP {r.status_code}. Démarrer : uvicorn target_app.app:app --reload --port 8000",
                }
            )
    except httpx.ConnectError:
        return json.dumps(
            {
                "status": "ERROR",
                "message": f"Agent web inaccessible sur {AGENT_WEB_URL}. Démarrer : uvicorn target_app.app:app --reload --port 8000",
            }
        )

    results = []
    errors_count = 0

    for tc in cases:
        result = execute_web_test(tc)
        results.append(result)
        if result.get("error"):
            errors_count += 1

    # Ferme le navigateur proprement
    close_browser()

    # Sauvegarde
    WEB_RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(WEB_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    return json.dumps(
        {
            "status": "OK",
            "total_executed": len(results),
            "errors": errors_count,
            "saved_to": WEB_RESULTS_PATH.name,
            "screenshots_dir": relative_to_root(SCREENSHOTS_DIR),
            "message": f"{len(results)} cas web exécutés, {errors_count} erreur(s).",
        },
        ensure_ascii=False,
    )


# ─── Point d'entrée autonome ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel(
            "[bold cyan]Executor Web — Playwright Browser Use[/bold cyan]\n"
            f"Cible : {AGENT_WEB_URL}\n"
            f"Corpus : {WEB_CORPUS_PATH.name}",
            border_style="cyan",
        )
    )

    result_str = execute_all_web_tests()
    result = json.loads(result_str)

    if result["status"] == "OK":
        console.print(f"\n[green]✓ {result['message']}[/green]")
        console.print(f"  Résultats : reports/{result['saved_to']}")
        console.print(f"  Screenshots : {result['screenshots_dir']}")
    else:
        console.print(f"\n[red]✗ {result['message']}[/red]")
        sys.exit(1)
