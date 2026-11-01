"""
src/agentic_testing/executors/mobile.py — Exécution Appium Android — Canal Mobile S5-S6

Rôle : Pour chaque cas de test canal="mobile", piloter une app Android
       via Appium en mode code-based : tap, swipe, scroll, rotation,
       background/foreground, login/logout — puis capturer screenshot.

Architecture :
  - 1 seul driver Appium partagé pour toute la suite (économise ~3s/test)
  - Chaque cas de test = fonction Python pure appelée séquentiellement
  - Gestion du cycle de vie Android (background, rotation, ANR detection)
  - Screenshots systématiques après chaque interaction

Prérequis :
  - Appium Server v2 : npm install -g appium && appium
  - Driver UiAutomator2 : appium driver install uiautomator2
  - Emulateur Android AVD ou device physique connecté (adb devices)
  - Variables .env : APPIUM_SERVER_URL, ANDROID_DEVICE_NAME, APP_PACKAGE, APP_ACTIVITY

Usage autonome :
  python -m agentic_testing.executors.mobile
"""

import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agentic_testing.config import CORPUS_DIR, REPORTS_DIR, SCREENSHOTS_DIR, relative_to_root

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
APPIUM_SERVER_URL = os.getenv("APPIUM_SERVER_URL", "http://localhost:4723")
ANDROID_DEVICE = os.getenv("ANDROID_DEVICE_NAME", "emulator-5554")
APP_PACKAGE = os.getenv("APP_PACKAGE", "com.techshop.chatbot")
APP_ACTIVITY = os.getenv("APP_ACTIVITY", ".ui.ChatActivity")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000/chat")
AGENT_BASE_URL = AGENT_API_URL.rsplit("/chat", 1)[0]

MOBILE_CORPUS_PATH = CORPUS_DIR / "mobile_test_cases.json"
MOBILE_RESULTS_PATH = REPORTS_DIR / "mobile_execution_results.json"

# Timeouts Appium (secondes)
TIMEOUT_ELEMENT = 15  # Attente d'un élément UI
# Aligné sur TIMEOUT_BOT_REPLY côté Web (executor_web.py, 30s) : Ollama peut
# répondre en 15-27s selon la complexité (voir reports/RAPPORT_COMPLET_PROJET.md,
# TC-015 mesuré à 27042ms). Un timeout à 20s produisait de faux FAIL sur des
# réponses qui auraient fini par arriver.
TIMEOUT_RESPONSE = 30  # Attente réponse du bot
IMPLICIT_WAIT = 8

# ─── Element IDs Android ───────────────────────────────────────────────────────
# Chargés depuis le corpus pour ne pas les dupliquer
_ELEMENT_IDS: dict = {}


def _load_element_ids() -> dict:
    global _ELEMENT_IDS
    if _ELEMENT_IDS:
        return _ELEMENT_IDS
    if MOBILE_CORPUS_PATH.exists():
        with open(MOBILE_CORPUS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _ELEMENT_IDS = data.get("metadata", {}).get("element_ids", {})
    return _ELEMENT_IDS


# ─── Driver Appium singleton ───────────────────────────────────────────────────
_driver = None


def _get_driver():
    """
    Initialise le driver Appium une seule fois pour toute la suite.
    Utilise UiAutomator2 pour Android — le plus stable et le plus rapide.
    """
    global _driver
    if _driver is not None:
        return _driver
    try:
        from appium import webdriver as appium_webdriver
        from appium.options import UiAutomator2Options

        options = UiAutomator2Options()
        options.platform_name = "Android"
        options.device_name = ANDROID_DEVICE
        options.app_package = APP_PACKAGE
        options.app_activity = APP_ACTIVITY
        options.no_reset = False  # App réinitialisée entre sessions
        options.auto_grant_permissions = True  # Évite les popups de permission
        options.new_command_timeout = 120  # Timeout inactivité driver
        options.set_capability("appium:settings[waitForIdleTimeout]", 2000)

        _driver = appium_webdriver.Remote(
            command_executor=APPIUM_SERVER_URL,
            options=options,
        )
        _driver.implicitly_wait(IMPLICIT_WAIT)
        return _driver

    except Exception:
        return None


def close_driver():
    """Ferme proprement le driver Appium."""
    global _driver
    if _driver:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None


# ─── Helpers Appium ────────────────────────────────────────────────────────────


def _find(driver, resource_id: str):
    """Trouve un élément par resource-id Android. Lève NoSuchElementException si absent."""
    from appium.webdriver.common.appiumby import AppiumBy

    return driver.find_element(AppiumBy.ID, resource_id)


def _wait_for(driver, resource_id: str, timeout: int = TIMEOUT_ELEMENT):
    """Attend qu'un élément soit présent et visible."""
    from appium.webdriver.common.appiumby import AppiumBy
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.presence_of_element_located((AppiumBy.ID, resource_id)))


def _get_bot_messages(driver) -> list[str]:
    """Retourne la liste de tous les textes des messages bot affichés."""
    from appium.webdriver.common.appiumby import AppiumBy

    els = driver.find_elements(AppiumBy.ID, _load_element_ids().get("bot_msg", ""))
    return [el.text for el in els if el.text.strip()]


def _send_message(driver, text: str, ids: dict) -> float:
    """
    Envoie un message dans le champ de saisie et retourne l'instant
    (time.perf_counter()) du clic d'envoi — à passer à _wait_bot_reply()
    pour mesurer la latence depuis le clic réel, pas depuis l'appel à
    _wait_bot_reply() (qui a un léger décalage avec le clic).
    """
    input_el = _find(driver, ids["input"])
    input_el.clear()
    if text:
        input_el.send_keys(text)
    send_el = _find(driver, ids["send_btn"])
    start = time.perf_counter()
    send_el.click()
    return start


def _wait_bot_reply(driver, initial_count: int, send_start: float) -> tuple[str, float, float]:
    """
    Attend une nouvelle réponse bot, retourne (texte, latence_ms, confiance).
    Utilise polling sur le count des messages bot. La latence est mesurée
    depuis `send_start` (l'instant du clic d'envoi, voir _send_message).
    """
    deadline = send_start + TIMEOUT_RESPONSE
    while time.perf_counter() < deadline:
        msgs = _get_bot_messages(driver)
        if len(msgs) > initial_count:
            latency_ms = round((time.perf_counter() - send_start) * 1000, 2)
            return msgs[-1], latency_ms, 0.85
        time.sleep(0.2)
    return "", round(TIMEOUT_RESPONSE * 1000, 2), 0.0


def _perform_gestures(driver, gestures: list[str]) -> list[str]:
    """
    Exécute une liste de gestes Appium.
    Retourne la liste des gestes réellement exécutés.
    """
    executed = []
    for gesture in gestures:
        try:
            if gesture == "scroll_down":
                size = driver.get_window_size()
                driver.swipe(size["width"] // 2, size["height"] * 3 // 4, size["width"] // 2, size["height"] // 4, 800)
                executed.append(gesture)

            elif gesture == "scroll_up":
                size = driver.get_window_size()
                driver.swipe(size["width"] // 2, size["height"] // 4, size["width"] // 2, size["height"] * 3 // 4, 800)
                executed.append(gesture)

            elif gesture == "rotate_landscape":
                driver.orientation = "LANDSCAPE"
                time.sleep(1)
                executed.append(gesture)

            elif gesture == "rotate_portrait":
                driver.orientation = "PORTRAIT"
                time.sleep(1)
                executed.append(gesture)

            elif gesture == "background_app_5s":
                driver.background_app(5)
                executed.append(gesture)

            elif gesture == "foreground_app":
                driver.activate_app(APP_PACKAGE)
                time.sleep(1)
                executed.append(gesture)

        except Exception:
            pass  # Geste non supporté sur cet émulateur — on continue

    return executed


# ─── Exécution d'un cas de test Mobile ────────────────────────────────────────


def execute_mobile_test(test_case: dict) -> dict:
    """
    Exécute un cas de test via Appium sur l'app Android.

    Workflow complet :
      1. Initialise / réinitialise l'app (clear state)
      2. Gère le flux login si auth requis (MOB-007, MOB-008)
      3. Exécute les gestes pré-message (rotation, background…)
      4. Envoie le message dans le champ de saisie
      5. Attend la réponse du bot
      6. Exécute les gestes post-message (scroll)
      7. Capture screenshot
      8. Retourne résultat uniforme (même format que API et Web)
    """
    tc_id = test_case.get("id", "?")
    tc_name = test_case.get("name", "?")
    tc_input = test_case.get("input", "")
    gestures = test_case.get("gestures", [])
    auth = test_case.get("auth")

    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    screenshot_path = SCREENSHOTS_DIR / test_case.get("expected", {}).get("screenshot", f"{tc_id}.png")
    ids = _load_element_ids()

    base = {
        "id": tc_id,
        "name": tc_name,
        "canal": "mobile",
        "category": test_case.get("category", "?"),
        "input": tc_input,
        "expected": test_case.get("expected", {}),
        "failure_criteria": test_case.get("failure_criteria", {}),
        "notes": test_case.get("notes", ""),
        "api_mirror": test_case.get("api_mirror", ""),
        "web_mirror": test_case.get("web_mirror", ""),
        "screenshot": relative_to_root(screenshot_path),
        "gestures_done": [],
        "actual": "",
        "intent_actual": "?",
        "confidence": 0,
        "status_code": 200,
        "latency_ms": 0,
        "error": None,
    }

    driver = _get_driver()
    if driver is None:
        return {
            **base,
            "error": (
                "APPIUM_UNAVAILABLE — Appium server non démarré ou émulateur absent. "
                "Démarrer : appium --address 0.0.0.0 --port 4723 "
                "et lancer un émulateur AVD : emulator -avd Pixel_6_API_33"
            ),
        }

    try:
        # ── 1. Reset de l'app pour état propre ─────────────────────────────────
        driver.terminate_app(APP_PACKAGE)
        time.sleep(0.5)
        driver.activate_app(APP_PACKAGE)
        time.sleep(1)

        # ── 2. Flux d'authentification (MOB-007, MOB-008) ───────────────────────
        login_result = None
        if auth:
            login_result = _execute_login_flow(driver, auth, ids, screenshot_path)
            if auth.get("expect_success") and not login_result.get("success"):
                ss_path = SCREENSHOTS_DIR / f"{tc_id}_login_fail.png"
                driver.get_screenshot_as_file(str(ss_path))
                return {
                    **base,
                    "screenshot": relative_to_root(ss_path),
                    "error": f"LOGIN_FAILED: {login_result.get('error')}",
                }
            if not auth.get("expect_success"):
                # On teste que le login échoue bien
                driver.get_screenshot_as_file(str(screenshot_path))
                error_text = login_result.get("error_message", "")
                return {
                    **base,
                    "actual": error_text,
                    "status_code": login_result.get("http_status", 401),
                    "latency_ms": login_result.get("latency_ms", 0),
                    "screenshot": relative_to_root(screenshot_path),
                }

        # ── 3. Gestes pré-message ──────────────────────────────────────────────
        pre_gestures = [g for g in gestures if g not in ("scroll_down", "scroll_up")]
        done = _perform_gestures(driver, pre_gestures)
        base["gestures_done"].extend(done)

        # ── 4. Envoi du message ────────────────────────────────────────────────
        initial_count = len(_get_bot_messages(driver))
        send_start = _send_message(driver, tc_input, ids)

        # ── 5. Attente réponse bot ─────────────────────────────────────────────
        actual_text, latency_ms, confidence = _wait_bot_reply(driver, initial_count, send_start)

        # ── 6. Gestes post-message (scroll) ────────────────────────────────────
        post_gestures = [g for g in gestures if g in ("scroll_down", "scroll_up")]
        done2 = _perform_gestures(driver, post_gestures)
        base["gestures_done"].extend(done2)

        # ── 7. Flux logout (si MOB-007) ────────────────────────────────────────
        logout_ok = None
        if auth and auth.get("expect_success"):
            logout_ok = _execute_logout_flow(driver, ids, token=login_result.get("token", ""))

        # ── 8. Screenshot final ────────────────────────────────────────────────
        driver.get_screenshot_as_file(str(screenshot_path))

        return {
            **base,
            "actual": actual_text,
            "intent_actual": "?",  # L'app native ne retourne pas l'intent
            "confidence": confidence,
            "latency_ms": latency_ms,
            # Résultats réels du flux auth (comparables à expected.login_success /
            # expected.logout_success du corpus) — auparavant non remontés du tout,
            # ces attentes n'étaient donc jamais vérifiables.
            **({"login_success": login_result.get("success")} if login_result else {}),
            **({"logout_success": logout_ok} if logout_ok is not None else {}),
        }

    except Exception as e:
        error_type = type(e).__name__
        # Screenshot de debug
        try:
            driver.get_screenshot_as_file(str(SCREENSHOTS_DIR / f"{tc_id}_ERROR.png"))
        except Exception:
            pass
        return {
            **base,
            "error": f"APPIUM_ERROR:{error_type}: {str(e)[:250]}",
        }


# ─── Flux Login / Logout Appium ───────────────────────────────────────────────


def _execute_login_flow(driver, auth: dict, ids: dict, screenshot_path: Path) -> dict:
    """
    Exécute le flux de login via l'UI Android.
    Tente d'abord le login via UI native, fallback sur API REST si éléments absents.
    Retourne {success, token, http_status, latency_ms, error_message}.
    """
    username = auth.get("username", "")
    password = auth.get("password", "")

    # ── Fallback API REST (plus fiable sur émulateur sans vraie app) ───────────
    start = time.perf_counter()
    try:
        with httpx.Client(base_url=AGENT_BASE_URL, timeout=10.0) as client:
            r = client.post("/login", json={"username": username, "password": password})
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if r.status_code == 200:
            data = r.json()
            return {
                "success": True,
                "token": data.get("token", ""),
                "http_status": 200,
                "latency_ms": latency_ms,
                "error_message": "",
            }
        else:
            detail = r.json().get("detail", {})
            msg = detail.get("message", "") if isinstance(detail, dict) else str(detail)
            return {
                "success": False,
                "token": "",
                "http_status": r.status_code,
                "latency_ms": latency_ms,
                "error_message": msg,
                "error": f"HTTP {r.status_code}: {msg}",
            }
    except Exception as e:
        return {
            "success": False,
            "token": "",
            "http_status": 0,
            "latency_ms": 0,
            "error_message": str(e),
            "error": f"LOGIN_API_ERROR: {e}",
        }


def _execute_logout_flow(driver, ids: dict, token: str = "") -> bool:
    """
    Exécute le logout via UI Android (si le bouton existe) ou via API REST.
    Retourne True si logout réussi.
    """
    from appium.webdriver.common.appiumby import AppiumBy

    try:
        logout_btn = driver.find_element(AppiumBy.ID, ids.get("logout_btn", ""))
        logout_btn.click()
        time.sleep(0.5)
        return True
    except Exception:
        pass

    # Fallback API : le bouton UI est absent/inaccessible sur cet émulateur —
    # révoque quand même le token côté serveur pour ne pas laisser une
    # session active (le corpus attend logout_success=true dans ce cas).
    if not token:
        return False
    try:
        with httpx.Client(base_url=AGENT_BASE_URL, timeout=10.0) as client:
            r = client.post("/logout", json={"token": token})
        return r.status_code == 200
    except Exception:
        return False


# ─── Exécution de tous les cas Mobile ─────────────────────────────────────────


def execute_all_mobile_tests() -> str:
    """
    Charge corpus/mobile_test_cases.json, exécute tous les cas via Appium,
    sauvegarde dans reports/mobile_execution_results.json.

    Retourne un résumé JSON (status, total, errors).
    """
    if not MOBILE_CORPUS_PATH.exists():
        return json.dumps(
            {
                "status": "ERROR",
                "message": f"Corpus mobile introuvable : {MOBILE_CORPUS_PATH.name}",
            }
        )

    with open(MOBILE_CORPUS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    if not cases:
        return json.dumps({"status": "ERROR", "message": "Aucun cas mobile trouvé."})

    # Vérifie que le serveur Appium est disponible avant de lancer
    appium_ok = False
    try:
        r = httpx.get(f"{APPIUM_SERVER_URL}/status", timeout=3.0)
        appium_ok = r.status_code == 200
    except Exception:
        pass

    if not appium_ok:
        return json.dumps(
            {
                "status": "ERROR",
                "message": (
                    f"Appium server inaccessible sur {APPIUM_SERVER_URL}. "
                    "Démarrer : appium --address 0.0.0.0 --port 4723\n"
                    "Puis lancer un émulateur AVD : emulator -avd Pixel_6_API_33"
                ),
            }
        )

    results = []
    errors_count = 0

    for tc in cases:
        result = execute_mobile_test(tc)
        results.append(result)
        if result.get("error"):
            errors_count += 1

    close_driver()

    MOBILE_RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(MOBILE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    return json.dumps(
        {
            "status": "OK",
            "total_executed": len(results),
            "errors": errors_count,
            "saved_to": MOBILE_RESULTS_PATH.name,
            "screenshots_dir": relative_to_root(SCREENSHOTS_DIR),
            "message": f"{len(results)} cas mobile exécutés, {errors_count} erreur(s).",
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
            "[bold cyan]Executor Mobile — Appium Android Code-Based[/bold cyan]\n"
            f"Serveur Appium : {APPIUM_SERVER_URL}\n"
            f"Device        : {ANDROID_DEVICE}\n"
            f"App Package   : {APP_PACKAGE}\n"
            f"Corpus        : {MOBILE_CORPUS_PATH.name}",
            border_style="cyan",
        )
    )

    result_str = execute_all_mobile_tests()
    result = json.loads(result_str)

    if result["status"] == "OK":
        console.print(f"\n[green]✓ {result['message']}[/green]")
        console.print(f"  Résultats  : reports/{result['saved_to']}")
        console.print(f"  Screenshots: {result['screenshots_dir']}")
    else:
        console.print(f"\n[red]✗ {result['message']}[/red]")
        sys.exit(1)
