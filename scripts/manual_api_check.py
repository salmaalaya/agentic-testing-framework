"""
scripts/manual_api_check.py — Test manuel de l'agent cible (S1-S2)

Charge les cas de test depuis corpus/test_cases.json,
les envoie un par un à l'agent cible via API REST et affiche les résultats.

Usage :
    # Démarrer d'abord le mock agent :
    #   uvicorn target_app.app:app --reload --port 8000

    python scripts/manual_api_check.py
    python scripts/manual_api_check.py --id TC-001          # Un seul test
    python scripts/manual_api_check.py --canal api          # Filtrer par canal
    python scripts/manual_api_check.py --category nominal   # Filtrer par catégorie
    python scripts/manual_api_check.py --verbose            # Détail de chaque test
"""

import argparse
import json
import time
from pathlib import Path

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

AGENT_BASE_URL = "http://localhost:8000"
AGENT_CHAT_URL = f"{AGENT_BASE_URL}/chat"
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "test_cases.json"


# ─── Chargement du corpus ──────────────────────────────────────────────────────


def load_test_cases(tc_id: str = None, canal: str = None, category: str = None) -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["test_cases"]
    if tc_id:
        cases = [c for c in cases if c["id"] == tc_id]
    if canal:
        cases = [c for c in cases if c["canal"] == canal]
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


# ─── Exécution d'un cas de test ────────────────────────────────────────────────


def run_test(tc: dict, client: httpx.Client) -> dict:
    """
    Envoie un cas de test à l'agent et retourne le résultat brut.

    Utilise un client HTTP persistant (connection pooling) pour éviter
    l'overhead de reconnexion TCP à chaque requête (~2s sur Windows).
    La latence mesurée est donc la latence NETTE de traitement de l'agent.
    """
    sid = f"manual_{tc['id']}"

    # Prépare le payload — inclut le contexte si le cas de test en a un
    payload: dict = {"message": tc["input"], "session_id": sid}
    if tc.get("context"):
        payload["context"] = tc["context"]

    # Pour TC-015 (session expirée) : on supprime la session avant d'envoyer
    if tc.get("session_state") == "expired":
        try:
            client.delete(f"{AGENT_BASE_URL}/session/{sid}")
        except Exception:
            pass

    start = time.perf_counter()  # perf_counter : plus précis que time.time()
    try:
        r = client.post(AGENT_CHAT_URL, json=payload, timeout=30.0)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        body = r.json() if r.status_code == 200 else {}
        return {
            "id": tc["id"],
            "name": tc["name"],
            "canal": tc["canal"],
            "category": tc["category"],
            "input": tc["input"],
            "expected": tc["expected"],
            "failure_criteria": tc["failure_criteria"],
            "notes": tc.get("notes", ""),
            "actual": body.get("response", ""),
            "intent_actual": body.get("intent", "?"),
            "confidence": body.get("confidence", 0),
            "status_code": r.status_code,
            "latency_ms": latency_ms,
            "error": None,
        }

    except httpx.ConnectError:
        return _error_result(tc, "CONNECTION_ERROR — agent non démarré ?")
    except httpx.TimeoutException:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return _error_result(tc, f"TIMEOUT après {latency_ms}ms")
    except Exception as e:
        return _error_result(tc, str(e))


def _error_result(tc: dict, error_msg: str) -> dict:
    """Construit un résultat d'erreur uniforme."""
    return {
        "id": tc["id"],
        "name": tc["name"],
        "canal": tc["canal"],
        "category": tc["category"],
        "input": tc["input"],
        "expected": tc["expected"],
        "failure_criteria": tc["failure_criteria"],
        "notes": tc.get("notes", ""),
        "actual": "",
        "intent_actual": "?",
        "confidence": 0,
        "status_code": 0,
        "latency_ms": 0,
        "error": error_msg,
    }


# ─── Évaluation manuelle ───────────────────────────────────────────────────────


def evaluate(result: dict) -> tuple[str, str]:
    """
    Évaluation déterministe par règles.
    Le vrai scoring LLM (nuancé) sera fait par l'Evaluator Agent en S3-S4.
    Retourne (verdict, reason) où verdict = PASS | FAIL | WARN | ERROR.
    """
    fc = result["failure_criteria"]
    expected = result["expected"]
    actual = result["actual"].lower()
    status = result["status_code"]
    latency = result["latency_ms"]

    # 1. Erreur de connexion ou exception
    if result["error"]:
        return "ERROR", result["error"]

    # 2. Status code HTTP bloquant
    fail_if = fc.get("fail_if_status_gte", 400)
    if status >= fail_if:
        return "FAIL", f"Status HTTP {status} >= {fail_if}"

    # 3. Latence dépassée
    max_latency = expected.get("max_latency_ms", 10000)
    if latency > max_latency:
        return "FAIL", f"Latence {latency}ms > max {max_latency}ms"

    # 4. Réponse vide sur input non-vide
    if fc.get("fail_if_empty_response") and not actual.strip():
        return "FAIL", "Réponse vide"

    # 5. Mots obligatoires (AND) — tous doivent être présents
    for kw in expected.get("contains", []):
        if kw.lower() not in actual:
            return "FAIL", f"Réponse ne contient pas '{kw}'"

    # 6. Mots obligatoires (AND strict) — alias de contains
    for kw in expected.get("contains_all", []):
        if kw.lower() not in actual:
            return "FAIL", f"Réponse ne contient pas '{kw}' (contains_all)"

    # 7. Mots alternatifs (OR) — au moins un doit être présent
    any_kws = expected.get("contains_any", [])
    if any_kws and not any(kw.lower() in actual for kw in any_kws):
        return "FAIL", f"Aucun mot de {any_kws} trouvé dans la réponse"

    # 8. Mots interdits — sécurité (XSS, hallucination)
    for kw in expected.get("not_contains", []):
        if kw.lower() in actual:
            return "FAIL", f"Réponse contient '{kw}' (interdit)"

    # 9. Intent attendu — WARN (non bloquant, la réponse est correcte)
    if "intent" in expected and result["intent_actual"] != expected["intent"]:
        return "WARN", f"Intent attendu: {expected['intent']}, obtenu: {result['intent_actual']}"

    return "PASS", "Tous les critères satisfaits"


# ─── Affichage ─────────────────────────────────────────────────────────────────


def print_summary(results: list[dict], verdicts: list[tuple]) -> None:
    table = Table(
        title="Résultats — Tests manuels S1-S2",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Nom", width=30)
    table.add_column("Canal", width=8)
    table.add_column("Catégorie", width=12)
    table.add_column("Latence", justify="right", width=10)
    table.add_column("HTTP", justify="center", width=6)
    table.add_column("Verdict", justify="center", width=9)
    table.add_column("Raison", width=35)

    pass_count = fail_count = warn_count = error_count = 0

    for r, (verdict, reason) in zip(results, verdicts, strict=False):
        http_str = f"[dim]{r['status_code']}[/dim]" if r["status_code"] < 400 else f"[red]{r['status_code']}[/red]"
        max_lat = r["expected"].get("max_latency_ms", 10000)
        lat_str = (
            "[red]N/A[/red]"
            if r["latency_ms"] == 0
            else f"[red]{r['latency_ms']}ms[/red]"
            if r["latency_ms"] > max_lat
            else f"[green]{r['latency_ms']}ms[/green]"
        )

        if verdict == "PASS":
            v_str = "[green]✓ PASS[/green]"
            pass_count += 1
        elif verdict == "WARN":
            v_str = "[yellow]⚠ WARN[/yellow]"
            warn_count += 1
        elif verdict == "ERROR":
            v_str = "[red]✗ ERROR[/red]"
            error_count += 1
        else:
            v_str = "[red]✗ FAIL[/red]"
            fail_count += 1

        table.add_row(
            r["id"],
            r["name"],
            r["canal"],
            r["category"],
            lat_str,
            http_str,
            v_str,
            reason[:33] + ".." if len(reason) > 35 else reason,
        )

    console.print(table)

    total = len(results)
    rate = round(pass_count / total * 100, 1) if total > 0 else 0
    color = "green" if rate >= 80 else "yellow" if rate >= 50 else "red"
    console.print(f"\n[bold {color}]Taux de réussite : {rate}% ({pass_count}/{total})[/bold {color}]")
    console.print(
        f"  PASS: [green]{pass_count}[/green]  "
        f"FAIL: [red]{fail_count}[/red]  "
        f"WARN: [yellow]{warn_count}[/yellow]  "
        f"ERROR: [red]{error_count}[/red]\n"
    )

    failures = [(r, reason) for r, (v, reason) in zip(results, verdicts, strict=False) if v in ("FAIL", "ERROR")]
    if failures:
        console.print("[bold red]Top échecs critiques :[/bold red]")
        for r, reason in failures[:3]:
            console.print(f"  • [red]{r['id']}[/red] — {r['name']} : {reason}")
        console.print()


def print_detail(result: dict, verdict: str, reason: str) -> None:
    color = {"PASS": "green", "FAIL": "red", "WARN": "yellow", "ERROR": "red"}.get(verdict, "white")
    console.print(
        Panel(
            f"[bold]INPUT[/bold]    : {result['input'][:120]}\n"
            f"[bold]ACTUAL[/bold]   : {result['actual'][:250]}\n"
            f"[bold]INTENT[/bold]   : {result['intent_actual']} (confidence: {result['confidence']})\n"
            f"[bold]LATENCE[/bold]  : {result['latency_ms']}ms  "
            f"(max: {result['expected'].get('max_latency_ms', 10000)}ms)\n"
            f"[bold]VERDICT[/bold]  : [{color}]{verdict}[/{color}] — {reason}",
            title=f"[bold]{result['id']}[/bold] — {result['name']}",
            border_style=color,
        )
    )


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Test manuel de l'agent cible")
    parser.add_argument("--id", help="Exécuter un seul test par ID (ex: TC-001)")
    parser.add_argument("--canal", choices=["api", "web", "mobile"], help="Filtrer par canal")
    parser.add_argument("--category", choices=["nominal", "edge_case", "adversarial"], help="Filtrer par catégorie")
    parser.add_argument("--verbose", "-v", action="store_true", help="Afficher le détail de chaque test")
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]Testing Agentique — Tests manuels S1-S2[/bold cyan]\n"
            "Agent cible : Mock Conversational Agent\n"
            f"URL : {AGENT_CHAT_URL}",
            border_style="cyan",
        )
    )

    # ── Warmup : plusieurs requêtes pour absorber le cold start uvicorn ─────────
    # Sur Windows, la première connexion TCP + la première session uvicorn
    # coûtent ~2s chacune. On préchauffe avec 3 sessions différentes pour que
    # n'importe quel test — quel que soit le filtre --category ou --id —
    # ne subisse pas le cold start dans son chrono.
    try:
        with httpx.Client(base_url=AGENT_BASE_URL, timeout=15.0) as warmup:
            warmup.get("/health")
            for sid in ("_warmup_a_", "_warmup_b_", "_warmup_c_"):
                warmup.post("/chat", json={"message": "warmup", "session_id": sid})
            console.print("[green]✓ Agent cible disponible[/green]")
            console.print("[dim]  (warmup TCP + sessions effectué — latences nettes)[/dim]\n")
    except Exception:
        console.print("[red]✗ Agent cible non disponible — démarrer avec :[/red]")
        console.print("  [bold]uvicorn target_app.app:app --reload --port 8000[/bold]\n")
        return

    # Chargement des cas de test
    cases = load_test_cases(tc_id=args.id, canal=args.canal, category=args.category)
    if not cases:
        console.print("[red]Aucun cas de test trouvé avec ces filtres.[/red]")
        return

    console.print(f"[bold]{len(cases)} cas de test chargés.[/bold] Exécution...\n")

    results = []
    verdicts = []

    # ── Client persistant : une seule connexion TCP réutilisée pour tous les tests
    with httpx.Client(base_url=AGENT_BASE_URL, timeout=30.0) as client:
        for tc in cases:
            console.print(f"  → [cyan]{tc['id']}[/cyan] {tc['name']}...", end="")
            result = run_test(tc, client)
            verdict, reason = evaluate(result)
            results.append(result)
            verdicts.append((verdict, reason))

            color = {"PASS": "green", "FAIL": "red", "WARN": "yellow", "ERROR": "red"}.get(verdict, "white")
            console.print(f" [{color}]{verdict}[/{color}]")

            if args.verbose:
                print_detail(result, verdict, reason)

    # Résumé final
    console.print()
    print_summary(results, verdicts)

    # Sauvegarde JSON des résultats bruts (réutilisé par l'Evaluator Agent en S3-S4)
    output_path = Path(__file__).parent.parent / "reports" / "manual_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"result": r, "verdict": v, "reason": reason} for r, (v, reason) in zip(results, verdicts, strict=False)],
            f,
            ensure_ascii=False,
            indent=2,
        )
    console.print(f"[dim]Résultats sauvegardés : {output_path}[/dim]")


if __name__ == "__main__":
    main()
