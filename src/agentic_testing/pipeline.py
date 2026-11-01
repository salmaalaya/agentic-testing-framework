"""
agentic_testing/pipeline.py (lancé via run_pipeline.py) — Entrypoint principal du pipeline de Testing Agentique

Orchestre les 4 étapes dans l'ordre :
  Generator → Executor → Evaluator → Reporter

Chaque étape est appelée directement en Python (pas via le tool-calling
Swarm, peu fiable avec des modèles locaux) ; le LLM (Ollama) n'est utilisé
QUE pour les tâches de contenu — un cas de test à la fois pour le Generator,
un scoring à la fois pour l'Evaluator — jamais pour orchestrer plusieurs
objets dans une seule réponse.

Canaux supportés :
  --canal api     → 23 cas manuels API via httpx (défaut)
  --canal web     → 15 cas Playwright via Chromium headless
  --canal mobile  → 15 cas Appium Android (émulateur AVD requis)
  --canal all     → API + Web + Mobile en séquence

Usage :
    python run_pipeline.py --skip-generator --no-check
    python run_pipeline.py --canal web --no-check
    python run_pipeline.py --canal mobile --no-check
    python run_pipeline.py --canal all --no-check
    python run_pipeline.py --start-from evaluator --no-check
    python run_pipeline.py --preview
    python run_pipeline.py --verbose
    python run_pipeline.py --canal all --skip-generator --no-check --dashboard --pdf
    python run_pipeline.py --skip-generator --no-check --nli   # + score NLI (~2.7s/cas en plus, CPU)

Détection de régressions entre deux versions de l'agent cible :
    python scripts/compare_versions.py --snapshot baseline   # avant modif
    python scripts/compare_versions.py --compare baseline    # après modif
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from agentic_testing.config import PROJECT_ROOT, ensure_project_root_on_path

load_dotenv()

console = Console()
ROOT = PROJECT_ROOT


# ─── Vérification des prérequis ───────────────────────────────────────────────
def check_prerequisites(skip_generator: bool = False) -> bool:
    """
    Vérifie qu'Ollama est démarré et que les modèles nécessaires sont disponibles.
    Retourne True si tout est OK, False sinon.
    """
    import httpx

    console.print(Rule("[bold cyan]Vérification des prérequis[/bold cyan]"))
    all_ok = True

    # 1. Ollama
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if r.status_code == 200:
            console.print("  [green]✓[/green] Ollama en cours d'exécution")
            available_models = [m["name"].split(":")[0] for m in r.json().get("models", [])]

            # Vérifie les modèles nécessaires
            needed = []
            if not skip_generator:
                needed.append(os.getenv("MODEL_GENERATOR", "mistral"))
            needed.append(os.getenv("MODEL_EXECUTOR", "mistral"))
            needed.append(os.getenv("MODEL_EVALUATOR", "llama3.2"))
            needed.append(os.getenv("MODEL_REPORTER", "mistral"))
            needed = list(dict.fromkeys(needed))  # Déduplique

            for model in needed:
                if model in available_models:
                    console.print(f"  [green]✓[/green] Modèle {model} disponible")
                else:
                    console.print(f"  [red]✗[/red] Modèle {model} manquant → ollama pull {model}")
                    all_ok = False
        else:
            console.print(f"  [red]✗[/red] Ollama répond avec status {r.status_code}")
            all_ok = False
    except Exception:
        console.print("  [red]✗[/red] Ollama non disponible → ollama serve")
        all_ok = False

    # 2. Mock Agent
    try:
        r = httpx.get("http://localhost:8000/health", timeout=3.0)
        if r.status_code == 200:
            console.print("  [green]✓[/green] Mock agent disponible (localhost:8000)")
        else:
            console.print(f"  [yellow]⚠[/yellow] Mock agent répond {r.status_code}")
    except Exception:
        console.print("  [red]✗[/red] Mock agent non disponible → uvicorn target_app.app:app --reload --port 8000")
        all_ok = False

    console.print()
    return all_ok


# ─── Pipeline complet ─────────────────────────────────────────────────────────
def run_full_pipeline(
    skip_generator: bool = False,
    start_from: str = "generator",
    canal: str = "api",
    verbose: bool = False,
    generator_max_attempts: int = 3,
    nli: bool = False,
) -> bool:
    """
    Lance le pipeline complet dans l'ordre :
    Generator → Executor → Evaluator → Reporter

    Paramètres :
      skip_generator         : si True, démarre à l'Executor avec les cas manuels
      start_from             : "generator" | "executor" | "evaluator" | "reporter"
      verbose                : affiche tous les échanges LLM
      generator_max_attempts : tentatives max PAR CAS DE TEST individuel si le
                                JSON reçu de mistral est invalide (voir
                                agents.generator.generate_all_cases)
      nli                    : si True, calcule le score NLI (entailment) en plus
                                du jugement LLM d'hallucination — désactivé par
                                défaut car coûte ~2.7s/cas mesuré sur CPU (voir
                                README section NLI, largement > 30% de la durée
                                totale d'un run typique). BLEU/ROUGE-L restent
                                toujours calculés (quasi-instantanés).

    Retourne True si le pipeline s'est terminé avec succès.
    """
    # Imports des agents (ici, après les définitions pour éviter les imports circulaires)
    from agentic_testing.agents.generator import generate_all_cases

    pipeline_start = time.time()

    console.print(
        Panel(
            "[bold cyan]Testing Agentique — Pipeline Swarm S3-S6[/bold cyan]\n"
            "Generator → Executor → Evaluator → Reporter\n"
            f"Mode   : {'Skip Generator' if skip_generator else 'Pipeline complet'}\n"
            f"Canal  : {canal.upper()}\n"
            f"Départ : {start_from}",
            border_style="cyan",
        )
    )
    console.print()

    results = {}

    # ── ÉTAPE 1 : Generator (direct) ──────────────────────────────────────────
    # Génération directe en Python (bypass Swarm), même principe que les
    # étapes Executor/Evaluator/Reporter ci-dessous : le LLM ne fait que
    # produire UN cas de test à la fois (voir src/agentic_testing/agents/generator.py), et c'est
    # generate_all_cases() qui boucle, valide et sauvegarde en Python — on
    # évite ainsi de dépendre du tool-calling Swarm pour transporter un gros
    # JSON de 10 cas d'un coup, point de rupture historique avec mistral.
    if start_from == "generator" and not skip_generator:
        console.print(Rule("[bold]Agent : Generator (direct)[/bold]"))
        gen_start = time.time()
        gen_str = generate_all_cases(max_attempts_per_case=generator_max_attempts, verbose=verbose)
        gen_data = json.loads(gen_str)
        gen_elapsed = round(time.time() - gen_start, 1)

        if gen_data.get("status") == "OK":
            console.print(f"  [green]✓[/green] {gen_data.get('message', gen_str[:200])}")
            if gen_data.get("total_failed"):
                console.print(
                    f"  [yellow]⚠ {gen_data['total_failed']}/{gen_data['total_requested']} cas n'ont pas pu "
                    f"être générés (numéros : {gen_data.get('failed_case_numbers')}).[/yellow]"
                )
            results["generator"] = {
                "success": True,
                "final_message": gen_data.get("message", ""),
                "elapsed_s": gen_elapsed,
            }
        else:
            console.print(f"  [red]✗ Generator : {gen_data.get('message', gen_str[:200])}[/red]")
            results["generator"] = {
                "success": False,
                "final_message": gen_data.get("message", ""),
                "elapsed_s": gen_elapsed,
            }
            return False
    else:
        console.print("[dim]Generator ignoré — utilisation des cas existants.[/dim]\n")

    # ── ÉTAPE 2 : Executor (par canal) ────────────────────────────────────────
    if start_from in ("generator", "executor") or skip_generator:
        # Détermine la source selon le canal demandé
        canal_map = {
            "api": "generated" if (start_from == "generator" and not skip_generator) else "manual",
            "web": "web",
            "mobile": "mobile",
        }
        # Pour le mode "all" on lance les 3 canaux séquentiellement
        canaux_to_run = canal if canal != "all" else ["api", "web", "mobile"]
        if isinstance(canaux_to_run, str):
            canaux_to_run = [canaux_to_run]

        for current_canal in canaux_to_run:
            source = canal_map.get(current_canal, "manual")
            canal_label = {"api": "API REST", "web": "Web (Playwright)", "mobile": "Mobile (Appium)"}
            console.print(Rule(f"[cyan]Canal : {canal_label.get(current_canal, current_canal)}[/cyan]"))

            # ── Exécution directe (bypass LLM pour garantir le bon source) ────
            # Les LLMs locaux ignorent parfois le paramètre source dans le message.
            # On appelle execute_all_tests() directement en Python, puis on notifie
            # l'Executor via Swarm uniquement pour le handoff vers l'Evaluator.
            from agentic_testing.agents.executor import execute_all_tests

            console.print(f"  [dim]Exécution directe execute_all_tests(source='{source}')...[/dim]")
            import json as _json

            exec_result_str = execute_all_tests(source=source)
            try:
                exec_result_data = _json.loads(exec_result_str)
            except _json.JSONDecodeError:
                # execute_all_tests() a retourné un message d'erreur en texte brut
                # (pas du JSON) — on l'affiche proprement au lieu de planter avec
                # une trace Python illisible.
                console.print(f"  [red]✗ Executor {current_canal} : {exec_result_str}[/red]")
                if current_canal == "api":
                    return False
                results[f"executor_{current_canal}"] = {
                    "success": False,
                    "final_message": exec_result_str,
                    "elapsed_s": 0,
                }
                continue
            elapsed_exec = 0

            if exec_result_data.get("status") != "OK":
                console.print(
                    f"  [red]✗ Executor {current_canal} : {exec_result_data.get('message', exec_result_str[:100])}[/red]"
                )
                if current_canal == "api":
                    return False
                else:
                    results[f"executor_{current_canal}"] = {
                        "success": False,
                        "final_message": exec_result_data.get("message", ""),
                        "elapsed_s": 0,
                    }
                    continue

            console.print(f"  [green]✓[/green] {exec_result_data.get('message', 'OK')}")
            results[f"executor_{current_canal}"] = {
                "success": True,
                "final_message": exec_result_data.get("message", ""),
                "elapsed_s": elapsed_exec,
            }

    # ── ÉTAPE 3 : Evaluator ───────────────────────────────────────────────────
    if start_from in ("generator", "executor", "evaluator") or skip_generator:
        import json as _json

        from agentic_testing.agents.evaluator import score_all_mobile_results, score_all_results, score_all_web_results

        console.print(Rule("[bold]Agent : Evaluator (direct)[/bold]"))
        if nli:
            console.print(
                "  [dim]Score NLI activé (--nli) : ~2.7s/cas ajoutées (mDeBERTa-v3-base-mnli-xnli, CPU).[/dim]"
            )
        eval_ok = True

        # Score API si les résultats existent
        api_results = ROOT / "reports" / "execution_results.json"
        if api_results.exists():
            scored_str = score_all_results(compute_nli=nli)
            scored_data = _json.loads(scored_str)
            eval_ok = scored_data.get("status") == "OK"
            console.print(f"  [green]✓[/green] API — {scored_data.get('message', scored_str[:80])}")
        else:
            console.print("  [dim]API — execution_results.json absent, skipped.[/dim]")
            # Si on est en mode --canal web ou mobile, ce n'est pas une erreur
            if canal == "api":
                console.print("[red]Pipeline interrompu : aucun résultat API à scorer.[/red]")
                return False

        # Score Web si les résultats existent
        web_results = ROOT / "reports" / "web_execution_results.json"
        if web_results.exists():
            ws = score_all_web_results(compute_nli=nli)
            wd = _json.loads(ws)
            console.print(f"  [green]✓[/green] Web — {wd.get('message', ws[:80])}")

        # Score Mobile si les résultats existent
        mob_results = ROOT / "reports" / "mobile_execution_results.json"
        if mob_results.exists():
            ms = score_all_mobile_results(compute_nli=nli)
            md_mob = _json.loads(ms)
            console.print(f"  [green]✓[/green] Mobile — {md_mob.get('message', ms[:80])}")

        results["evaluator"] = {
            "success": eval_ok,
            "final_message": scored_data.get("message", "") if api_results.exists() else "skipped",
            "elapsed_s": 0,
        }
        if not eval_ok:
            console.print("[red]Pipeline interrompu : Evaluator en erreur.[/red]")
            return False

    # ── ÉTAPE 4 : Reporter (direct) ───────────────────────────────────────────
    import json as _json

    from agentic_testing.agents.reporter import generate_and_save_report

    console.print(Rule("[bold]Agent : Reporter (direct)[/bold]"))
    import time as _time

    _rep_start = _time.time()
    rep_str = generate_and_save_report()
    rep_data = _json.loads(rep_str)
    rep_elapsed = round(_time.time() - _rep_start, 1)

    if rep_data.get("status") == "OK":
        console.print(f"  [green]✓[/green] Rapport généré ({rep_elapsed}s)")
        console.print(f"  [dim]Web inclus   : {rep_data.get('web_scores_included', False)}[/dim]")
        console.print(f"  [dim]Mobile inclus: {rep_data.get('mobile_scores_included', False)}[/dim]")
        results["reporter"] = {"success": True, "final_message": rep_data.get("message", ""), "elapsed_s": rep_elapsed}
    else:
        console.print(f"  [red]✗ Reporter : {rep_data.get('message', rep_str[:100])}[/red]")
        results["reporter"] = {"success": False, "final_message": rep_data.get("message", ""), "elapsed_s": rep_elapsed}

    # ── Résumé final ──────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - pipeline_start, 1)
    console.print(Rule("[bold green]Pipeline Terminé[/bold green]"))

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan")
    table.add_column("Statut", justify="center")
    table.add_column("Durée")

    for agent_name, res in results.items():
        status = "[green]✓ OK[/green]" if res["success"] else "[red]✗ ERREUR[/red]"
        table.add_row(
            agent_name.capitalize(),
            status,
            f"{res.get('elapsed_s', '?')}s",
        )
    console.print(table)
    console.print(f"\n[bold]Durée totale : {total_elapsed}s[/bold]\n")

    # Affiche le chemin des fichiers générés
    reports_dir = ROOT / "reports"
    if reports_dir.exists():
        generated = list(reports_dir.glob("*.json")) + list(reports_dir.glob("*.md"))
        if generated:
            console.print("[bold]Fichiers générés :[/bold]")
            for f in sorted(generated):
                size = f.stat().st_size
                console.print(f"  [green]✓[/green] {f.name} ({size:,} bytes)")

    console.print()
    return results.get("reporter", {}).get("success", False)


# ─── Affichage rapide du rapport ──────────────────────────────────────────────
def preview_report() -> None:
    """Affiche un aperçu du rapport généré."""
    report_path = ROOT / "reports" / "test_report.md"
    if not report_path.exists():
        console.print("[yellow]Aucun rapport trouvé dans reports/test_report.md[/yellow]")
        return

    content = report_path.read_text(encoding="utf-8")
    # Affiche les 80 premières lignes
    lines = content.split("\n")[:80]
    console.print(Panel("\n".join(lines), title="[bold]Aperçu du rapport[/bold]", border_style="green"))

    summary_path = ROOT / "reports" / "final_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        s = summary.get("summary", {})
        rate = s.get("success_rate_pct", "?")
        console.print(f"\n[bold]Taux de réussite global : {rate}%[/bold]")
        console.print(
            f"PASS: {s.get('passed', '?')}  FAIL: {s.get('failed', '?')}  "
            f"WARN: {s.get('warned', '?')}  ERROR: {s.get('errors', '?')}"
        )
        hallucs = s.get("hallucinations_detected", 0)
        if hallucs:
            console.print(f"[red]Hallucinations détectées : {hallucs}[/red]")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Point d'entrée CLI : parse les arguments puis lance run_full_pipeline()."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Testing Agentique — Swarm + Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_pipeline.py                              # API, pipeline complet
  python run_pipeline.py --skip-generator --no-check # API, cas manuels
  python run_pipeline.py --canal web --no-check      # Web (Playwright)
  python run_pipeline.py --canal mobile --no-check   # Mobile (Appium)
  python run_pipeline.py --canal all --no-check      # Les 3 canaux
  python run_pipeline.py --start-from evaluator      # Reprend depuis l'Evaluator
  python run_pipeline.py --preview                   # Aperçu du dernier rapport
  python run_pipeline.py --verbose                   # Affiche tous les échanges LLM
        """,
    )
    parser.add_argument(
        "--canal",
        choices=["api", "web", "mobile", "all"],
        default="api",
        help="Canal de test : api (défaut), web (Playwright), mobile (Appium), all (les 3)",
    )
    parser.add_argument(
        "--skip-generator",
        action="store_true",
        help="Ignorer le Generator et utiliser les cas manuels du corpus (API)",
    )
    parser.add_argument(
        "--start-from",
        choices=["generator", "executor", "evaluator", "reporter"],
        default="generator",
        help="Démarrer le pipeline depuis un agent spécifique",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher tous les échanges LLM et appels d'outils",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Afficher un aperçu du dernier rapport généré",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Ignorer la vérification des prérequis",
    )
    parser.add_argument(
        "--generator-max-attempts",
        type=int,
        default=3,
        help="Tentatives max PAR CAS DE TEST si le JSON reçu de mistral est invalide (défaut : 3)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Génère reports/dashboard.html (Plotly) à la fin du pipeline",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Génère reports/rapport_comprehensible.pdf (langage clair, non technique) à la fin du pipeline",
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help=(
            "Calcule le score NLI (entailment vs catalogue/commandes) en plus du jugement LLM "
            "d'hallucination. Désactivé par défaut : mesuré ~2.7s/cas sur CPU (modèle "
            "mDeBERTa-v3-base-mnli-xnli), soit +130-200s sur un run typique — voir README section NLI."
        ),
    )
    args = parser.parse_args()

    # Mode aperçu uniquement
    if args.preview:
        preview_report()
        return

    # Normalise start_from si skip_generator
    start_from = args.start_from
    if args.skip_generator:
        start_from = "executor"

    # Vérification des prérequis
    if not args.no_check:
        ok = check_prerequisites(skip_generator=args.skip_generator or start_from != "generator")
        if not ok:
            console.print("[red]Prérequis non satisfaits. Corrige les erreurs ci-dessus puis relance.[/red]")
            console.print("[dim]Utilise --no-check pour ignorer cette vérification.[/dim]")
            sys.exit(1)

    # Lancement du pipeline
    success = run_full_pipeline(
        skip_generator=args.skip_generator,
        start_from=start_from,
        canal=args.canal,
        verbose=args.verbose,
        generator_max_attempts=args.generator_max_attempts,
        nli=args.nli,
    )

    if success:
        console.print("[bold green]✓ Pipeline terminé avec succès.[/bold green]")
        console.print("  Rapport : [cyan]reports/test_report.md[/cyan]")
        console.print("  Synthèse : [cyan]reports/final_summary.json[/cyan]")

        if args.dashboard:
            try:
                ensure_project_root_on_path()
                from scripts.generate_dashboard import build_dashboard

                dash_path = build_dashboard()
                console.print(f"  Dashboard : [cyan]{dash_path.relative_to(ROOT)}[/cyan]")
            except Exception as e:
                console.print(f"[yellow]⚠ Dashboard non généré : {e}[/yellow]")

        if args.pdf:
            try:
                ensure_project_root_on_path()
                from scripts.generate_pdf_report import build_pdf_report

                pdf_path = build_pdf_report()
                console.print(f"  Rapport PDF (langage clair) : [cyan]{pdf_path.relative_to(ROOT)}[/cyan]")
            except Exception as e:
                console.print(f"[yellow]⚠ Rapport PDF non généré : {e}[/yellow]")
    else:
        console.print("[bold red]✗ Pipeline terminé avec des erreurs.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
