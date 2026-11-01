"""
check_setup.py — Vérifie que l'environnement est correctement configuré
pour le projet de Testing Agentique.

Usage :
    python scripts/check_setup.py
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

# ─── Dépendances Python à vérifier ───────────────────────────────────────────
PYTHON_DEPS = [
    ("openai", "OpenAI client (compatible Ollama)"),
    ("swarm.core", "OpenAI Swarm framework"),
    ("playwright", "Playwright browser automation"),
    ("httpx", "HTTP async client"),
    ("fastapi", "FastAPI mock agent server"),
    ("uvicorn", "ASGI server"),
    ("pytest", "Test runner"),
    ("nltk", "BLEU score NLP"),
    ("rouge_score", "ROUGE score NLP"),
    ("plotly", "Dashboard HTML"),
    ("pydantic", "JSON schema validation"),
    ("rich", "Console colorée"),
]

# ─── Services externes à vérifier ─────────────────────────────────────────────
SERVICES = [
    {
        "name": "Ollama",
        "url": "http://localhost:11434",
        "description": "Serveur LLM local",
    },
    {
        "name": "Mock Agent",
        "url": "http://localhost:8000/health",
        "description": "Agent conversationnel cible",
    },
]

# ─── Modèles Ollama attendus ───────────────────────────────────────────────────
# phi3 ne supporte pas les tool calls — seuls mistral, llama3.2, qwen2.5 sont compatibles
EXPECTED_MODELS = ["mistral", "llama3.2"]


def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 10
    status = "[green]✓[/green]" if ok else "[red]✗[/red]"
    console.print(
        f"  {status} Python {major}.{minor} {'(OK — >=3.10 requis)' if ok else '(FAIL — Python >=3.10 requis)'}"
    )
    return ok


def check_python_deps() -> list[tuple[str, bool]]:
    results = []
    for module, description in PYTHON_DEPS:
        # swarm.core = vrai OpenAI Swarm depuis GitHub (affiché "swarm" dans le tableau)
        display_name = "swarm" if module == "swarm.core" else module
        try:
            importlib.import_module(module)
            results.append((display_name, description, True))
        except ImportError:
            results.append((display_name, description, False))
    return results


def check_service(url: str) -> bool:
    try:
        import httpx

        r = httpx.get(url, timeout=3.0)
        return r.status_code < 500
    except Exception:
        return False


def check_ollama_models() -> list[tuple[str, bool]]:
    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if r.status_code == 200:
            available = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            return [(m, m in available) for m in EXPECTED_MODELS]
    except Exception:
        pass
    return [(m, False) for m in EXPECTED_MODELS]


def check_android_mobile_prereqs() -> bool:
    """
    Vérifie les prérequis spécifiques au canal Mobile (Appium + Android SDK).

    Contrairement aux autres checks, un échec ici n'affecte PAS le statut
    global de l'environnement : le canal Mobile est optionnel (voir README,
    section "Prérequis") — api et web restent utilisables sans Android SDK.

    Retourne True si le canal Mobile peut être lancé tel quel.
    """
    mobile_ok = True

    # 1. Node.js / npm — requis pour installer le CLI Appium
    node_path = shutil.which("node")
    npm_path = shutil.which("npm")
    if node_path and npm_path:
        console.print("  [green]✓[/green] Node.js / npm disponibles")
    else:
        console.print("  [red]✗[/red] Node.js / npm introuvables")
        console.print("  [yellow]→ Installer Node.js LTS : https://nodejs.org[/yellow]")
        mobile_ok = False

    # 2. CLI Appium installée globalement
    appium_path = shutil.which("appium")
    if appium_path:
        console.print(f"  [green]✓[/green] CLI Appium installée ({appium_path})")
    else:
        console.print("  [red]✗[/red] CLI Appium non installée")
        console.print("  [yellow]→ npm install -g appium[/yellow]")
        mobile_ok = False

    # 3. Driver UiAutomator2 (seulement vérifiable si la CLI est présente)
    if appium_path:
        try:
            result = subprocess.run(
                ["appium", "driver", "list", "--installed"],
                capture_output=True,
                text=True,
                timeout=15,
                shell=(sys.platform == "win32"),
            )
            if "uiautomator2" in result.stdout.lower():
                console.print("  [green]✓[/green] Driver Appium uiautomator2 installé")
            else:
                console.print("  [red]✗[/red] Driver Appium uiautomator2 non installé")
                console.print("  [yellow]→ appium driver install uiautomator2[/yellow]")
                mobile_ok = False
        except Exception:
            console.print("  [yellow]⚠[/yellow] Impossible de vérifier les drivers Appium (timeout ou erreur CLI)")

    # 4. Serveur Appium démarré
    appium_url = os.getenv("APPIUM_SERVER_URL", "http://localhost:4723")
    if check_service(f"{appium_url}/status"):
        console.print(f"  [green]✓[/green] Serveur Appium en cours d'exécution ({appium_url})")
    else:
        console.print(f"  [red]✗[/red] Serveur Appium non accessible ({appium_url})")
        console.print("  [yellow]→ appium --address 0.0.0.0 --port 4723[/yellow]")
        mobile_ok = False

    # 5. ANDROID_HOME / ANDROID_SDK_ROOT
    android_home = os.getenv("ANDROID_HOME") or os.getenv("ANDROID_SDK_ROOT")
    if android_home and Path(android_home).is_dir():
        console.print(f"  [green]✓[/green] ANDROID_HOME défini et valide : {android_home}")
    else:
        console.print("  [red]✗[/red] ANDROID_HOME / ANDROID_SDK_ROOT non défini ou introuvable")
        console.print(
            "  [yellow]→ Installer Android Studio puis définir la variable "
            "(voir README § Configurer le canal Mobile)[/yellow]"
        )
        mobile_ok = False
        android_home = None

    # 6. adb — dans le PATH, ou déductible depuis ANDROID_HOME
    adb_path = shutil.which("adb")
    if not adb_path and android_home:
        candidate = Path(android_home) / "platform-tools" / ("adb.exe" if sys.platform == "win32" else "adb")
        if candidate.exists():
            adb_path = str(candidate)
    if adb_path:
        console.print(f"  [green]✓[/green] adb disponible ({adb_path})")
    else:
        console.print("  [red]✗[/red] adb introuvable (ni dans le PATH, ni dans ANDROID_HOME/platform-tools)")
        console.print("  [yellow]→ Ajouter <ANDROID_HOME>\\platform-tools au PATH[/yellow]")
        mobile_ok = False

    # 7. emulator + au moins un AVD créé
    emulator_path = shutil.which("emulator")
    if not emulator_path and android_home:
        candidate = Path(android_home) / "emulator" / ("emulator.exe" if sys.platform == "win32" else "emulator")
        if candidate.exists():
            emulator_path = str(candidate)

    avds: list[str] = []
    if emulator_path:
        try:
            result = subprocess.run([emulator_path, "-list-avds"], capture_output=True, text=True, timeout=15)
            avds = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception:
            pass

    if avds:
        console.print(f"  [green]✓[/green] {len(avds)} AVD(s) disponible(s) : {', '.join(avds)}")
    else:
        console.print("  [red]✗[/red] Aucun AVD (émulateur Android virtuel) trouvé")
        console.print("  [yellow]→ Créer un AVD : voir README § Configurer le canal Mobile[/yellow]")
        mobile_ok = False

    # 8. Device/émulateur actuellement connecté (informatif, ne bloque pas
    #    le statut — l'émulateur n'a pas forcément besoin d'être déjà lancé
    #    au moment du check, seulement au moment de `run_pipeline.py --canal mobile`)
    if adb_path:
        try:
            result = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=10)
            connected = [
                line.split()[0]
                for line in result.stdout.splitlines()[1:]
                if line.strip() and line.split()[-1] == "device"
            ]
            if connected:
                console.print(f"  [green]✓[/green] {len(connected)} device(s) connecté(s) : {', '.join(connected)}")
            else:
                hint = f" (ex: emulator -avd {avds[0]})" if avds else ""
                console.print(f"  [dim]⚠ Aucun device connecté pour l'instant{hint}[/dim]")
        except Exception:
            pass

    return mobile_ok


def main():
    console.print("\n[bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   Agentic Testing — Vérification de l'environnement ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]\n")

    all_ok = True

    # 1. Python version
    console.print("[bold]1. Version Python[/bold]")
    py_ok = check_python_version()
    all_ok = all_ok and py_ok

    # 2. Dépendances Python
    console.print("\n[bold]2. Dépendances Python[/bold]")
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("Module", style="cyan")
    table.add_column("Description")
    table.add_column("Statut", justify="center")

    dep_results = check_python_deps()
    for module, description, ok in dep_results:
        status = "[green]✓ OK[/green]" if ok else "[red]✗ MANQUANT[/red]"
        table.add_row(module, description, status)
        if not ok:
            all_ok = False

    console.print(table)

    missing = [m for m, _, ok in dep_results if not ok]
    if missing:
        console.print("[yellow]→ Installer les dépendances manquantes :[/yellow]")
        # Swarm doit être installé depuis GitHub, pas PyPI
        install_cmds = []
        for m in missing:
            if m == "swarm":
                install_cmds.append("git+https://github.com/openai/swarm.git")
            else:
                install_cmds.append(m)
        if "git+https://github.com/openai/swarm.git" in install_cmds:
            console.print("  [bold]pip install git+https://github.com/openai/swarm.git[/bold]")
            install_cmds.remove("git+https://github.com/openai/swarm.git")
        if install_cmds:
            console.print(f"  [bold]pip install {' '.join(install_cmds)}[/bold]")
        console.print("  [dim](Note : ne pas utiliser 'openai-swarm' depuis PyPI — c'est un faux package)[/dim]\n")

    # 3. Services externes
    console.print("[bold]3. Services externes[/bold]")
    for svc in SERVICES:
        ok = check_service(svc["url"])
        status = "[green]✓ RUNNING[/green]" if ok else "[red]✗ OFFLINE[/red]"
        console.print(f"  {status} {svc['name']:15} ({svc['description']}) — {svc['url']}")
        if not ok and svc["name"] == "Ollama":
            console.print("  [yellow]→ Démarrer Ollama : ollama serve[/yellow]")
        if not ok and svc["name"] == "Mock Agent":
            console.print("  [yellow]→ Démarrer le mock : uvicorn target_app.app:app --reload[/yellow]")

    # 4. Modèles Ollama
    console.print("\n[bold]4. Modèles Ollama disponibles[/bold]")
    model_results = check_ollama_models()
    for model, ok in model_results:
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        pull_hint = "" if ok else f"  → [yellow]ollama pull {model}[/yellow]"
        console.print(f"  {status} {model:15}{pull_hint}")
        if not ok:
            all_ok = False

    # 5. Playwright browsers
    console.print("\n[bold]5. Playwright browsers[/bold]")
    try:
        subprocess.run(
            ["python", "-m", "playwright", "install", "--dry-run"], capture_output=True, text=True, timeout=10
        )
        # Si la commande ne plante pas, on considère que playwright est installé
        console.print("  [green]✓[/green] Playwright installé")
        console.print("  [yellow]→ Si navigateurs non installés : python -m playwright install chromium[/yellow]")
    except Exception:
        console.print("  [red]✗[/red] Playwright non disponible")
        console.print("  [yellow]→ pip install playwright && python -m playwright install chromium[/yellow]")

    # 6. Dépendances audio (Voice Interface)
    console.print("\n[bold]6. Dépendances audio (Voice Interface)[/bold]")
    audio_deps = [
        ("vosk", "vosk", "STT Speech-to-Text offline (modèle local)"),
        ("speech_recognition", "SpeechRecognition", "STT fallback Google"),
        ("sounddevice", "sounddevice", "Capture microphone"),
        ("soundfile", "soundfile", "Lecture/écriture fichiers audio"),
        ("pyttsx3", "pyttsx3", "TTS Text-to-Speech (voix système)"),
        ("numpy", "numpy", "Traitement tableaux audio"),
    ]
    audio_missing = []
    for module, pkg, description in audio_deps:
        try:
            importlib.import_module(module)
            console.print(f"  [green]✓[/green] {pkg:20} {description}")
        except ImportError:
            console.print(f"  [red]✗[/red] {pkg:20} {description}")
            audio_missing.append(pkg)
            all_ok = False

    if audio_missing:
        console.print(f"\n  [yellow]→ pip install {' '.join(audio_missing)}[/yellow]")

    # 7. Canal Mobile — Android SDK / Appium (optionnel, n'affecte pas all_ok)
    console.print("\n[bold]7. Canal Mobile — Android SDK / Appium (optionnel)[/bold]")
    mobile_ok = check_android_mobile_prereqs()
    if mobile_ok:
        console.print("  [bold green]✓ Canal Mobile prêt — vous pouvez lancer --canal mobile.[/bold green]")
    else:
        console.print(
            "  [dim]Canal Mobile non prêt (api/web restent utilisables normalement).\n"
            "  Guide complet : README § Configurer le canal Mobile.[/dim]"
        )

    # 8. Résumé final
    console.print("\n" + "─" * 55)
    if all_ok:
        console.print("[bold green]✓ Environnement prêt ! Vous pouvez lancer les tests.[/bold green]")
    else:
        console.print("[bold yellow]⚠ Environnement incomplet. Suivez les instructions ci-dessus.[/bold yellow]")
    if not mobile_ok:
        console.print("[dim]  (Canal Mobile optionnel non configuré — voir section 7 ci-dessus)[/dim]")
    console.print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
