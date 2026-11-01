"""
voice/chat_voice.py — Chatbot interactif multimodal complet

Point d'entrée principal du module voice.
Lance le chatbot dans le mode choisi via argument ou menu interactif.

Usage :
    # Menu interactif (choix du mode)
    python -m voice.chat_voice

    # Mode direct par argument
    python -m voice.chat_voice --mode voice_to_voice
    python -m voice.chat_voice --mode text_to_voice
    python -m voice.chat_voice --mode voice_to_text
    python -m voice.chat_voice --mode text_to_text

    # Avec session ID personnalisé
    python -m voice.chat_voice --mode voice_to_voice --session demo_session

Options supplémentaires :
    --list-voices     Affiche les voix TTS disponibles sur le système
    --test-stt        Teste le microphone et la transcription Whisper
    --test-tts        Teste la synthèse vocale
"""

import argparse
import sys

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_testing.voice.voice_interface import Mode, run, select_mode_interactive

load_dotenv()
console = Console()


# ─── Commandes de test ─────────────────────────────────────────────────────────


def test_stt() -> None:
    """Teste le micro + Whisper en dehors d'une session complète."""
    from agentic_testing.voice.stt import record_and_transcribe

    console.print(
        Panel(
            "[bold cyan]Test STT — faster-whisper[/bold cyan]\nParlez après le signal. Le texte transcrit s'affichera.",
            border_style="cyan",
        )
    )

    text = record_and_transcribe(auto_stop=True)
    if text:
        console.print(f"\n[bold green]Transcription :[/bold green] {text}")
        console.print("[green]✓ STT fonctionne correctement.[/green]")
    else:
        console.print("[yellow]Aucune parole détectée. Vérifiez votre microphone.[/yellow]")


def test_tts() -> None:
    """Teste la synthèse vocale en dehors d'une session complète."""
    from agentic_testing.voice.tts import list_voices, speak

    voices = list_voices()
    console.print(
        Panel(
            f"[bold cyan]Test TTS — pyttsx3[/bold cyan]\n{len(voices)} voix disponibles sur ce système.",
            border_style="cyan",
        )
    )

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Nom de la voix")
    table.add_column("ID", style="dim")

    for i, v in enumerate(voices, 1):
        table.add_row(str(i), v["name"] or "?", (v["id"] or "?")[:60])

    console.print(table)
    console.print("\n[dim]Lecture du texte de test...[/dim]")
    speak("Bonjour, je suis l'assistant e-commerce de TechShop. La synthèse vocale fonctionne correctement.")
    console.print("[green]✓ TTS fonctionne correctement.[/green]")


def list_voices_cmd() -> None:
    """Affiche les voix disponibles et quitte."""
    from agentic_testing.voice.tts import list_voices

    voices = list_voices()
    console.print(f"\n[bold]{len(voices)} voix TTS disponibles :[/bold]\n")
    for v in voices:
        console.print(f"  • [cyan]{v['name']}[/cyan]")
        console.print(f"    ID : [dim]{v['id']}[/dim]")


# ─── Vérification des prérequis audio ─────────────────────────────────────────


def check_audio_prerequisites() -> bool:
    """
    Vérifie que les dépendances audio sont disponibles.
    Retourne True si tout est OK.
    """
    all_ok = True
    checks = [
        ("vosk", "vosk (STT offline)"),
        ("speech_recognition", "SpeechRecognition (STT fallback)"),
        ("sounddevice", "sounddevice (microphone)"),
        ("soundfile", "soundfile (audio I/O)"),
        ("pyttsx3", "pyttsx3 (TTS)"),
        ("numpy", "numpy (traitement audio)"),
    ]

    for module, label in checks:
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {label}")
        except ImportError:
            pkg = module.replace("_", "-")
            console.print(f"  [red]✗[/red] {label} → pip install {pkg}")
            all_ok = False

    return all_ok


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="TechShop Assistant — Interface multimodale voice/text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles :
  voice_to_voice  🎤→🔊  Parlez, l'agent vous répond à voix haute
  text_to_voice   ⌨️→🔊  Tapez, l'agent vous répond à voix haute
  voice_to_text   🎤→💬  Parlez, la réponse s'affiche dans le terminal
  text_to_text    ⌨️→💬  Tapez, la réponse s'affiche dans le terminal

Exemples :
  python -m voice.chat_voice
  python -m voice.chat_voice --mode voice_to_voice
  python -m voice.chat_voice --mode text_to_text --session test_1
  python -m voice.chat_voice --test-stt
  python -m voice.chat_voice --test-tts
  python -m voice.chat_voice --list-voices
        """,
    )

    parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=None,
        help="Mode d'interaction (défaut : menu interactif)",
    )
    parser.add_argument(
        "--session",
        default="voice_session",
        help="Identifiant de session (défaut : voice_session)",
    )
    parser.add_argument(
        "--test-stt",
        action="store_true",
        help="Tester le microphone + Whisper STT et quitter",
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Tester la synthèse vocale et quitter",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Lister les voix TTS disponibles et quitter",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Ignorer la vérification des dépendances audio",
    )

    args = parser.parse_args()

    # ── Commandes rapides ──────────────────────────────────────────────────────
    if args.list_voices:
        list_voices_cmd()
        sys.exit(0)

    if args.test_tts:
        test_tts()
        sys.exit(0)

    if args.test_stt:
        test_stt()
        sys.exit(0)

    # ── Vérification des prérequis ─────────────────────────────────────────────
    if not args.no_check:
        console.print("\n[bold]Vérification des dépendances audio...[/bold]")
        ok = check_audio_prerequisites()
        if not ok:
            console.print(
                "\n[red]Dépendances manquantes. Installe-les avec :[/red]\n"
                "  [bold]pip install vosk SpeechRecognition sounddevice soundfile pyttsx3[/bold]"
            )
            sys.exit(1)
        console.print()

    # ── Sélection du mode ──────────────────────────────────────────────────────
    if args.mode:
        mode = Mode(args.mode)
    else:
        mode = select_mode_interactive()

    # ── Lancement ─────────────────────────────────────────────────────────────
    run(mode=mode, session_id=args.session)


if __name__ == "__main__":
    main()
