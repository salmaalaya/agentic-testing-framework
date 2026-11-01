"""
voice/voice_interface.py — Orchestration des 4 modes d'interaction

Les 4 modes disponibles :
  ┌─────────────────┬────────────┬─────────────┐
  │ Mode            │ Entrée     │ Sortie      │
  ├─────────────────┼────────────┼─────────────┤
  │ voice_to_voice  │ Microphone │ Haut-parleur│
  │ text_to_voice   │ Clavier    │ Haut-parleur│
  │ voice_to_text   │ Microphone │ Terminal    │
  │ text_to_text    │ Clavier    │ Terminal    │
  └─────────────────┴────────────┴─────────────┘

Chaque mode appelle le mock agent via HTTP POST /chat exactement
comme le fait l'Executor du pipeline Swarm.

Usage :
    from agentic_testing.voice.voice_interface import VoiceInterface, Mode
    vi = VoiceInterface()
    vi.interact(mode=Mode.VOICE_TO_VOICE)
"""

import os
from enum import Enum
from typing import Optional

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

# ─── Configuration ─────────────────────────────────────────────────────────────
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000/chat")
AGENT_BASE_URL = AGENT_API_URL.rsplit("/chat", 1)[0]


# ─── Enum des modes ────────────────────────────────────────────────────────────
class Mode(Enum):
    VOICE_TO_VOICE = "voice_to_voice"  # 🎤 → 🔊
    TEXT_TO_VOICE = "text_to_voice"  # ⌨️  → 🔊
    VOICE_TO_TEXT = "voice_to_text"  # 🎤 → 💬
    TEXT_TO_TEXT = "text_to_text"  # ⌨️  → 💬

    @property
    def uses_mic(self) -> bool:
        return self in (Mode.VOICE_TO_VOICE, Mode.VOICE_TO_TEXT)

    @property
    def uses_speaker(self) -> bool:
        return self in (Mode.VOICE_TO_VOICE, Mode.TEXT_TO_VOICE)

    @property
    def label(self) -> str:
        icons = {
            Mode.VOICE_TO_VOICE: "🎤 → 🔊  Voice to Voice",
            Mode.TEXT_TO_VOICE: "⌨️  → 🔊  Text to Voice",
            Mode.VOICE_TO_TEXT: "🎤 → 💬  Voice to Text",
            Mode.TEXT_TO_TEXT: "⌨️  → 💬  Text to Text",
        }
        return icons[self]


# ─── Classe principale ─────────────────────────────────────────────────────────
class VoiceInterface:
    """
    Orchestrateur multimodal.
    Gère l'entrée (micro ou clavier), appelle l'agent, gère la sortie
    (haut-parleur ou terminal) selon le mode choisi.
    """

    def __init__(self, session_id: str = "voice_session"):
        self.session_id = session_id
        self._http_client: Optional[httpx.Client] = None

        # Imports lazy — chargés seulement si le mode en a besoin
        self._stt = None
        self._tts = None

    # ── Client HTTP ────────────────────────────────────────────────────────────

    def _get_client(self) -> httpx.Client:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(
                base_url=AGENT_BASE_URL,
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
            )
        return self._http_client

    def close(self):
        """Libère les ressources réseau."""
        if self._http_client and not self._http_client.is_closed:
            self._http_client.close()

    # ── Import lazy STT/TTS ────────────────────────────────────────────────────

    def _get_stt(self):
        if self._stt is None:
            from agentic_testing.voice.stt import record_and_transcribe

            self._stt = record_and_transcribe
        return self._stt

    def _get_tts(self):
        if self._tts is None:
            from agentic_testing.voice.tts import speak

            self._tts = speak
        return self._tts

    # ── Entrée ─────────────────────────────────────────────────────────────────

    def get_input(self, mode: Mode) -> str:
        """
        Récupère l'entrée utilisateur selon le mode.
        Retourne une string vide si rien n'est capturé.
        """
        if mode.uses_mic:
            text = self._get_stt()(auto_stop=True)
            if text:
                console.print(f"[bold cyan]Vous (voix) :[/bold cyan] {text}")
            return text
        else:
            try:
                text = console.input("[bold cyan]Vous :[/bold cyan] ").strip()
                return text
            except (KeyboardInterrupt, EOFError):
                return ""

    # ── Appel agent ───────────────────────────────────────────────────────────

    def call_agent(self, message: str) -> dict:
        """
        Envoie `message` au mock agent via POST /chat.
        Retourne le dict complet de la réponse.
        """
        try:
            r = self._get_client().post(
                "/chat",
                json={"message": message, "session_id": self.session_id},
            )
            if r.status_code == 200:
                return r.json()
            return {"response": f"[Erreur HTTP {r.status_code}]", "intent": "error", "confidence": 0}
        except httpx.ConnectError:
            return {
                "response": "L'agent n'est pas disponible. Vérifiez qu'il est démarré.",
                "intent": "error",
                "confidence": 0,
            }
        except Exception as e:
            return {"response": f"Erreur : {e}", "intent": "error", "confidence": 0}

    # ── Sortie ─────────────────────────────────────────────────────────────────

    def deliver_output(self, mode: Mode, response: str, metadata: dict) -> None:
        """
        Délivre la réponse de l'agent selon le mode :
        - speaker  → pyttsx3 lit la réponse à voix haute
        - terminal → affiche dans la console avec Rich
        """
        intent = metadata.get("intent", "?")
        confidence = metadata.get("confidence", 0)
        latency = metadata.get("latency_ms", 0)

        # Affichage terminal dans tous les cas (même en mode voice)
        console.print(
            f"[bold green]Agent :[/bold green] {response}  [dim](intent={intent}, conf={confidence}, {latency}ms)[/dim]"
        )

        # Lecture vocale si le mode l'exige
        if mode.uses_speaker:
            self._get_tts()(response, blocking=True)

    # ── Tour de conversation ───────────────────────────────────────────────────

    def one_turn(self, mode: Mode) -> bool:
        """
        Exécute un seul tour de conversation dans le mode donné.
        Retourne False si l'utilisateur veut quitter, True sinon.
        """
        # 1. Entrée
        user_input = self.get_input(mode)

        if not user_input:
            if mode.uses_mic:
                console.print("[yellow]Aucune parole détectée. Réessayez.[/yellow]")
                return True
            else:
                return True  # ligne vide → on continue

        # 2. Commandes de contrôle
        if user_input.lower() in ("quit", "exit", "quitter", "au revoir", "stop"):
            farewell = "Au revoir ! Bonne journée."
            console.print(f"[bold green]Agent :[/bold green] {farewell}")
            if mode.uses_speaker:
                self._get_tts()(farewell)
            return False

        # 3. Appel agent
        result = self.call_agent(user_input)
        response = result.get("response", "")

        # 4. Sortie
        self.deliver_output(mode, response, result)
        return True

    # ── Boucle conversationnelle ───────────────────────────────────────────────

    def interact(self, mode: Mode, max_turns: int = 50) -> None:
        """
        Lance une conversation interactive dans le mode choisi.
        S'arrête quand l'utilisateur dit "quitter" ou après `max_turns` tours.
        """
        # Vérifie que l'agent est disponible
        try:
            r = self._get_client().get("/health", timeout=3.0)
            if r.status_code != 200:
                console.print("[red]✗ Mock agent non disponible.[/red]")
                return
        except Exception:
            console.print(
                "[red]✗ Mock agent non disponible — démarrer : uvicorn target_app.app:app --reload --port 8000[/red]"
            )
            return

        console.print(
            Panel(
                f"[bold cyan]TechShop Assistant — {mode.label}[/bold cyan]\n"
                f"Session : [dim]{self.session_id}[/dim]\n"
                f"[dim]Tapez/dites 'quitter' pour terminer[/dim]",
                border_style="cyan",
            )
        )

        # Message de bienvenue vocal si mode speaker
        welcome = "Bonjour ! Je suis l'assistant TechShop. Comment puis-je vous aider ?"
        console.print(f"[bold green]Agent :[/bold green] {welcome}")
        if mode.uses_speaker:
            self._get_tts()(welcome)

        turns = 0
        try:
            while turns < max_turns:
                console.print()  # Ligne vide entre les tours
                should_continue = self.one_turn(mode)
                turns += 1
                if not should_continue:
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrompue.[/yellow]")
        finally:
            self.close()


# ─── Fonctions utilitaires publiques ──────────────────────────────────────────


def run(mode: Mode, session_id: str = "voice_session") -> None:
    """Lance une session dans le mode donné. Point d'entrée simplifié."""
    vi = VoiceInterface(session_id=session_id)
    vi.interact(mode=mode)


def select_mode_interactive() -> Mode:
    """
    Affiche un menu interactif et retourne le mode choisi par l'utilisateur.
    """
    console.print(
        Panel(
            "[bold cyan]Sélectionnez un mode d'interaction[/bold cyan]",
            border_style="cyan",
        )
    )

    modes = list(Mode)
    for i, m in enumerate(modes, 1):
        console.print(f"  [bold]{i}[/bold]. {m.label}")

    console.print()
    while True:
        try:
            choice = console.input("Votre choix (1-4) : ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(modes):
                return modes[idx]
            console.print("[yellow]Choix invalide. Entrez un numéro entre 1 et 4.[/yellow]")
        except (ValueError, KeyboardInterrupt):
            console.print("[yellow]Choix invalide.[/yellow]")


# ─── Point d'entrée autonome ──────────────────────────────────────────────────

if __name__ == "__main__":
    mode = select_mode_interactive()
    run(mode=mode)
