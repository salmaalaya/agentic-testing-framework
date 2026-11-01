"""
voice/tts.py — Text-to-Speech (TTS) avec pyttsx3

Convertit du texte en parole via les voix système (Windows SAPI5 / Linux espeak).
100% local, aucune connexion réseau, aucune clé API.

Usage autonome :
    python -m voice.tts
"""

import os
import threading

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# ─── Configuration ─────────────────────────────────────────────────────────────
TTS_RATE = int(os.getenv("TTS_RATE", "175"))  # Vitesse de parole (mots/min). 150-200 = naturel
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))  # Volume 0.0 → 1.0
TTS_VOICE = os.getenv("TTS_VOICE", "french")  # Nom partiel de la voix à chercher

# ─── Moteur TTS (singleton thread-safe) ───────────────────────────────────────
_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """
    Initialise le moteur pyttsx3 une seule fois.
    Configure la voix française si disponible, sinon prend la voix par défaut.
    """
    global _engine
    if _engine is None:
        import pyttsx3

        _engine = pyttsx3.init()
        _engine.setProperty("rate", TTS_RATE)
        _engine.setProperty("volume", TTS_VOLUME)

        # Cherche une voix correspondant à TTS_VOICE (insensible à la casse)
        voices = _engine.getProperty("voices")
        for voice in voices:
            voice_name = (voice.name or "").lower()
            voice_id = (voice.id or "").lower()
            if TTS_VOICE.lower() in voice_name or TTS_VOICE.lower() in voice_id:
                _engine.setProperty("voice", voice.id)
                console.print(f"[dim]Voix TTS sélectionnée : {voice.name}[/dim]")
                break
        else:
            # Aucune voix française trouvée — utilise la première disponible
            if voices:
                _engine.setProperty("voice", voices[0].id)
                console.print(f"[dim]Voix TTS par défaut : {voices[0].name}[/dim]")

    return _engine


# ─── Fonctions publiques ───────────────────────────────────────────────────────


def speak(text: str, blocking: bool = True) -> None:
    """
    Synthétise `text` en parole et le joue via les haut-parleurs.

    blocking=True  : attend la fin de la synthèse avant de rendre la main
    blocking=False : lance la synthèse en arrière-plan (non-bloquant)
    """
    if not text or not text.strip():
        return

    # Nettoie le texte des balises markdown/emojis qui sonnent mal à l'oral
    clean = _clean_for_tts(text)

    if blocking:
        with _engine_lock:
            engine = _get_engine()
            engine.say(clean)
            engine.runAndWait()
    else:
        t = threading.Thread(target=_speak_async, args=(clean,), daemon=True)
        t.start()


def _speak_async(text: str) -> None:
    """Parle en arrière-plan (thread daemon)."""
    with _engine_lock:
        engine = _get_engine()
        engine.say(text)
        engine.runAndWait()


def _clean_for_tts(text: str) -> str:
    """
    Nettoie le texte pour une meilleure prononciation :
    - Retire les balises markdown (**, *, #, `)
    - Retire les emojis courants
    - Remplace les URLs par "lien"
    - Limite la longueur (évite les lectures trop longues)
    """
    import re

    # Retire les balises markdown
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **gras** → gras
    text = re.sub(r"\*(.+?)\*", r"\1", text)  # *italique* → italique
    text = re.sub(r"`(.+?)`", r"\1", text)  # `code` → code
    text = re.sub(r"#+\s", "", text)  # ## Titre → Titre
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # [texte](url) → texte

    # Retire les URLs
    text = re.sub(r"https?://\S+", "lien", text)

    # Retire les emojis (plage Unicode emoji principale)
    text = re.sub(
        r"[\U0001F300-\U0001FFFF\U00002600-\U000027BF]",
        "",
        text,
    )

    # Normalise les espaces
    text = re.sub(r"\s+", " ", text).strip()

    # Limite à 500 caractères pour les réponses longues
    if len(text) > 500:
        text = text[:497] + "..."

    return text


def list_voices() -> list[dict]:
    """Retourne la liste des voix disponibles sur le système."""
    engine = _get_engine()
    voices = engine.getProperty("voices")
    return [{"id": v.id, "name": v.name, "languages": getattr(v, "languages", [])} for v in voices]


def set_rate(rate: int) -> None:
    """Change la vitesse de lecture en cours de session."""
    engine = _get_engine()
    engine.setProperty("rate", rate)


def set_volume(volume: float) -> None:
    """Change le volume (0.0 → 1.0) en cours de session."""
    engine = _get_engine()
    engine.setProperty("volume", max(0.0, min(1.0, volume)))


# ─── Point d'entrée autonome ──────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold cyan]Test TTS — pyttsx3[/bold cyan]")

    voices = list_voices()
    console.print(f"\n{len(voices)} voix disponibles :")
    for v in voices:
        console.print(f"  • {v['name']} — {v['id']}")

    console.print("\n[dim]Test de lecture...[/dim]")
    speak("Bonjour ! Je suis l'assistant e-commerce de TechShop. Comment puis-je vous aider aujourd'hui ?")
    console.print("[green]✓ TTS fonctionne correctement.[/green]")
