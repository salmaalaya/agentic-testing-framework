"""
voice/stt.py — Speech-to-Text (STT) multibackend

Architecture à deux niveaux :
  1. Vosk  (offline, 100% local, ~50MB modèle)  ← priorité
  2. SpeechRecognition + Google (online, fallback si vosk absent)

Vosk ne nécessite aucune DLL native supplémentaire — fonctionne
nativement sur Windows sans Visual C++ Redistributable.

Téléchargement automatique du modèle vosk français au premier lancement
(~50MB, stocké dans src/agentic_testing/voice/models/vosk-fr/, ignoré par git).

Usage autonome :
    python -m voice.stt
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# ─── Configuration ─────────────────────────────────────────────────────────────
STT_BACKEND = os.getenv("STT_BACKEND", "vosk")  # "vosk" ou "google"
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "fr-FR")  # BCP-47 : "fr-FR" ou "en-US"
SAMPLE_RATE = 16000  # Hz — Vosk et Whisper attendent 16kHz
RECORD_SECONDS = int(os.getenv("STT_RECORD_SECONDS", "5"))

# Dossier local pour le modèle Vosk
MODELS_DIR = Path(__file__).parent / "models"
VOSK_MODEL_DIR = MODELS_DIR / "vosk-fr"

# URL du modèle Vosk français léger (~50MB)
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"

# ─── Téléchargement automatique du modèle Vosk ────────────────────────────────


def _download_vosk_model() -> bool:
    """
    Télécharge et extrait le modèle Vosk français si absent.
    Retourne True si le modèle est disponible après l'opération.
    """
    if VOSK_MODEL_DIR.exists() and any(VOSK_MODEL_DIR.iterdir()):
        return True

    console.print("[yellow]Modèle Vosk absent. Téléchargement (~50MB)...[/yellow]")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        import zipfile

        zip_path = MODELS_DIR / "vosk-fr.zip"

        # Téléchargement avec barre de progression
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                print(f"\r  Téléchargement... {pct}%", end="", flush=True)

        urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path, reporthook)
        print()  # Nouvelle ligne après la barre de progression

        # Extraction
        console.print("[dim]  Extraction du modèle...[/dim]")
        with zipfile.ZipFile(zip_path, "r") as z:
            # Le zip contient un dossier "vosk-model-small-fr-0.22"
            top_dir = z.namelist()[0].split("/")[0]
            z.extractall(MODELS_DIR)

        # Renommage vers le chemin attendu
        extracted = MODELS_DIR / top_dir
        if extracted.exists() and extracted != VOSK_MODEL_DIR:
            extracted.rename(VOSK_MODEL_DIR)

        zip_path.unlink(missing_ok=True)
        console.print(f"[green]✓ Modèle Vosk installé dans {VOSK_MODEL_DIR}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Téléchargement modèle Vosk échoué : {e}[/red]")
        console.print("[yellow]  Fallback sur SpeechRecognition (Google, nécessite internet)[/yellow]")
        return False


# ─── Singleton Vosk ───────────────────────────────────────────────────────────
_vosk_model = None
_vosk_recognizer = None


def _get_vosk_recognizer():
    """Charge le modèle et le recognizer Vosk une seule fois."""
    global _vosk_model, _vosk_recognizer
    if _vosk_recognizer is None:
        if not _download_vosk_model():
            return None
        try:
            import vosk

            vosk.SetLogLevel(-1)  # Supprime les logs verbeux de Vosk
            console.print("[dim]Chargement Vosk...[/dim]", end="")
            _vosk_model = vosk.Model(str(VOSK_MODEL_DIR))
            _vosk_recognizer = vosk.KaldiRecognizer(_vosk_model, SAMPLE_RATE)
            console.print(" [green]✓[/green]")
        except Exception as e:
            console.print(f" [red]✗ {e}[/red]")
            return None
    return _vosk_recognizer


# ─── Enregistrement microphone ────────────────────────────────────────────────


def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    """
    Enregistre `duration` secondes depuis le microphone.
    Retourne un tableau numpy float32 mono 16kHz.
    """
    console.print(f"[bold yellow]🎤 Parlez maintenant ({duration}s)...[/bold yellow]")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    console.print("[dim]  Enregistrement terminé.[/dim]")
    return audio.flatten()


def record_until_silence(
    max_duration: int = 30,
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
) -> np.ndarray:
    """
    Enregistre jusqu'à détecter un silence de `silence_duration` secondes.
    Plus naturel pour une conversation interactive.
    """
    console.print("[bold yellow]🎤 Parlez... (silence pour terminer)[/bold yellow]")

    chunk_size = int(SAMPLE_RATE * 0.1)  # blocs de 100ms
    max_chunks = int(max_duration * SAMPLE_RATE / chunk_size)
    silent_needed = int(silence_duration * SAMPLE_RATE / chunk_size)
    audio_chunks = []
    silent_count = 0
    speaking_started = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            flat = chunk.flatten()
            audio_chunks.append(flat)
            amplitude = np.abs(flat).mean()

            if amplitude > silence_threshold:
                speaking_started = True
                silent_count = 0
            elif speaking_started:
                silent_count += 1
                if silent_count >= silent_needed:
                    break

    console.print("[dim]  Fin de la parole détectée.[/dim]")
    return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype="float32")


# ─── Transcription ─────────────────────────────────────────────────────────────


def _transcribe_vosk(audio: np.ndarray) -> str:
    """Transcrit avec Vosk (offline)."""
    rec = _get_vosk_recognizer()
    if rec is None:
        return ""

    # Convertit float32 → int16 PCM (format attendu par Vosk)
    audio_int16 = (audio * 32767).astype(np.int16)
    raw_bytes = audio_int16.tobytes()

    rec.AcceptWaveform(raw_bytes)
    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()


def _transcribe_google(audio: np.ndarray) -> str:
    """
    Transcrit via SpeechRecognition + Google (fallback online).
    Nécessite une connexion internet.
    """
    import speech_recognition as sr

    # Sauvegarde dans un WAV temporaire pour SpeechRecognition
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=STT_LANGUAGE)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        console.print(f"[red]  Erreur Google STT : {e}[/red]")
        return ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe(audio: np.ndarray) -> str:
    """
    Transcrit un tableau numpy audio en texte.
    Essaie Vosk d'abord, puis Google en fallback.
    """
    if len(audio) == 0:
        return ""

    backend = STT_BACKEND.lower()

    if backend == "google":
        return _transcribe_google(audio)

    # Vosk en priorité
    text = _transcribe_vosk(audio)
    if text:
        return text

    # Fallback Google si Vosk ne retourne rien
    console.print("[dim]  Vosk vide → fallback Google...[/dim]")
    return _transcribe_google(audio)


def record_and_transcribe(duration: int = None, auto_stop: bool = True) -> str:
    """
    Enregistre puis transcrit en une seule fonction.
    Point d'entrée principal appelé depuis voice_interface.py.

    auto_stop=True  → s'arrête sur silence (recommandé pour conversation)
    auto_stop=False → enregistre exactement `duration` secondes
    """
    if auto_stop:
        audio = record_until_silence()
    else:
        audio = record_audio(duration or RECORD_SECONDS)

    if audio is None or len(audio) == 0:
        return ""

    console.print("[dim]  Transcription...[/dim]", end="")
    text = transcribe(audio)
    console.print(" [green]✓[/green]")
    return text


# ─── Point d'entrée autonome ──────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold cyan]Test STT — Vosk (offline) + SpeechRecognition[/bold cyan]")
    console.print(f"Backend : {STT_BACKEND} | Langue : {STT_LANGUAGE}\n")

    text = record_and_transcribe(auto_stop=True)
    if text:
        console.print(f"\n[bold green]Transcription :[/bold green] {text}")
    else:
        console.print("[yellow]Aucune parole détectée.[/yellow]")
