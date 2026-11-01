"""
target_app/app.py — Agent conversationnel intelligent (cible des tests)

Moteur de compréhension HYBRIDE :
  1. Ollama LLM (mistral) — comprend toutes les langues du monde,
     les fautes d'orthographe, le contexte, les formulations implicites.
  2. Keyword matching — fallback instantané si Ollama est indisponible.

Exemples de messages compris :
  - "combien coute le laptop?"           → product_query (FR)
  - "how much is the laptop?"            → product_query (EN)
  - "كم سعر الحاسوب؟"                   → product_query (AR)
  - "mon ordi est kasé je veu rembourse" → complaint (FR avec fautes)
  - "suivi colis 1234"                   → order_tracking
  - "ji voudré savoir le prix"           → product_query (fautes phonétiques)

Canaux exposés :
  - API REST : POST /chat           (Canal api — S1-S4)
  - Web UI   : GET  /               (Canal web — S5-S6, Playwright)
"""

import base64
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

try:
    # Lancement documenté : uvicorn target_app.app:app (cwd = racine du projet)
    from target_app.knowledge_base import ORDERS, PRODUCTS
except ImportError:
    # Lancement depuis le dossier target_app/ (cd target_app && uvicorn app:app) :
    # "target_app" n'est alors pas importable comme package depuis lui-même
    from knowledge_base import ORDERS, PRODUCTS

load_dotenv()

# ─── Configuration Ollama ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("MODEL_EXECUTOR", "mistral")

# ─── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TechShop Intelligent Agent",
    description="Agent e-commerce multilingue piloté par Ollama + fallback keyword",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))

# Dossier des rapports générés par le pipeline (dashboard.html, PDF, etc.)
# On le sert directement en statique pour que le site "Studio" puisse y renvoyer.
_REPORTS_DIR = _BASE.parent / "reports"
_REPORTS_DIR.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=_REPORTS_DIR, check_dir=False), name="reports")

# Captures d'écran réelles (Playwright/Appium) — utilisées par le dashboard dynamique
_SCREENSHOTS_DIR = _BASE.parent / "screenshots"
_SCREENSHOTS_DIR.mkdir(exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=_SCREENSHOTS_DIR, check_dir=False), name="screenshots")


# ─── Modèles ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    context: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    latency_ms: float
    intent: str
    confidence: float
    engine: str  # "llm" ou "keyword" — pour les métriques de test


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    token: str
    username: str
    name: str
    message: str


class LogoutRequest(BaseModel):
    token: str


class SaveScreenshotRequest(BaseModel):
    session_id: str
    image_base64: str  # PNG en base64 (avec ou sans préfixe data:image/png;base64,)
    label: str = "Conversation capturée"


class RunPipelineRequest(BaseModel):
    canal: str = "api"  # api | web | mobile | all
    skip_generator: bool = True
    dashboard: bool = True
    pdf: bool = False


# ─── Base de connaissances ─────────────────────────────────────────────────────
# PRODUCTS et ORDERS vivent dans target_app/knowledge_base.py (importé plus
# haut) — partagés avec src/agentic_testing/agents/evaluator.py pour le score NLI, sans que
# l'Evaluator ait besoin d'importer ce module FastAPI complet.

# ─── Sessions de conversation ──────────────────────────────────────────────────
# Chaque page Web (chat.html, studio.html) génère un session_id ALÉATOIRE à
# chaque chargement (genSessionId() / Math.random()) et n'appelle jamais
# DELETE /session/{id} — sans nettoyage, `sessions` grossit indéfiniment au
# fil des runs Playwright/Studio répétés (constaté après ~2 jours d'uptime
# du serveur dev). Éviction par TTL d'inactivité + plafond dur en garde-fou.
sessions: dict[str, list[dict]] = {}
_session_last_seen: dict[str, float] = {}

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(30 * 60)))  # 30 min d'inactivité
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))  # garde-fou absolu


def _touch_session(session_id: str) -> None:
    _session_last_seen[session_id] = time.time()


def _prune_sessions() -> None:
    """
    Supprime les sessions inactives depuis plus de SESSION_TTL_SECONDS, puis
    si le nombre de sessions dépasse encore MAX_SESSIONS, évince les plus
    anciennes (par dernière activité) jusqu'à repasser sous la limite.
    Appelé à chaque requête /chat — coût négligeable (dict de quelques
    centaines d'entrées max grâce à cette fonction elle-même).
    """
    now = time.time()
    expired = [sid for sid, last in _session_last_seen.items() if now - last > SESSION_TTL_SECONDS]
    for sid in expired:
        sessions.pop(sid, None)
        _session_last_seen.pop(sid, None)

    overflow = len(sessions) - MAX_SESSIONS
    if overflow > 0:
        oldest = sorted(_session_last_seen.items(), key=lambda kv: kv[1])[:overflow]
        for sid, _ in oldest:
            sessions.pop(sid, None)
            _session_last_seen.pop(sid, None)


# ─── Gestion des utilisateurs (auth pour tests Mobile) ────────────────────────
# Utilisateurs en mémoire : format {username: {password, name, token}}
USERS: dict[str, dict] = {
    "testuser": {"password": "test123", "name": "Alice Martin", "token": None},
    "demouser": {"password": "demo456", "name": "Bob Dupont", "token": None},
    "qatester": {"password": "qa789", "name": "QA Engineer", "token": None},
}

# Sessions authentifiées : {token: username}
auth_tokens: dict[str, str] = {}

import secrets


def _generate_token() -> str:
    """Génère un token d'authentification sécurisé."""
    return secrets.token_hex(24)


def _get_username_from_token(token: str) -> str | None:
    """Retourne le username associé à un token, ou None si invalide."""
    return auth_tokens.get(token)


# ─── Prompt système Ollama ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es l'assistant e-commerce intelligent de TechShop, une boutique tunisienne.

CAPACITÉS :
- Tu comprends TOUTES les langues du monde (français, anglais, arabe, espagnol, etc.)
- Tu comprends les fautes d'orthographe et de frappe
- Tu déduis le contexte et l'intention même si le message est ambigu
- Tu réponds TOUJOURS dans la même langue que le client
- Tous les prix sont exprimés en Dinars Tunisiens (DT)

CATALOGUE TECHSHOP :
1. LaptopPro X1 — 1299.99 DT — Stock : 15 unités
   Intel Core i7, 16GB RAM, 512GB SSD, 15 pouces
2. PhoneUltra 5G — 899.99 DT — Stock : 42 unités
   6.7 pouces, 128GB, caméra 108MP
3. HeadsetPro ANC — 249.99 DT — RUPTURE DE STOCK (0 unité)
   Casque Bluetooth, réduction de bruit active
4. TabletAir 10 — 899.00 DT — Stock : 22 unités
   Tablette 10.9 pouces, 128GB, écran Full HD, stylet inclus
5. EarBudsX TWS — 89.90 DT — Stock : 60 unités
   Écouteurs sans fil Bluetooth 5.3, boîtier de charge, 25h d'autonomie
6. GamerBook RTX — 3819.00 DT — Stock : 6 unités
   PC portable gamer 16 pouces, RTX 5060, Intel Core i7, 16GB RAM, 1TB SSD

COMMANDES EN BASE :
- #1234 : En livraison (LaptopPro X1), estimée le 22 juillet 2026
- #5678 : Livré (PhoneUltra 5G), livré le 13 juillet 2026
- #9999 : En préparation (HeadsetPro ANC), estimée le 24 juillet 2026

RÈGLES ABSOLUES :
- Ne JAMAIS inventer un prix, un stock ou un statut de commande
- Tous les prix sont en DT (Dinars Tunisiens), jamais en euros ou dollars
- Le HeadsetPro ANC est EN RUPTURE DE STOCK — ne jamais dire qu'il est disponible
- Si le client demande un produit hors catalogue, propose le produit le plus proche du catalogue TechShop plutôt que de refuser sèchement
- Sois flexible sur la FORME (fautes, langues, formulations indirectes, comparaisons, demandes de conseil) tant que la question reste liée à TechShop, ses produits ou ses commandes
- Pour les questions totalement hors e-commerce (météo, actualité, recettes, politique...), refuse poliment d'y répondre et recentre vers TechShop, SANS répondre à la question elle-même
- Détecter l'intent parmi : greeting, product_query, order_tracking, complaint, farewell, out_of_domain

RÉPONSE JSON OBLIGATOIRE (rien d'autre, pas de texte autour) :
{
  "intent": "<intent>",
  "confidence": <float 0.0-1.0>,
  "response": "<réponse dans la langue du client>"
}"""


# ─── Moteur LLM (Ollama) ───────────────────────────────────────────────────────
def _build_ollama_messages(message: str, history: list[dict]) -> list[dict]:
    """Construit la liste de messages pour Ollama avec l'historique de session."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inclut les 6 derniers échanges pour le contexte (évite les prompts trop longs)
    for entry in history[-6:]:
        msgs.append({"role": entry["role"], "content": entry["content"]})

    msgs.append({"role": "user", "content": message})
    return msgs


def _extract_json(raw: str) -> dict | None:
    """
    Extrait le JSON de la réponse Ollama même si le LLM ajoute du texte autour.
    Stratégies dans l'ordre :
      1. Parse direct si c'est du JSON pur
      2. Extrait le bloc ```json ... ```
      3. Cherche le premier { ... } avec regex
    """
    import re

    raw = raw.strip()

    # Stratégie 1 : JSON pur
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Stratégie 2 : bloc ```json
    if "```" in raw:
        lines = raw.split("\n")
        start = next((i + 1 for i, line in enumerate(lines) if line.strip().startswith("```")), None)
        end = next((i for i, line in enumerate(lines) if line.strip() == "```" and i > (start or 0)), None)
        if start is not None and end is not None:
            try:
                return json.loads("\n".join(lines[start:end]))
            except json.JSONDecodeError:
                pass

    # Stratégie 3 : regex {…}
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_ollama(message: str, history: list[dict]) -> dict | None:
    """
    Appelle Ollama pour comprendre le message et générer une réponse.
    Retourne un dict {intent, confidence, response} ou None si Ollama est indisponible.
    """
    try:
        messages = _build_ollama_messages(message, history)

        with httpx.Client(timeout=httpx.Timeout(connect=2.0, read=25.0, write=5.0, pool=5.0)) as client:
            r = client.post(
                f"{OLLAMA_BASE_URL.rstrip('/v1')}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Faible = réponses cohérentes et factuelles
                        "num_predict": 180,  # Réduit de 300 → 180 : réponses de chatbot courtes,
                        # génère nettement plus vite sans perte de qualité perçue
                    },
                },
            )

        if r.status_code != 200:
            return None

        raw = r.json().get("message", {}).get("content", "")
        data = _extract_json(raw)

        if not data:
            return None

        # Valide et normalise les champs
        intent = str(data.get("intent", "unknown")).strip()
        confidence = float(data.get("confidence", 0.85))
        response = str(data.get("response", "")).strip()

        if not response:
            return None

        return {"intent": intent, "confidence": round(confidence, 2), "response": response}

    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        return None  # Fallback keyword automatique


# ─── Moteur Keyword (fallback) ─────────────────────────────────────────────────
# Défini au niveau module (et non plus local à _keyword_intent) pour être
# réutilisable par _is_short_and_simple() (heuristique fast-path, cf. plus bas).
_INTENT_KEYWORDS = [
    (
        "complaint",
        [
            "défectueux",
            "cassé",
            "remboursement",
            "retour",
            "sav",
            "problème",
            "problem",
            "broken",
            "refund",
            "return",
            "defectuoso",
            "roto",
            "reembolso",
            "معطل",
            "استرداد",
            "고장",
        ],
    ),
    (
        "order_tracking",
        [
            "commande",
            "livraison",
            "suivi",
            "tracking",
            "colis",
            "order",
            "delivery",
            "pedido",
            "seguimiento",
            "طلب",
            "توصيل",
            "주문",
        ],
    ),
    (
        "greeting",
        [
            "bonjour",
            "salut",
            "hello",
            "hi",
            "hey",
            "hola",
            "ciao",
            "مرحبا",
            "안녕",
            "你好",
            "привет",
            "مرحباً",
        ],
    ),
    (
        "farewell",
        [
            "au revoir",
            "bye",
            "goodbye",
            "adios",
            "ciao",
            "merci",
            "thank",
            "شكرا",
            "감사",
            "谢谢",
            "спасибо",
        ],
    ),
    (
        "out_of_domain",
        [
            "météo",
            "weather",
            "recette",
            "recipe",
            "film",
            "movie",
            "sport",
            "politique",
            "politics",
            "news",
            "actualité",
        ],
    ),
    (
        "product_query",
        [
            "prix",
            "price",
            "coût",
            "cost",
            "combien",
            "how much",
            "disponible",
            "available",
            "stock",
            "produit",
            "product",
            "acheter",
            "buy",
            "laptop",
            "phone",
            "casque",
            "headset",
            "tablette",
            "tablet",
            "écouteurs",
            "earbuds",
            "gamer",
            "gaming",
            "سعر",
            "متوفر",
            "가격",
            "价格",
            "ordinateur",
            "computer",
        ],
    ),
]


def _keyword_intent(message: str) -> tuple[str, float]:
    """
    Détection d'intent par mots-clés multilingues.
    Ordre de priorité : complaint > order_tracking > greeting > farewell
    > out_of_domain > product_query > unknown
    """
    msg = message.lower().strip()
    if not msg:
        return "empty_input", 1.0

    if msg in ("merci", "merci !", "merci.", "thanks", "thank you"):
        return "farewell", 0.95

    # Détection par MOT ENTIER (\b...\b), pas par sous-chaîne.
    # Bug corrigé : l'ancienne version faisait "sav" in msg, qui matchait
    # à l'intérieur de "savoir", "savais", etc. → complaint détecté à tort
    # sur "je veux savoir les prix". Idem pour "hi" dans "chinois", etc.
    for intent, kws in _INTENT_KEYWORDS:
        for kw in kws:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", msg, flags=re.UNICODE):
                return intent, round(random.uniform(0.72, 0.88), 2)

    return "unknown", round(random.uniform(0.30, 0.55), 2)


def _keyword_response(message: str, intent: str, confidence: float) -> tuple[str, str, float]:
    """Génère une réponse basée sur l'intent détecté par keyword."""
    msg = message.lower().strip()

    if intent == "empty_input":
        return "Je n'ai pas reçu de message. Pouvez-vous reformuler ?", intent, confidence

    if intent == "greeting":
        return (
            "Bonjour ! Je suis l'assistant TechShop. "
            "Je peux vous aider avec nos produits (LaptopPro X1, PhoneUltra 5G, HeadsetPro ANC) "
            "ou le suivi de votre commande.",
            intent,
            confidence,
        )

    if intent == "product_query":
        for pid, product in PRODUCTS.items():
            all_kws = [pid, product["name"].lower(), product["category"]] + product.get("keywords", [])
            if any(kw in msg for kw in all_kws):
                stock_msg = f"{product['stock']} unités disponibles" if product["stock"] > 0 else "rupture de stock"
                return (
                    f"Le {product['name']} est disponible à {product['price']} DT. "
                    f"{product['description']}. Stock : {stock_msg}.",
                    intent,
                    confidence,
                )
        catalogue = ", ".join(f"{p['name']} ({p['price']} DT)" for p in PRODUCTS.values())
        return (
            f"Nous proposons : {catalogue}. Sur quel produit souhaitez-vous plus d'informations ?",
            intent,
            confidence,
        )

    if intent == "order_tracking":
        for order_id, order in ORDERS.items():
            if order_id in msg:
                base = f"Commande #{order_id} ({order['product']}) : {order['status']}. Passée le {order['date']}."
                if "delivered" in order:
                    return base + f" Livrée le {order['delivered']}.", intent, confidence
                return base + f" Livraison estimée : {order['estimated']}.", intent, confidence
        return ("Pour suivre votre commande, merci de me communiquer votre numéro (ex: #1234).", intent, confidence)

    if intent == "complaint":
        return (
            "Je suis désolé pour ce problème. Notre service client est disponible "
            "du lundi au vendredi de 9h à 18h. "
            "Vous pouvez initier un retour depuis votre espace client sous 30 jours.",
            intent,
            confidence,
        )

    if intent == "farewell":
        return "Merci de votre visite ! N'hésitez pas à revenir. Bonne journée !", intent, confidence

    if intent == "out_of_domain":
        return (
            "Je suis spécialisé dans l'assistance e-commerce TechShop. "
            "Je ne peux pas répondre à cette question. "
            "Puis-je vous aider avec nos produits ou le suivi d'une commande ?",
            intent,
            confidence,
        )

    return (
        "Je n'ai pas bien compris votre demande. "
        "Je peux vous aider avec nos produits, les tarifs ou le suivi de commande.",
        "unknown",
        confidence,
    )


# ─── Moteur hybride principal ─────────────────────────────────────────────────
# Stratégie "fast-path" : la majorité des questions client (salutation, prix
# d'un produit connu, suivi d'une commande avec numéro clair) sont détectées
# avec une très haute confiance par le moteur mots-clés — dans ce cas on
# répond INSTANTANÉMENT sans appeler Ollama, qui prend 8 à 15 secondes.
# Ollama n'est appelé que pour les messages ambigus, multilingues avec fautes,
# ou qui ne correspondent à aucun motif connu : c'est là qu'un LLM apporte
# une vraie valeur, pas sur "bonjour".
FAST_PATH_MIN_CONFIDENCE = 0.90  # conservé pour référence, non utilisé directement

# Mots-clés product_query qui ne désignent AUCUN produit précis — trop
# génériques pour garantir que la réponse mots-clés (catalogue) est adaptée
# au ton/contexte réel du message (ex: sarcasme, plainte teintée d'ironie).
_GENERIC_PRODUCT_WORDS = {
    "prix",
    "price",
    "coût",
    "cost",
    "combien",
    "how much",
    "disponible",
    "available",
    "stock",
    "produit",
    "product",
    "acheter",
    "buy",
    "سعر",
    "متوفر",
    "가격",
    "价格",
}

# Mots-clés product_query SPÉCIFIQUES à un produit du catalogue (nom du
# produit, catégorie, alias multilingues) — présents dans le message, ils
# garantissent que la question porte bien sur un produit précis.
_SPECIFIC_PRODUCT_TERMS: set[str] = set()
for _pid, _product in PRODUCTS.items():
    _SPECIFIC_PRODUCT_TERMS.add(_pid)
    _SPECIFIC_PRODUCT_TERMS.add(_product["name"].lower())
    _SPECIFIC_PRODUCT_TERMS.update(kw.lower() for kw in _product.get("keywords", []))
_SPECIFIC_PRODUCT_TERMS -= _GENERIC_PRODUCT_WORDS

_EXCESSIVE_PUNCTUATION = ("!!", "??", "?!", "!?")
_MAX_SIMPLE_WORD_COUNT = 8
# Mots signalant l'ironie/la frustration : on réutilise la liste "complaint"
# déjà définie dans _INTENT_KEYWORDS plutôt que d'en dupliquer une nouvelle.
_COMPLAINT_HINT_WORDS = dict(_INTENT_KEYWORDS)["complaint"]


def _has_specific_product_reference(message: str) -> bool:
    """Vrai si le message cite un produit précis (nom, id ou alias) du catalogue."""
    msg = message.lower()
    return any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", msg, flags=re.UNICODE) for term in _SPECIFIC_PRODUCT_TERMS)


def _is_short_and_simple(message: str) -> bool:
    """
    Heuristique pour les questions produit GÉNÉRIQUES (pas de produit précis
    cité) : on ne fast-path que si le message est court, sans ponctuation
    excessive, et sans marqueur d'ironie/frustration — sinon on laisse
    Ollama gérer la nuance (ex: sarcasme contenant juste le mot "produit").
    """
    msg = message.strip()
    if len(msg.split()) > _MAX_SIMPLE_WORD_COUNT:
        return False
    if any(p in msg for p in _EXCESSIVE_PUNCTUATION):
        return False
    msg_lower = msg.lower()
    for kw in _COMPLAINT_HINT_WORDS:
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", msg_lower, flags=re.UNICODE):
            return False
    return True


def _is_fast_path_eligible(message: str, intent: str, confidence: float) -> bool:
    """Détermine si le message peut être répondu instantanément sans Ollama."""
    if intent in ("greeting", "farewell", "empty_input"):
        return True
    if intent == "order_tracking":
        return True
    if intent == "product_query":
        # Fast-path uniquement si (a) un produit précis est cité — la
        # réponse mots-clés est alors forcément la bonne — OU (b) le
        # message est court/simple, sans signe d'ironie/frustration.
        # Sinon (ex: sarcasme contenant juste "produit") on laisse Ollama
        # gérer la nuance.
        if _has_specific_product_reference(message):
            return True
        return _is_short_and_simple(message)
    return False


def smart_respond(message: str, session_id: str) -> tuple[str, str, float, str]:
    """
    Moteur hybride optimisé pour la latence :
      1. Message vide → réponse immédiate
      2. Cas clairs et fréquents (salutation, prix d'un produit du catalogue,
         suivi de commande avec numéro) → réponse mots-clés INSTANTANÉE
      3. Sinon → Ollama (compréhension multilingue, fautes, ambiguïté)
      4. Si Ollama échoue/timeout → repli mots-clés

    Retourne (response_text, intent, confidence, engine)
    engine = "llm" si Ollama a répondu, "keyword" sinon.
    """
    history = sessions.get(session_id, [])

    # ── Cas spécial : message vide ─────────────────────────────────────────────
    if not message or not message.strip():
        return ("Je n'ai pas reçu de message. Pouvez-vous reformuler ?", "empty_input", 1.0, "keyword")

    # ── Fast-path : réponse instantanée pour les cas clairs (pas d'appel LLM) ──
    fast_intent, fast_confidence = _keyword_intent(message)
    if _is_fast_path_eligible(message, fast_intent, fast_confidence):
        response, intent, confidence = _keyword_response(message, fast_intent, fast_confidence)
        return response, intent, confidence, "keyword"

    # ── Tentative Ollama (messages ambigus, multilingues, avec fautes...) ──────
    llm_result = call_ollama(message, history)

    if llm_result:
        return (llm_result["response"], llm_result["intent"], llm_result["confidence"], "llm")

    # ── Fallback keyword ───────────────────────────────────────────────────────
    intent, confidence = _keyword_intent(message)
    response, intent, confidence = _keyword_response(message, intent, confidence)
    return response, intent, confidence, "keyword"


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check."""
    # Vérifie si Ollama est disponible
    ollama_ok = False
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/v1')}/api/tags", timeout=2.0)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "agent": "techshop-intelligent-agent",
        "version": "2.0.0",
        "engine": "llm+keyword" if ollama_ok else "keyword-only",
        "ollama": "available" if ollama_ok else "offline",
        "sessions_in_memory": len(sessions),  # visibilité sur la croissance mémoire (cf. _prune_sessions)
    }


@app.get("/", response_class=HTMLResponse)
def chat_ui(request: Request):
    """Interface Web du chatbot — testée par Playwright."""
    return templates.TemplateResponse(request, "chat.html")


@app.get("/studio", response_class=HTMLResponse)
def studio_ui(request: Request):
    """
    Site 'Studio' : parler en direct au chatbot TechShop + visualiser les 4
    agents de test Swarm et leurs résultats réels, avec accès direct au
    tableau de bord et au rapport PDF.
    """
    return templates.TemplateResponse(request, "studio.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_ui(request: Request):
    """
    Dashboard DYNAMIQUE : même design que /studio, mais entièrement piloté
    par /api/pipeline-details en JS — se rafraîchit tout seul, contrairement
    au fichier reports/dashboard.html qui est un instantané figé au moment
    de sa génération.
    """
    return templates.TemplateResponse(request, "dashboard_live.html")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/api/pipeline-stats")
def pipeline_stats():
    """
    Agrège en direct les résultats produits par le pipeline (reports/*.json)
    pour alimenter le site Studio : statistiques par agent, par canal, et
    disponibilité des rapports (dashboard, PDF).
    """
    score_files = {
        "api": _REPORTS_DIR / "scored_results.json",
        "web": _REPORTS_DIR / "web_scored_results.json",
        "mobile": _REPORTS_DIR / "mobile_scored_results.json",
    }
    exec_files = {
        "api": _REPORTS_DIR / "execution_results.json",
        "web": _REPORTS_DIR / "web_execution_results.json",
        "mobile": _REPORTS_DIR / "mobile_execution_results.json",
    }

    canaux = {name: _read_json(path) for name, path in score_files.items()}
    canaux = {name: data for name, data in canaux.items() if data}

    all_results = []
    for name, data in canaux.items():
        for r in data.get("scored_results", []):
            all_results.append({**r, "canal_group": name})

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("verdict") == "PASS")
    hallucinations = sum(1 for r in all_results if r.get("hallucination") is True)

    canal_stats = {}
    for name in score_files:
        data = canaux.get(name)
        results = data.get("scored_results", []) if data else []
        c_passed = sum(1 for r in results if r.get("verdict") == "PASS")
        canal_stats[name] = {"total": len(results), "passed": c_passed}

    executor_stats = {}
    for name, path in exec_files.items():
        data = _read_json(path)
        executor_stats[name] = len(data.get("results", [])) if data else 0

    generated = _read_json(_REPORTS_DIR.parent / "corpus" / "generated_test_cases.json")
    generated_count = len(generated.get("test_cases", [])) if generated else 0

    dims = ["pertinence", "exactitude", "coherence"]
    evaluator_avgs = {}
    for d in dims:
        vals = [r.get(d, 0) for r in all_results]
        evaluator_avgs[d] = round(sum(vals) / len(vals), 2) if vals else 0

    report_md = _REPORTS_DIR / "test_report.md"
    report_updated_at = (
        datetime.fromtimestamp(report_md.stat().st_mtime).strftime("%d/%m/%Y %H:%M") if report_md.exists() else None
    )

    return {
        "has_data": total > 0,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "hallucinations": hallucinations,
        "success_rate": round(passed / total * 100, 1) if total else 0,
        "canaux": canal_stats,
        "generator": {"generated_count": generated_count, "active": generated_count > 0},
        "executor": executor_stats,
        "evaluator": evaluator_avgs,
        "reporter": {
            "updated_at": report_updated_at,
            "has_dashboard": (_REPORTS_DIR / "dashboard.html").exists(),
            "has_pdf": (_REPORTS_DIR / "rapport_comprehensible.pdf").exists(),
        },
    }


@app.get("/api/pipeline-details")
def pipeline_details():
    """
    Version détaillée de /api/pipeline-stats pour le dashboard dynamique :
    inclut la liste complète des résultats scorés (pour le tableau et le
    graphique de latence) et la liste des vraies captures d'écran
    disponibles (servies via /screenshots/<fichier>.png).
    """
    import re as _re

    score_files = {
        "API": _REPORTS_DIR / "scored_results.json",
        "Web": _REPORTS_DIR / "web_scored_results.json",
        "Mobile": _REPORTS_DIR / "mobile_scored_results.json",
    }
    canaux = {name: _read_json(path) for name, path in score_files.items()}
    canaux = {name: data for name, data in canaux.items() if data}

    all_results = []
    for canal_name, data in canaux.items():
        for r in data.get("scored_results", []):
            r = dict(r)
            r["canal_group"] = canal_name
            all_results.append(r)
    by_id = {r["id"]: r for r in all_results}

    gallery = []
    if _SCREENSHOTS_DIR.exists():
        for img_path in sorted(_SCREENSHOTS_DIR.glob("*.png")):
            m = _re.match(r"^([A-Za-z]+-\d+)", img_path.stem)
            test_id = m.group(1) if m else img_path.stem
            is_error_shot = "_ERROR" in img_path.stem.upper()
            is_chat_capture = test_id.upper().startswith("CHAT")
            meta = by_id.get(test_id, {})
            if is_chat_capture:
                canal_default, name_default, verdict_default = "studio", "Conversation en direct (Studio)", None
            elif test_id.upper().startswith("WEB"):
                canal_default, name_default, verdict_default = (
                    "web",
                    img_path.stem,
                    ("ERROR" if is_error_shot else None),
                )
            else:
                canal_default, name_default, verdict_default = (
                    "mobile",
                    img_path.stem,
                    ("ERROR" if is_error_shot else None),
                )
            gallery.append(
                {
                    "id": test_id,
                    "url": f"/screenshots/{img_path.name}",
                    "name": meta.get("name", name_default),
                    "canal": meta.get("canal", canal_default),
                    "verdict": meta.get("verdict") or verdict_default,
                }
            )

    return {
        "results": all_results,
        "gallery": gallery,
        "canal_names": list(canaux.keys()),
    }


@app.post("/api/save-screenshot")
def save_screenshot(req: SaveScreenshotRequest):
    """
    Sauvegarde une capture d'écran prise CÔTÉ NAVIGATEUR (html2canvas) de la
    conversation en cours sur /studio. La capture atterrit dans le même
    dossier screenshots/ que les captures Playwright — elle apparaît donc
    automatiquement dans la galerie du dashboard, avec l'historique complet
    de la discussion visible dedans (pas juste le dernier message).
    """
    try:
        raw = req.image_base64
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        image_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image base64 invalide : {e}") from e

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Capture trop volumineuse (max 8MB).")

    timestamp = int(time.time())
    short_sid = req.session_id.replace("studio_", "")[:8]
    filename = f"CHAT-{timestamp}-{short_sid}.png"
    out_path = _SCREENSHOTS_DIR / filename
    out_path.write_bytes(image_bytes)

    return {"status": "OK", "filename": filename, "url": f"/screenshots/{filename}"}


# ─── Pilotage du pipeline depuis le site (pas seulement le terminal) ──────────
_pipeline_state = {
    "running": False,
    "output": [],
    "returncode": None,
    "started_at": None,
    "finished_at": None,
    "args": None,
}
_pipeline_lock = threading.Lock()


def _stream_pipeline_output(proc: subprocess.Popen) -> None:
    """Lit la sortie du subprocess ligne par ligne et l'accumule pour le polling web."""
    try:
        for line in proc.stdout:
            with _pipeline_lock:
                _pipeline_state["output"].append(line.rstrip("\n"))
                # Garde un buffer raisonnable pour ne pas grossir indéfiniment
                if len(_pipeline_state["output"]) > 2000:
                    _pipeline_state["output"] = _pipeline_state["output"][-2000:]
    finally:
        proc.wait()
        with _pipeline_lock:
            _pipeline_state["running"] = False
            _pipeline_state["returncode"] = proc.returncode
            _pipeline_state["finished_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@app.post("/api/run-pipeline")
def run_pipeline_from_web(req: RunPipelineRequest):
    """
    Lance run_pipeline.py en arrière-plan (vrai subprocess Python, pas une
    simulation) pour pouvoir tester les 4 agents directement depuis le site,
    sans terminal. La progression se suit via GET /api/run-pipeline/status.
    """
    with _pipeline_lock:
        if _pipeline_state["running"]:
            return {"status": "already_running", "message": "Un pipeline est déjà en cours d'exécution."}

        if req.canal not in ("api", "web", "mobile", "all"):
            raise HTTPException(status_code=400, detail="canal doit être api, web, mobile ou all")

        project_root = _BASE.parent
        cmd = [sys.executable, "run_pipeline.py", "--canal", req.canal, "--no-check"]
        if req.skip_generator:
            cmd.append("--skip-generator")
        if req.dashboard:
            cmd.append("--dashboard")
        if req.pdf:
            cmd.append("--pdf")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Impossible de lancer le pipeline : {e}") from e

        _pipeline_state["running"] = True
        _pipeline_state["output"] = []
        _pipeline_state["returncode"] = None
        _pipeline_state["started_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        _pipeline_state["finished_at"] = None
        _pipeline_state["args"] = " ".join(cmd[1:])

        threading.Thread(target=_stream_pipeline_output, args=(proc,), daemon=True).start()

    return {"status": "started", "args": _pipeline_state["args"], "started_at": _pipeline_state["started_at"]}


@app.get("/api/run-pipeline/status")
def run_pipeline_status():
    """Statut courant du pipeline lancé depuis le site (pour le polling JS)."""
    with _pipeline_lock:
        return {
            "running": _pipeline_state["running"],
            "returncode": _pipeline_state["returncode"],
            "started_at": _pipeline_state["started_at"],
            "finished_at": _pipeline_state["finished_at"],
            "args": _pipeline_state["args"],
            "output": "\n".join(_pipeline_state["output"][-300:]),
            "success": _pipeline_state["returncode"] == 0 if _pipeline_state["returncode"] is not None else None,
        }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint principal — comprend toutes les langues et les fautes d'orthographe.
    Utilise Ollama (LLM) si disponible, sinon keyword matching multilingue.
    """
    start = time.time()

    if request.message is None:
        raise HTTPException(status_code=422, detail="Le champ 'message' est requis.")

    _prune_sessions()

    # Historique de session
    if request.session_id not in sessions:
        sessions[request.session_id] = []
    sessions[request.session_id].append({"role": "user", "content": request.message})
    _touch_session(request.session_id)

    # Réponse intelligente
    response_text, intent, confidence, engine = smart_respond(request.message, request.session_id)

    sessions[request.session_id].append({"role": "assistant", "content": response_text})

    latency_ms = round((time.time() - start) * 1000, 2)

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        latency_ms=latency_ms,
        intent=intent,
        confidence=confidence,
        engine=engine,
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Retourne l'historique de conversation d'une session (vide si inconnue)."""
    if session_id not in sessions:
        return {"session_id": session_id, "history": []}
    return {"session_id": session_id, "history": sessions[session_id]}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Supprime une session et son horodatage d'activité."""
    sessions.pop(session_id, None)
    _session_last_seen.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/products")
def list_products():
    """Catalogue complet (fictif) de TechShop."""
    return {"products": PRODUCTS}


# ─── Auth Endpoints (testés par Appium — flux login→chat→logout) ─────────────


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """
    Authentifie un utilisateur et retourne un token de session.

    Utilisateurs de test :
      - testuser / test123
      - demouser / demo456
      - qatester / qa789

    Testé par : MOB-007 (login valide), MOB-008 (login invalide)
    """
    user = USERS.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(
            status_code=401,
            detail={
                "success": False,
                "message": "Identifiants incorrects. Vérifiez votre nom d'utilisateur et mot de passe.",
            },
        )

    # Révoque l'ancien token si déjà connecté
    old_token = user.get("token")
    if old_token and old_token in auth_tokens:
        del auth_tokens[old_token]

    # Génère un nouveau token
    token = _generate_token()
    user["token"] = token
    auth_tokens[token] = request.username

    return LoginResponse(
        success=True,
        token=token,
        username=request.username,
        name=user["name"],
        message=f"Bienvenue, {user['name']} ! Connexion réussie.",
    )


@app.post("/logout")
def logout(request: LogoutRequest):
    """
    Déconnecte l'utilisateur en révoquant son token.
    Testé par : MOB-009 (logout + vérification session invalidée)
    """
    username = auth_tokens.pop(request.token, None)
    if username and username in USERS:
        USERS[username]["token"] = None

    return {
        "success": True,
        "message": "Déconnexion réussie. À bientôt sur TechShop !",
    }


@app.get("/me")
def get_profile(token: str):
    """
    Retourne le profil de l'utilisateur connecté.
    Utilisé pour vérifier qu'une session est bien active après login.
    Testé par : MOB-007 (vérification post-login)
    """
    username = _get_username_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré. Veuillez vous reconnecter.")
    user = USERS[username]
    return {
        "username": username,
        "name": user["name"],
        "authenticated": True,
        "orders": [k for k, v in ORDERS.items()],
    }


@app.get("/chat/authenticated")
def chat_authenticated_check(token: str):
    """
    Vérifie si un token donne accès au chat.
    Retourne 200 si authentifié, 401 sinon.
    Testé par : MOB-008 (token invalide → 401)
    """
    username = _get_username_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Accès refusé. Authentification requise.")
    return {"authenticated": True, "username": username}
