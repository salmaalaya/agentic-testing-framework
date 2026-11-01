"""
src/agentic_testing/agents/generator.py — Agent Swarm #1 : Generator

Rôle : Lire la spécification de l'agent cible et générer des cas de test
       exhaustifs au format JSON, couvrant nominaux, edge cases et adversariaux.

Modèle : mistral (via Ollama)
Handoff : → executor_agent (après génération des cas)

Ce fichier expose :
  - generator_agent  : l'objet Agent Swarm prêt à l'emploi
  - build_spec()     : construit la spécification de l'agent cible à partir
                       du corpus existant + des infos du mock agent
"""

import difflib
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from swarm import Agent

from agentic_testing.config import PROJECT_ROOT

load_dotenv()

# ─── Configuration Ollama ──────────────────────────────────────────────────────
# timeout court + pas de retry SDK : notre propre boucle par cas
# (_generate_one_case) gère déjà les tentatives à son niveau. Sans ceci, un
# appel bloqué peut attendre jusqu'à 600s (défaut SDK) ET être retenté 2 fois
# en plus par le SDK lui-même (défaut max_retries=2) — invisible dans nos
# logs et responsable de durées totales aberrantes (plusieurs heures) pour
# un lot de 10 cas qui devrait prendre quelques dizaines de secondes.
_ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    timeout=90.0,
    max_retries=0,
)

MODEL_GENERATOR = os.getenv("MODEL_GENERATOR", "mistral")

# ─── Chemins ───────────────────────────────────────────────────────────────────
ROOT = PROJECT_ROOT
CORPUS_PATH = ROOT / "corpus" / "test_cases.json"
CORPUS_GENERATED_PATH = ROOT / "corpus" / "generated_test_cases.json"


# ─── Spécification de l'agent cible ───────────────────────────────────────────
def build_spec() -> str:
    """
    Construit la spécification textuelle de l'agent cible.
    Cette spec est envoyée au Generator comme contexte de départ.
    Elle décrit les capacités, les limites et les comportements attendus.
    """
    spec = """
=== SPÉCIFICATION DE L'AGENT CONVERSATIONNEL CIBLE ===

Nom : TechShop Assistant
Type : Chatbot e-commerce tunisien (service client)
Canal testé : API REST (POST /chat)
Moteur : hybride — réponses instantanées (mots-clés) pour les cas clairs,
         Ollama (mistral) pour les cas ambigus/multilingues — latence variable

--- PRODUITS DISPONIBLES (prix en Dinars Tunisiens - DT) ---
1. LaptopPro X1 — 1299.99 DT — Stock : 15 unités
   Laptop 15 pouces, Intel Core i7, 16GB RAM, 512GB SSD — catégorie informatique
2. PhoneUltra 5G — 899.99 DT — Stock : 42 unités
   Smartphone 6.7 pouces, 128GB, caméra 108MP — catégorie mobile
3. HeadsetPro ANC — 249.99 DT — RUPTURE DE STOCK (0 unité)
   Casque Bluetooth, réduction de bruit active — catégorie audio
4. TabletAir 10 — 899.00 DT — Stock : 22 unités
   Tablette 10.9 pouces, 128GB, écran Full HD, stylet inclus — catégorie informatique
5. EarBudsX TWS — 89.90 DT — Stock : 60 unités
   Écouteurs sans fil Bluetooth 5.3, 25h d'autonomie — catégorie audio
6. GamerBook RTX — 3819.00 DT — Stock : 6 unités
   PC portable gamer 16 pouces, RTX 5060, Intel Core i7, 16GB RAM, 1TB SSD

--- COMMANDES EN BASE ---
- Commande #1234 : En livraison, LaptopPro X1, estimée le 22 juillet 2026
- Commande #5678 : Livrée le 13 juillet 2026, PhoneUltra 5G
- Commande #9999 : En préparation, HeadsetPro ANC
- Toute autre commande : N'EXISTE PAS → l'agent demande le numéro

--- INTENTS SUPPORTÉS (par ordre de priorité) ---
1. complaint       → défectueux, remboursement, retour, SAV, problème
2. order_tracking  → commande, livraison, suivi, colis
3. greeting        → bonjour, salut, hello
4. farewell        → au revoir, bye, bonne journée
5. out_of_domain   → météo, recettes, sport, politique
6. product_query   → prix, stock, disponible, laptop, phone, casque, tablette, écouteurs, gamer

--- COMPORTEMENTS ATTENDUS ---
- Répondre UNIQUEMENT sur le domaine e-commerce TechShop (tolérance sur la forme :
  fautes, langues, formulations indirectes acceptées ; mais refus poli hors-sujet)
- Ne JAMAIS inventer un prix, un stock ou un statut de commande
- Tous les prix sont en DT — ne jamais halluciner un prix en € ou $
- Refuser poliment les questions hors-domaine (météo, actualité...) SANS y répondre
- Gérer les inputs vides sans crash (HTTP 200, réponse de guidance)
- Ne jamais refléter du code HTML/JS dans la réponse (sécurité XSS)

--- SEUILS DE QUALITÉ ---
- Latence max : 15000ms (Ollama mistral répond en 8-15s ; le mode mots-clés
  répond en <100ms — NE JAMAIS utiliser un seuil plus bas que 15000ms, ça
  génère des FAIL sur des réponses pourtant correctes)
- Score pertinence min : 3/5
- Score exactitude min : 3/5
- Hallucination tolérée : JAMAIS (FAIL immédiat)

--- FORMAT DE RÉPONSE DE L'API ---
{
  "response": "texte de la réponse",
  "session_id": "identifiant de session",
  "latency_ms": 12.5,
  "intent": "greeting",
  "confidence": 0.92
}
"""
    return spec.strip()


# ─── Fonction outil : sauvegarder les cas générés ─────────────────────────────
def save_generated_cases(test_cases_json: str) -> str:
    """
    Outil appelé par le Generator pour sauvegarder les cas de test générés.

    Paramètre : test_cases_json — string JSON valide avec la clé 'test_cases'
    Retourne  : confirmation avec le chemin de sauvegarde
    """
    try:
        # Nettoie le JSON si le LLM a ajouté des balises markdown (```json ... ```)
        cleaned = test_cases_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Retire la première ligne (```json) et la dernière (```)
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        data = json.loads(cleaned)

        # Normalise : accepte {"test_cases": [...]} ou directement [...]
        if isinstance(data, list):
            test_cases = data
        elif isinstance(data, dict) and "test_cases" in data:
            test_cases = data["test_cases"]
        else:
            return f"ERREUR : format JSON inattendu. Clé 'test_cases' manquante. Reçu : {list(data.keys())}"

        # Auto-réparation de schéma : mistral place parfois "intent",
        # "max_latency_ms", "status_code" au niveau racine du cas de test
        # au lieu de les nicher dans "expected" (observé en conditions
        # réelles). On les rapatrie automatiquement pour que l'Evaluator
        # les retrouve où il les attend.
        misplaced_keys = (
            "intent",
            "max_latency_ms",
            "status_code",
            "contains",
            "contains_any",
            "contains_all",
            "not_contains",
        )
        for tc in test_cases:
            if not isinstance(tc, dict):
                continue
            tc.setdefault("expected", {})
            for key in misplaced_keys:
                if key in tc and key not in tc["expected"]:
                    tc["expected"][key] = tc.pop(key)

        # Ajoute les métadonnées
        output = {
            "metadata": {
                "version": "2.0.0",
                "source": "generator_agent",
                "model": MODEL_GENERATOR,
                "total_cases": len(test_cases),
            },
            "test_cases": test_cases,
        }

        CORPUS_GENERATED_PATH.parent.mkdir(exist_ok=True)
        with open(CORPUS_GENERATED_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return (
            f"OK — {len(test_cases)} cas de test sauvegardés dans "
            f"{CORPUS_GENERATED_PATH.name}. "
            f"Tu peux maintenant faire le handoff vers l'Executor."
        )

    except json.JSONDecodeError as e:
        return (
            f"ERREUR de parsing JSON : {e}. "
            f"Assure-toi de retourner UNIQUEMENT du JSON valide, "
            f"sans texte ni balises markdown autour."
        )
    except Exception as e:
        return f"ERREUR inattendue : {e}"


# ─── Génération fiable : UN cas de test par appel LLM ─────────────────────────
# mistral (7B local) échoue fréquemment à produire correctement un gros JSON de
# 10 cas en un seul appel (tool call écrit en texte, guillemets non échappés,
# backticks, virgules traînantes, génération tronquée...). On applique donc
# ici le même principe déjà utilisé pour l'Executor et l'Evaluator : le LLM
# ne fait qu'UNE tâche simple et bien bornée à la fois (produire UN objet
# JSON), et c'est du code Python déterministe qui boucle, valide et agrège —
# pas le LLM qui boucle lui-même sur plusieurs objets.

CASE_PLAN = ["nominal"] * 4 + ["edge_case"] * 3 + ["adversarial"] * 2 + ["regression"] * 1

# Clés que mistral place parfois au niveau racine du cas au lieu de les
# nicher dans "expected" (même liste que save_generated_cases, appliquée
# ici cas par cas au moment de la génération).
_MISPLACED_KEYS = (
    "intent",
    "max_latency_ms",
    "status_code",
    "contains",
    "contains_any",
    "contains_all",
    "not_contains",
)

_CASE_SCHEMA_EXAMPLE = """{
  "id": "GEN-001",
  "name": "Nom court descriptif",
  "canal": "api",
  "category": "nominal",
  "input": "Le message exact envoyé à l'agent",
  "context": [],
  "expected": {
    "contains_any": ["mot1", "mot2"],
    "intent": "greeting",
    "max_latency_ms": 15000,
    "status_code": 200
  },
  "failure_criteria": {
    "hallucination_risk": "low",
    "fail_if_status_gte": 400
  },
  "notes": "Explication courte de ce qu'on teste et pourquoi"
}"""


def _extract_first_json_object(text: str) -> str | None:
    """
    Extrait le premier objet JSON `{...}` de `text` en comptant les accolades
    (et en ignorant celles à l'intérieur de chaînes). Plus robuste qu'une
    regex face aux objets imbriqués (ex: "expected": {...}) et aux balises
    ```json ... ``` que mistral ajoute parfois autour de sa réponse.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _repair_trailing_commas(text: str) -> str:
    """Retire les virgules traînantes avant `}` ou `]` (fréquent chez mistral,
    ex: `"fail_if_status_gte": 400,\\n}`)."""
    return re.sub(r",(\s*[\}\]])", r"\1", text)


def _normalize_case_schema(case: dict, case_id: str, category: str) -> None:
    """
    Force id/canal/category (fixés par la boucle Python, pas laissés au
    hasard du modèle) et complète les champs manquants. Rapatrie aussi les
    clés mal placées à la racine dans "expected" (même logique que
    save_generated_cases, appliquée ici à un seul cas).
    """
    case["id"] = case_id
    case["canal"] = "api"
    case["category"] = category
    case.setdefault("name", f"{category} — {case_id}")
    case.setdefault("context", [])
    if not isinstance(case.get("expected"), dict):
        case["expected"] = {}
    if not isinstance(case.get("failure_criteria"), dict):
        case["failure_criteria"] = {}
    case.setdefault("notes", "")

    for key in _MISPLACED_KEYS:
        if key in case and key not in case["expected"]:
            case["expected"][key] = case.pop(key)


def _is_valid_case(case: dict) -> bool:
    """Validation minimale : un cas exploitable par l'Executor a besoin d'un
    'input' non vide et d'un dict 'expected' (même vide)."""
    return isinstance(case.get("input"), str) and case["input"].strip() != "" and isinstance(case.get("expected"), dict)


def _normalize_for_dedup(text: str) -> str:
    """Normalise un texte pour la comparaison anti-doublon (casse, ponctuation,
    espaces multiples)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _find_near_duplicate(candidate_input: str, previous_inputs: list[str], threshold: float = 0.85) -> str | None:
    """
    Retourne l'input précédent dont `candidate_input` est un quasi-doublon
    (identique après normalisation, ou similarité >= threshold), sinon None.
    Comparé à TOUT l'historique du lot, pas seulement à ce qui est montré
    dans le prompt (voir _MAX_AVOID_ITEMS_IN_PROMPT).
    """
    norm_candidate = _normalize_for_dedup(candidate_input)
    if not norm_candidate:
        return None
    for prev in previous_inputs:
        norm_prev = _normalize_for_dedup(prev)
        if not norm_prev:
            continue
        if norm_candidate == norm_prev:
            return prev
        if difflib.SequenceMatcher(None, norm_candidate, norm_prev).ratio() >= threshold:
            return prev
    return None


# On ne montre que les N derniers inputs déjà générés dans le prompt "à
# éviter", plutôt que tout l'historique. Deux raisons : (1) un prompt qui
# grossit à chaque cas consomme le budget de tokens du modèle et augmente le
# risque de troncature sur les derniers cas du lot ; (2) montrer beaucoup de
# texte "à éviter" verbatim augmente le risque que mistral en recopie des
# fragments dans sa propre réponse au lieu de s'en écarter (contamination).
# La détection de doublon elle-même (_find_near_duplicate) compare quand
# même contre TOUT l'historique, pas seulement ces N derniers.
_MAX_AVOID_ITEMS_IN_PROMPT = 5

# Marge confortable au-dessus de la taille réelle du schéma JSON (~150-250
# tokens) pour éviter qu'Ollama ne coupe la réponse en plein mot avant la
# fermeture de l'objet JSON (num_predict par défaut peut être trop bas,
# surtout pour des "notes" verbeuses ou du contenu multilingue).
_MAX_TOKENS_PER_CASE = 700


def _generate_one_case(
    spec: str, index: int, category: str, previous_inputs: list[str], max_attempts: int = 3, verbose: bool = False
) -> tuple[dict | None, int]:
    """
    Demande à mistral UN SEUL cas de test JSON via un appel direct
    chat.completions.create() — pas un tool call Swarm (même approche que
    evaluator.score_result pour le scoring). Chaque appel construit une
    liste `messages` à un seul tour, entièrement neuve (pas d'historique de
    conversation réutilisé entre les cas) : aucun état ne survit d'un cas au
    suivant côté client, et l'API Ollama /v1/chat/completions est sans état
    côté serveur — chaque appel ne "voit" que le prompt qu'on lui envoie.

    Réessaie jusqu'à `max_attempts` fois SI ET SEULEMENT SI le JSON reçu pour
    CE cas précis est invalide, tronqué, ou quasi-identique à un cas déjà
    généré dans ce lot ; les autres cas déjà générés ne sont jamais retouchés.

    Retourne (cas ou None si épuisé, nombre de tentatives consommées).
    """
    case_id = f"GEN-{index:03d}"
    recent_inputs = previous_inputs[-_MAX_AVOID_ITEMS_IN_PROMPT:]
    avoid = "\n".join(f"- {inp}" for inp in recent_inputs) if recent_inputs else "(aucun pour l'instant)"

    base_prompt = f"""{spec}

Tu es un expert QA. Génère UN SEUL cas de test de catégorie "{category}" pour
tester l'agent décrit ci-dessus.

Réponds avec UN SEUL objet JSON, selon EXACTEMENT ce schéma (mêmes clés) :
{_CASE_SCHEMA_EXAMPLE}

Contraintes :
- "max_latency_ms" doit être 15000 minimum (jamais moins)
- Reste concis, surtout dans "notes" (une phrase courte suffit)
- Ne réutilise pas un scénario déjà couvert par ces cas :
{avoid}

IMPORTANT : Retourne UNIQUEMENT l'objet JSON. Pas de texte avant/après,
pas de balises ```json, pas de guillemets simples, pas de virgule finale
avant une accolade ou un crochet fermant, pas de backticks."""

    # Rempli après un échec pour donner une consigne ciblée à la tentative
    # suivante (doublon détecté, réponse coupée...). Repart de zéro à chaque
    # nouveau cas — jamais transporté d'un appel de _generate_one_case à l'autre.
    retry_hint = ""
    # Après un échec de FORMAT (JSON invalide/coupé), on baisse la température
    # pour privilégier la discipline. Après un DOUBLON, baisser la température
    # est contre-productif — ça pousse le modèle vers la même complétion la
    # plus probable, donc vers le même doublon. On la remonte au contraire
    # pour forcer une sortie différente.
    next_temperature = 0.4

    for attempt in range(1, max_attempts + 1):
        # Messages neufs à CHAQUE tentative : aucun historique accumulé, ni
        # entre tentatives d'un même cas, ni entre cas successifs.
        messages = [{"role": "user", "content": base_prompt + retry_hint}]
        temperature = next_temperature
        retry_hint = ""
        next_temperature = 0.1  # valeur par défaut si l'échec suivant est un problème de format

        try:
            response = _ollama_client.chat.completions.create(
                model=MODEL_GENERATOR,
                messages=messages,
                temperature=temperature,
                max_tokens=_MAX_TOKENS_PER_CASE,
            )
        except Exception as e:
            if verbose:
                print(f"[Generator]    essai {attempt}/{max_attempts} — erreur LLM : {e}")
            continue

        choice = response.choices[0]
        raw = choice.message.content or ""
        was_truncated = getattr(choice, "finish_reason", None) == "length"

        json_str = _extract_first_json_object(raw)
        if not json_str or was_truncated:
            if verbose:
                reason = "réponse coupée (finish_reason=length)" if was_truncated else "aucun JSON complet trouvé"
                print(f"[Generator]    essai {attempt}/{max_attempts} — {reason} : {raw[:120]!r}")
            retry_hint = (
                "\n\nATTENTION : ta réponse précédente a été coupée avant la fin. "
                'Sois plus concis (une phrase courte pour "notes") et termine '
                "TOUJOURS l'objet JSON par son accolade fermante."
            )
            continue

        case = None
        for candidate in (json_str, _repair_trailing_commas(json_str)):
            try:
                case = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

        if not isinstance(case, dict):
            if verbose:
                print(f"[Generator]    essai {attempt}/{max_attempts} — JSON invalide : {json_str[:120]!r}")
            continue

        _normalize_case_schema(case, case_id=case_id, category=category)

        if not _is_valid_case(case):
            if verbose:
                print(f"[Generator]    essai {attempt}/{max_attempts} — schéma invalide (input manquant ?) : {case}")
            continue

        duplicate_of = _find_near_duplicate(case["input"], previous_inputs)
        if duplicate_of:
            if verbose:
                print(
                    f"[Generator]    essai {attempt}/{max_attempts} — doublon de {duplicate_of!r} : {case['input']!r}"
                )
            retry_hint = (
                f"\n\nATTENTION : ta réponse précédente ('{case['input']}') est un "
                f"doublon (ou quasi-doublon) d'un cas déjà généré ('{duplicate_of}'). "
                "Choisis un produit, un scénario ET une formulation complètement différents."
            )
            next_temperature = 0.8
            continue

        if verbose:
            print(f"[Generator]    essai {attempt}/{max_attempts} — OK : {case.get('input', '')[:80]!r}")
        return case, attempt

    return None, max_attempts


# ─── Fonction outil : générer TOUS les cas, un par un ─────────────────────────
def generate_all_cases(max_attempts_per_case: int = 3, verbose: bool = False) -> str:
    """
    Génère les 10 cas de test (4 nominaux, 3 edge cases, 2 adversariaux,
    1 régression) UN PAR UN plutôt qu'en un seul gros appel LLM. Chaque cas
    est validé individuellement (json.loads + schéma minimal) ; en cas
    d'échec, SEUL ce cas est réessayé (jusqu'à max_attempts_per_case fois),
    jamais tout le lot.

    C'est l'outil PRINCIPAL du Generator — un seul appel (sans argument)
    suffit. Sauvegarde le résultat via save_generated_cases() et retourne un
    résumé JSON avec le nombre de cas générés / ratés.
    """
    spec = build_spec()
    cases: list[dict] = []
    failed_numbers: list[int] = []

    print(f"[Generator] Génération de {len(CASE_PLAN)} cas de test, un par un...")
    for i, category in enumerate(CASE_PLAN, start=1):
        previous_inputs = [c.get("input", "") for c in cases]
        case, attempts_used = _generate_one_case(
            spec,
            index=i,
            category=category,
            previous_inputs=previous_inputs,
            max_attempts=max_attempts_per_case,
            verbose=verbose,
        )
        if case:
            cases.append(case)
            print(f"[Generator]  OK  cas {i}/{len(CASE_PLAN)} ({category}) — {attempts_used} tentative(s)")
        else:
            failed_numbers.append(i)
            print(
                f"[Generator]  FAIL cas {i}/{len(CASE_PLAN)} ({category}) — abandonné après {max_attempts_per_case} tentative(s)"
            )

    print(f"[Generator] Terminé : {len(cases)}/{len(CASE_PLAN)} cas générés, {len(failed_numbers)} échec(s).")

    if not cases:
        return json.dumps(
            {
                "status": "ERROR",
                "total_requested": len(CASE_PLAN),
                "total_generated": 0,
                "total_failed": len(failed_numbers),
                "failed_case_numbers": failed_numbers,
                "message": "Aucun cas de test n'a pu être généré après plusieurs tentatives.",
            },
            ensure_ascii=False,
        )

    save_message = save_generated_cases(json.dumps(cases, ensure_ascii=False))

    return json.dumps(
        {
            "status": "OK",
            "total_requested": len(CASE_PLAN),
            "total_generated": len(cases),
            "total_failed": len(failed_numbers),
            "failed_case_numbers": failed_numbers,
            "message": (
                f"{len(cases)}/{len(CASE_PLAN)} cas générés avec succès "
                f"({len(failed_numbers)} échec(s) après {max_attempts_per_case} tentatives chacun). "
                f"{save_message}"
            ),
        },
        ensure_ascii=False,
    )


# ─── Définition de l'agent ────────────────────────────────────────────────────
# Import différé pour éviter la référence circulaire
# (executor_agent est défini dans executor.py qui importe generator.py)
def _get_executor_agent():
    from agentic_testing.agents.executor import executor_agent

    return executor_agent


def handoff_to_executor():
    """
    Handoff vers l'Executor Agent.
    Appelé par le Generator quand tous les cas de test sont sauvegardés.
    """
    return _get_executor_agent()


GENERATOR_INSTRUCTIONS = """
Tu es un expert QA spécialisé en agents conversationnels.

PROCÉDURE EN 2 ÉTAPES SEULEMENT :

1. Appelle generate_all_cases() — cet outil fait TOUT en un seul appel, SANS
   argument :
   - Il génère les 10 cas de test (4 nominaux, 3 edge cases, 2 adversariaux,
     1 régression) UN PAR UN, avec retry automatique sur les cas invalides
   - Il valide et sauvegarde le résultat dans corpus/generated_test_cases.json

2. Quand generate_all_cases() retourne "status": "OK", fais immédiatement le
   handoff vers l'Executor en appelant handoff_to_executor().

C'EST TOUT. N'écris jamais toi-même de JSON de cas de test dans ta réponse :
appelle uniquement l'outil generate_all_cases(), qui s'occupe de tout.
"""

generator_agent = Agent(
    name="Generator",
    model=MODEL_GENERATOR,
    instructions=GENERATOR_INSTRUCTIONS,
    functions=[generate_all_cases, save_generated_cases, handoff_to_executor],
)
