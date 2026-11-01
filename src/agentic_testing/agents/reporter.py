"""
src/agentic_testing/agents/reporter.py — Agent Swarm #4 : Reporter

Rôle : Lire les résultats scorés et produire un rapport Markdown structuré
       + un fichier JSON de synthèse final.

Modèle : mistral (bon rédacteur structuré)
Fin du pipeline — pas de handoff après.

Ce fichier expose :
  - reporter_agent       : l'objet Agent Swarm prêt à l'emploi
  - generate_md_report() : construit le rapport Markdown à partir des scores
  - save_report()        : sauvegarde le rapport sur disque
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from swarm import Agent

from agentic_testing.config import PROJECT_ROOT

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
_ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)

MODEL_REPORTER = os.getenv("MODEL_REPORTER", "mistral")

# ─── Chemins ───────────────────────────────────────────────────────────────────
ROOT = PROJECT_ROOT
SCORES_PATH = ROOT / "reports" / "scored_results.json"
REPORT_MD = ROOT / "reports" / "test_report.md"
REPORT_JSON = ROOT / "reports" / "final_summary.json"


# ─── Fonction outil : charger les résultats scorés ────────────────────────────
def load_scored_results() -> str:
    """
    Charge les résultats scorés depuis reports/scored_results.json.
    Retourne leur contenu JSON sous forme de string.
    """
    if not SCORES_PATH.exists():
        return "ERREUR : reports/scored_results.json introuvable. L'Evaluator doit tourner d'abord."
    with open(SCORES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False)


# ─── Fonction outil : générer le rapport Markdown ────────────────────────────
def generate_md_report(scored_results_json: str, web_raw: str = None, mobile_raw: str = None) -> str:
    """
    Génère le rapport Markdown complet — tricanal si les scores web et mobile
    sont disponibles. Calcule automatiquement la divergence API/Web/Mobile.

    Paramètres :
      scored_results_json — JSON scores API (obligatoire)
      web_raw             — JSON scores Web (optionnel)
      mobile_raw          — JSON scores Mobile (optionnel)
    """
    try:
        data = json.loads(scored_results_json)
        summary = data.get("summary", {})
        scored = data.get("scored_results", [])

        web_data = json.loads(web_raw) if web_raw else None
        mobile_data = json.loads(mobile_raw) if mobile_raw else None

        # ── En-tête ──────────────────────────────────────────────────────────
        now = datetime.now().strftime("%d/%m/%Y à %H:%M")
        md = [
            "# Rapport de Testing Agentique\n",
            f"> Généré le {now} par le pipeline Swarm + Ollama\n",
            "---\n",
        ]

        # ── Résumé global ─────────────────────────────────────────────────────
        total = summary.get("total", len(scored))
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        warned = summary.get("warned", 0)
        errors = summary.get("errors", 0)
        rate = summary.get("success_rate_pct", 0)
        hallucs = summary.get("hallucinations_detected", 0)
        avg = summary.get("avg_scores", {})

        status_emoji = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"

        md.append(f"## {status_emoji} Résumé Global\n")
        md.append("| Métrique | Valeur |\n|---|---|\n")
        md.append(f"| Taux de réussite | **{rate}%** ({passed}/{total}) |\n")
        md.append(f"| PASS | {passed} |\n")
        md.append(f"| FAIL | {failed} |\n")
        md.append(f"| WARN | {warned} |\n")
        md.append(f"| ERROR | {errors} |\n")
        md.append(f"| Hallucinations détectées | {'⚠️ ' + str(hallucs) if hallucs > 0 else '✅ 0'} |\n")
        md.append(f"| Score moyen pertinence | {avg.get('pertinence', 'N/A')}/5 |\n")
        md.append(f"| Score moyen exactitude | {avg.get('exactitude', 'N/A')}/5 |\n")
        md.append(f"| Score moyen cohérence  | {avg.get('coherence', 'N/A')}/5 |\n\n")

        # ── Tableau détaillé ──────────────────────────────────────────────────
        md.append("## 📋 Tableau Détaillé des Résultats\n\n")
        md.append("| ID | Nom | Canal | Catégorie | Latence | Score | Hallu. | Verdict | Raison |\n")
        md.append("|---|---|---|---|---|---|---|---|---|\n")

        for s in scored:
            verdict = s.get("verdict", "?")
            v_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "ERROR": "🔴"}.get(verdict, "?")
            hallu = "⚠️ OUI" if s.get("hallucination") else "✅ non"
            score = s.get("score_moyen", 0)
            latency = s.get("latency_ms", 0)
            reason = s.get("reason", "")[:60]

            md.append(
                f"| {s.get('id', '?')} | {s.get('name', '?')} | {s.get('canal', '?')} "
                f"| {s.get('category', '?')} | {latency}ms "
                f"| {score}/5 | {hallu} | {v_icon} {verdict} | {reason} |\n"
            )

        md.append("\n")

        # ── Métriques NLP complémentaires (BLEU / ROUGE-L / NLI) ────────────────
        # INFORMATIF UNIQUEMENT — n'influence jamais le verdict PASS/FAIL
        # ci-dessus (basé sur hallucination + score_moyen + status_code).
        # BLEU/ROUGE-L : uniquement pour les cas avec expected.reference_response
        # (champ optionnel du corpus). NLI : score d'entailment vs le contexte
        # factuel connu (catalogue/commandes) — renfort du jugement LLM
        # "hallucination", pas un remplacement. Absent (—) si le modèle NLI
        # n'est pas installé sur cette machine (cf. requirements.txt).
        nlp_rows = [
            s
            for s in scored
            if s.get("bleu_score") is not None
            or s.get("rouge_l_score") is not None
            or s.get("nli_entailment_score") is not None
        ]
        md.append("## 🧮 Métriques NLP Complémentaires (BLEU / ROUGE-L / NLI)\n\n")
        md.append("> Purement informatif — n'affecte pas le verdict PASS/FAIL ci-dessus.\n\n")
        if nlp_rows:
            md.append("| ID | BLEU | ROUGE-L | NLI (entailment) |\n|---|---|---|---|\n")
            for s in nlp_rows:
                bleu = s.get("bleu_score")
                rouge_l = s.get("rouge_l_score")
                nli = s.get("nli_entailment_score")
                md.append(
                    f"| {s.get('id', '?')} "
                    f"| {bleu if bleu is not None else '—'} "
                    f"| {rouge_l if rouge_l is not None else '—'} "
                    f"| {nli if nli is not None else '—'} |\n"
                )
            md.append("\n")
        else:
            md.append(
                "Aucune métrique disponible pour ce run — soit aucun cas du corpus "
                "n'a de `reference_response` renseigné (BLEU/ROUGE), soit le modèle "
                "NLI n'est pas installé sur cette machine (voir `requirements.txt`, "
                "section NLI).\n\n"
            )

        # ── Top 3 échecs critiques ────────────────────────────────────────────
        failures = [s for s in scored if s.get("verdict") in ("FAIL", "ERROR")]
        failures.sort(key=lambda x: x.get("score_moyen", 5))  # Plus bas score en premier

        if failures:
            md.append("## 🚨 Échecs Critiques\n\n")
            for i, f in enumerate(failures[:3], 1):
                md.append(f"### #{i} — {f.get('id')} : {f.get('name')}\n\n")
                md.append(f"- **Canal** : {f.get('canal')}  \n")
                md.append(f"- **Catégorie** : {f.get('category')}  \n")
                md.append(f"- **Input** : `{f.get('input', '')}`  \n")
                md.append(f"- **Réponse obtenue** : {f.get('actual', '(vide)')[:200]}  \n")
                md.append(
                    f"- **Scores** : pertinence={f.get('pertinence', 0)} "
                    f"exactitude={f.get('exactitude', 0)} "
                    f"cohérence={f.get('coherence', 0)}  \n"
                )
                md.append(f"- **Hallucination** : {'OUI ⚠️' if f.get('hallucination') else 'non'}  \n")
                md.append(f"- **Raison FAIL** : {f.get('reason', '?')}  \n")
                if f.get("notes"):
                    md.append(f"- **Note QA** : {f.get('notes')}  \n")
                md.append("\n")
        else:
            md.append("## ✅ Aucun échec critique\n\nTous les tests ont passé.\n\n")

        # ── Hallucinations détaillées ─────────────────────────────────────────
        hallu_cases = [s for s in scored if s.get("hallucination") is True]
        if hallu_cases:
            md.append("## ⚠️ Hallucinations Détectées\n\n")
            md.append("Ces cas sont des **FAIL bloquants** — l'agent a inventé des informations.\n\n")
            for h in hallu_cases:
                md.append(f"- **{h.get('id')}** — {h.get('name')} : `{h.get('input', '')}`\n")
                md.append(f"  Réponse : *{h.get('actual', '')[:150]}*\n\n")

        # ── Répartition par canal (si multicanal) ────────────────────────────
        all_scored = list(scored)
        if web_data:
            all_scored += web_data.get("scored_results", [])
        if mobile_data:
            all_scored += mobile_data.get("scored_results", [])

        canaux = {}
        for s in all_scored:
            c = s.get("canal", "api")
            if c not in canaux:
                canaux[c] = {"total": 0, "pass": 0, "score_sum": 0.0}
            canaux[c]["total"] += 1
            if s.get("verdict") == "PASS":
                canaux[c]["pass"] += 1
            canaux[c]["score_sum"] += s.get("score_moyen", 0)

        if len(canaux) > 1:
            md.append("## 📡 Résultats par Canal\n\n")
            md.append("| Canal | Tests | PASS | Taux | Score moyen |\n|---|---|---|---|---|\n")
            for c, stats in sorted(canaux.items()):
                c_rate = round(stats["pass"] / stats["total"] * 100, 1) if stats["total"] else 0
                c_score = round(stats["score_sum"] / stats["total"], 2) if stats["total"] else 0
                icon = {"api": "🔌", "web": "🌐", "mobile": "📱"}.get(c, "?")
                md.append(f"| {icon} {c.upper()} | {stats['total']} | {stats['pass']} | {c_rate}% | {c_score}/5 |\n")
            md.append("\n")

            # ── Divergence Web ↔ API ──────────────────────────────────────────
            if web_data:
                api_by_id = {s["id"]: s for s in scored}
                web_scored = web_data.get("scored_results", [])
                divergences = []
                max_div = float(os.getenv("MAX_WEB_MOBILE_DIVERGENCE", "0.20"))

                for ws in web_scored:
                    mirror_id = ws.get("api_mirror")
                    if mirror_id and mirror_id in api_by_id:
                        api_s = api_by_id[mirror_id]
                        api_score = api_s.get("score_moyen", 0)
                        web_score = ws.get("score_moyen", 0)
                        delta = abs(api_score - web_score)
                        if delta > 0:
                            divergences.append(
                                {
                                    "web_id": ws["id"],
                                    "api_id": mirror_id,
                                    "name": ws.get("name", "?"),
                                    "api_score": api_score,
                                    "web_score": web_score,
                                    "delta": round(delta, 2),
                                    "over_threshold": delta > max_div * 5,
                                }
                            )

                if divergences:
                    divergences.sort(key=lambda x: -x["delta"])
                    md.append("## 🔄 Divergence Web ↔ API\n\n")
                    md.append(f"> Seuil d'alerte : Δ > {max_div * 5:.1f}/5 ({int(max_div * 100)}%)\n\n")
                    md.append("| Cas Web | Cas API | Nom | Score API | Score Web | Δ | Alerte |\n")
                    md.append("|---|---|---|---|---|---|---|\n")
                    for d in divergences:
                        alert = "⚠️ DIVERGENCE" if d["over_threshold"] else "✅ OK"
                        md.append(
                            f"| {d['web_id']} | {d['api_id']} | {d['name'][:35]} "
                            f"| {d['api_score']}/5 | {d['web_score']}/5 "
                            f"| **{d['delta']}** | {alert} |\n"
                        )
                    over = [d for d in divergences if d["over_threshold"]]
                    if over:
                        md.append(
                            f"\n⚠️ **{len(over)} cas dépassent le seuil de divergence** — "
                            f"l'agent répond différemment sur Web et API.\n"
                        )
                    md.append("\n")

            # ── Divergence Mobile ↔ API ───────────────────────────────────────
            if mobile_data:
                api_by_id = {s["id"]: s for s in scored}
                mob_scored = mobile_data.get("scored_results", [])
                max_div = float(os.getenv("MAX_WEB_MOBILE_DIVERGENCE", "0.20"))
                mob_divs = []

                for ms in mob_scored:
                    mirror_id = ms.get("api_mirror")
                    if mirror_id and mirror_id in api_by_id:
                        api_s = api_by_id[mirror_id]
                        delta = abs(api_s.get("score_moyen", 0) - ms.get("score_moyen", 0))
                        if delta > 0:
                            mob_divs.append(
                                {
                                    "mob_id": ms["id"],
                                    "api_id": mirror_id,
                                    "name": ms.get("name", "?"),
                                    "api_score": api_s.get("score_moyen", 0),
                                    "mob_score": ms.get("score_moyen", 0),
                                    "delta": round(delta, 2),
                                    "over": delta > max_div * 5,
                                }
                            )

                if mob_divs:
                    mob_divs.sort(key=lambda x: -x["delta"])
                    md.append("## 📱 Divergence Mobile ↔ API\n\n")
                    md.append(f"> Seuil d'alerte : Δ > {max_div * 5:.1f}/5\n\n")
                    md.append("| Cas Mobile | Cas API | Nom | Score API | Score Mob | Δ | Alerte |\n")
                    md.append("|---|---|---|---|---|---|---|\n")
                    for d in mob_divs:
                        alert = "⚠️ DIVERGENCE" if d["over"] else "✅ OK"
                        md.append(
                            f"| {d['mob_id']} | {d['api_id']} | {d['name'][:35]} "
                            f"| {d['api_score']}/5 | {d['mob_score']}/5 "
                            f"| **{d['delta']}** | {alert} |\n"
                        )
                    md.append("\n")

        # ── Répartition par catégorie ─────────────────────────────────────────
        md.append("## 📊 Répartition par Catégorie\n\n")
        categories = {}
        for s in scored:
            cat = s.get("category", "?")
            if cat not in categories:
                categories[cat] = {"total": 0, "pass": 0}
            categories[cat]["total"] += 1
            if s.get("verdict") == "PASS":
                categories[cat]["pass"] += 1

        md.append("| Catégorie | Tests | PASS | Taux |\n|---|---|---|---|\n")
        for cat, stats in sorted(categories.items()):
            cat_rate = round(stats["pass"] / stats["total"] * 100, 1) if stats["total"] else 0
            md.append(f"| {cat} | {stats['total']} | {stats['pass']} | {cat_rate}% |\n")
        md.append("\n")

        # ── Recommandations ───────────────────────────────────────────────────
        md.append("## 💡 Recommandations\n\n")

        if hallucs > 0:
            md.append(
                f"1. ⚠️ **Hallucinations critiques** : {hallucs} cas détecté(s). "
                f"Renforcer le RAG ou les guardrails du modèle.  \n"
            )
        if failed > 0:
            low_scores = [s for s in scored if s.get("exactitude", 5) <= 2]
            if low_scores:
                md.append(
                    f"2. 📉 **Exactitude faible** sur {len(low_scores)} test(s). Vérifier la base de connaissances.  \n"
                )
        if errors > 0:
            md.append(
                f"3. 🔴 **{errors} erreur(s) de connexion**. Vérifier la disponibilité de l'agent et les timeouts.  \n"
            )
        if rate >= 80:
            md.append(
                "✅ Le taux de réussite est satisfaisant (≥ 80%). "
                "Déploiement possible sous réserve de traiter les échecs critiques.  \n"
            )
        else:
            md.append(
                "❌ Le taux de réussite est insuffisant (< 80%). Ne pas déployer avant correction des échecs.  \n"
            )

        md.append("\n---\n")
        md.append(f"*Rapport généré automatiquement par le pipeline Swarm — {now}*\n")

        return "".join(md)

    except Exception as e:
        return f"# ERREUR lors de la génération du rapport\n\n{e}"


# ─── Fonction outil : sauvegarder le rapport ─────────────────────────────────
def save_report(report_content: str) -> str:
    """
    Sauvegarde le rapport Markdown dans reports/test_report.md.
    Sauvegarde aussi un JSON de synthèse dans reports/final_summary.json.

    Paramètre : report_content — contenu Markdown du rapport
    Retourne  : confirmation avec les chemins des fichiers créés
    """
    try:
        REPORT_MD.parent.mkdir(exist_ok=True)

        # Sauvegarde Markdown
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(report_content)

        # Sauvegarde JSON de synthèse (si les scores sont disponibles)
        summary_saved = False
        if SCORES_PATH.exists():
            with open(SCORES_PATH, encoding="utf-8") as f:
                scores_data = json.load(f)

            summary = {
                "generated_at": datetime.now().isoformat(),
                "pipeline": "Swarm (Generator → Executor → Evaluator → Reporter)",
                "summary": scores_data.get("summary", {}),
            }
            with open(REPORT_JSON, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary_saved = True

        return (
            f"✅ Rapport sauvegardé :\n"
            f"  - {REPORT_MD}\n"
            f"  - {REPORT_JSON if summary_saved else '(JSON non généré)'}\n\n"
            f"Pipeline terminé avec succès !"
        )
    except Exception as e:
        return f"ERREUR lors de la sauvegarde : {e}"


# ─── Fonction outil principale : tout en un seul appel ───────────────────────
def generate_and_save_report() -> str:
    """
    Charge les scores (API + Web si disponibles), génère le rapport Markdown
    complet avec section divergence et le sauvegarde en un seul appel.
    C'est l'outil PRINCIPAL du Reporter.
    """
    # Chargement des scores API
    raw = load_scored_results()
    if raw.startswith("ERREUR"):
        return json.dumps({"status": "ERROR", "message": raw})

    # Chargement optionnel des scores Web
    web_scores_path = ROOT / "reports" / "web_scored_results.json"
    web_raw = None
    if web_scores_path.exists():
        with open(web_scores_path, encoding="utf-8") as f:
            web_raw = json.dumps(json.load(f), ensure_ascii=False)

    # Chargement optionnel des scores Mobile
    mobile_scores_path = ROOT / "reports" / "mobile_scored_results.json"
    mobile_raw = None
    if mobile_scores_path.exists():
        with open(mobile_scores_path, encoding="utf-8") as f:
            mobile_raw = json.dumps(json.load(f), ensure_ascii=False)

    # Génération du Markdown (avec divergence si scores web/mobile disponibles)
    md_content = generate_md_report(raw, web_raw, mobile_raw)

    # Sauvegarde
    save_msg = save_report(md_content)

    return json.dumps(
        {
            "status": "OK",
            "report_md": str(REPORT_MD),
            "report_json": str(REPORT_JSON),
            "web_scores_included": web_raw is not None,
            "mobile_scores_included": mobile_raw is not None,
            "message": f"Rapport généré et sauvegardé. {save_msg[:100]}",
        },
        ensure_ascii=False,
    )


REPORTER_INSTRUCTIONS = """
Tu es un ingénieur QA senior chargé de produire le rapport final de testing.

PROCÉDURE EN 2 ÉTAPES SEULEMENT :

1. Appelle generate_and_save_report() — cet outil fait TOUT en un seul appel :
   - Il charge les résultats scorés depuis reports/scored_results.json
   - Il génère le rapport Markdown complet avec métriques, tableau et recommandations
   - Il sauvegarde le rapport dans reports/test_report.md et reports/final_summary.json

2. Quand generate_and_save_report retourne "status": "OK", le pipeline est terminé.
   Réponds avec un message de confirmation court mentionnant le taux de réussite.

C'EST TOUT. Un seul appel à generate_and_save_report suffit.
Ne tente pas d'appeler load_scored_results, generate_md_report ou save_report séparément.
"""

reporter_agent = Agent(
    name="Reporter",
    model=MODEL_REPORTER,
    instructions=REPORTER_INSTRUCTIONS,
    functions=[generate_and_save_report, load_scored_results, generate_md_report, save_report],
)
