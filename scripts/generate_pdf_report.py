"""
scripts/generate_pdf_report.py — Rapport PDF en langage clair

Objectif : un rapport que N'IMPORTE QUI peut lire (responsable non-technicien,
jury, direction) — sans jargon d'ingénieur. Pas de "PASS/FAIL", pas de
"hallucination" non expliquée, pas de "latence" sans traduction.

Contenu :
  1. Page de garde avec le chiffre clé
  2. "Qu'est-ce que ce document ?" — en une minute
  3. Comment fonctionne le système (les 4 robots, en langage simple)
  4. Résultats globaux (graphique + explication)
  5. Résultats détaillés par canal, en langage clair
  6. Exemples concrets avec VRAIES captures d'écran
  7. "Qu'est-ce qu'une hallucination ?" (encadré pédagogique)
  8. Recommandations

Usage :
    python scripts/generate_pdf_report.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
SCREENSHOTS = ROOT / "screenshots"
TMP = ROOT / "reports" / "_pdf_assets"
TMP.mkdir(parents=True, exist_ok=True)

SCORE_FILES = {
    "API": (
        REPORTS / "scored_results.json",
        "Test technique direct",
        "On interroge directement le \u00abcerveau\u00bb du robot, sans passer par un site ou une appli, "
        "comme un technicien qui teste un moteur sur un banc d'essai.",
    ),
    "Web": (
        REPORTS / "web_scored_results.json",
        "Test sur le site web",
        "On utilise un vrai navigateur (comme Chrome) pour cliquer, taper et discuter avec le robot "
        "exactement comme le ferait un client sur le site TechShop.",
    ),
    "Mobile": (
        REPORTS / "mobile_scored_results.json",
        "Test sur l'application mobile",
        "On pilote une vraie application Android pour v\u00e9rifier que le robot fonctionne aussi bien "
        "sur t\u00e9l\u00e9phone que sur ordinateur.",
    ),
}

INDIGO = colors.HexColor("#4F46E5")
INDIGO_DARK = colors.HexColor("#312E81")
INDIGO_SOFT = colors.HexColor("#EEF2FF")
INK = colors.HexColor("#1E293B")
GRAY = colors.HexColor("#64748B")
GREEN = colors.HexColor("#16A34A")
GREEN_SOFT = colors.HexColor("#DCFCE7")
RED = colors.HexColor("#DC2626")
RED_SOFT = colors.HexColor("#FEE2E2")
AMBER = colors.HexColor("#D97706")
AMBER_SOFT = colors.HexColor("#FEF3C7")
LINE = colors.HexColor("#E2E8F0")


# ── Chargement des données ────────────────────────────────────────────────
def _load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _plain_reason(reason: str, verdict: str) -> str:
    """Traduit les messages techniques en phrases compréhensibles par tous."""
    if not reason:
        return "Bonne réponse, sans problème particulier." if verdict == "PASS" else "Réponse jugée insuffisante."
    r = reason

    m = re.search(r"Latence ([\d.]+)ms dépasse le max (\d+)ms", r)
    if m:
        secs = round(float(m.group(1)) / 1000, 1)
        return (
            f"Le robot a mis {secs} secondes à répondre — c'était considéré trop lent au moment du test "
            f"(le seuil autorisé a depuis été ajusté pour refléter le temps de réponse réel du robot)."
        )

    if "hallucination" in r.lower():
        return "Le robot a inventé une information qui n'existe pas réellement (voir l'encadré \u00abQu'est-ce qu'une hallucination ?\u00bb)."

    if re.search(r"mot.*absent|mot.*manquant|contains", r, re.IGNORECASE):
        return "La réponse ne contenait pas les informations attendues."

    if re.search(r"interdit|not_contains", r, re.IGNORECASE):
        return "La réponse contenait une information qui n'aurait pas dû y figurer (risque de sécurité)."

    if re.search(r"vide|empty", r, re.IGNORECASE):
        return "Le robot n'a donné aucune réponse."

    if re.search(r"hors[\s-]?sujet", r, re.IGNORECASE):
        return "Le robot a répondu à côté de la question, ou a inventé une réponse qui ne correspond pas à ce qui était réellement demandé."

    if verdict == "PASS":
        return "La réponse répondait correctement à la question posée."

    # Nettoyage générique : on retire le jargon technique entre parenthèses
    # (ex. "(confiance : 1.0)") pour garder une phrase lisible par tous.
    cleaned = re.sub(r"\s*\((?:confiance|score|intent)[^)]*\)", "", r, flags=re.IGNORECASE).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or "Réponse jugée insuffisante par le système de vérification."


def _fmt_pct(n: int, total: int) -> str:
    return f"{round(n / total * 100)}%" if total else "0%"


# ── Graphiques (images statiques, non techniques) ─────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": "#1E293B",
        "axes.edgecolor": "#E2E8F0",
        "axes.labelcolor": "#1E293B",
    }
)


def make_donut(passed: int, failed: int, other: int, out: Path) -> Path:
    """Dessine le graphique en anneau (bonnes réponses / à corriger / autres) et l'enregistre en PNG."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=200)
    values = [v for v in [passed, failed, other] if v > 0]
    labels_all = ["Bonnes réponses", "Réponses à corriger", "Autres cas"]
    colors_all = ["#16A34A", "#DC2626", "#94A3B8"]
    labels = [lbl for lbl, v in zip(labels_all, [passed, failed, other], strict=False) if v > 0]
    cols = [c for c, v in zip(colors_all, [passed, failed, other], strict=False) if v > 0]
    wedges, _ = ax.pie(values, colors=cols, startangle=90, wedgeprops=dict(width=0.38, edgecolor="white", linewidth=3))
    total = passed + failed + other
    ax.text(
        0, 0.08, f"{_fmt_pct(passed, total)}", ha="center", va="center", fontsize=30, fontweight="bold", color="#1E293B"
    )
    ax.text(0, -0.22, "bonnes réponses", ha="center", va="center", fontsize=12, color="#64748B")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, fontsize=11)
    ax.set(aspect="equal")
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out


def make_canal_bars(canal_stats: dict, out: Path) -> Path:
    """Dessine le taux de réussite par canal (barres horizontales) et l'enregistre en PNG."""
    names = list(canal_stats.keys())
    good = [canal_stats[n]["passed"] for n in names]
    bad = [canal_stats[n]["failed"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    y = range(len(names))
    ax.barh(list(y), good, color="#16A34A", label="Bonnes réponses")
    ax.barh(list(y), bad, left=good, color="#DC2626", label="À corriger")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=13)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Petit composant : diagramme du pipeline (dessiné, pas une image) ─────
class PipelineDiagram(Flowable):
    """Dessine les 4 étapes du pipeline reliées par des flèches."""

    def __init__(self, width, height=95):
        super().__init__()
        self.width = width
        self.height = height
        self.steps = [
            ("1", "Le Générateur", "invente des questions à poser au robot"),
            ("2", "L'Exécutant", "pose ces questions au robot, sur le site ou l'appli"),
            ("3", "L'Évaluateur", "vérifie si chaque réponse est bonne"),
            ("4", "Le Rédacteur", "résume tout dans un rapport (comme celui-ci)"),
        ]

    def draw(self):
        c = self.canv
        n = len(self.steps)
        gap = 14
        box_w = (self.width - gap * (n - 1)) / n
        box_h = 78
        for i, (num, title, desc) in enumerate(self.steps):
            x = i * (box_w + gap)
            y = self.height - box_h
            c.setFillColor(INDIGO_SOFT)
            c.roundRect(x, y, box_w, box_h, 8, stroke=0, fill=1)
            c.setFillColor(INDIGO)
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(x + box_w / 2, y + box_h - 22, num)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 9.5)
            c.drawCentredString(x + box_w / 2, y + box_h - 38, title)
            c.setFont("Helvetica", 7.6)
            c.setFillColor(GRAY)
            words = desc.split(" ")
            lines, cur = [], ""
            for w in words:
                test = (cur + " " + w).strip()
                if c.stringWidth(test, "Helvetica", 7.6) > box_w - 12:
                    lines.append(cur)
                    cur = w
                else:
                    cur = test
            lines.append(cur)
            ty = y + box_h - 52
            for line in lines[:3]:
                c.drawCentredString(x + box_w / 2, ty, line)
                ty -= 10
            if i < n - 1:
                ax = x + box_w + 2
                ay = y + box_h / 2
                c.setStrokeColor(GRAY)
                c.setLineWidth(1.3)
                c.line(ax, ay, ax + gap - 4, ay)
                c.line(ax + gap - 4, ay, ax + gap - 8, ay + 3)
                c.line(ax + gap - 4, ay, ax + gap - 8, ay - 3)


# ── Construction du document ──────────────────────────────────────────────
def build_pdf_report() -> Path:
    """Construit reports/rapport_comprehensible.pdf (langage clair, non technique) à partir des scores."""
    canaux = {}
    for name, (path, subtitle, explanation) in SCORE_FILES.items():
        data = _load(path)
        if data:
            canaux[name] = {"data": data, "subtitle": subtitle, "explanation": explanation}

    if not canaux:
        raise SystemExit("Aucun résultat trouvé dans reports/. Lance d'abord run_pipeline.py.")

    all_results = []
    canal_stats = {}
    for name, c in canaux.items():
        results = c["data"].get("scored_results", [])
        passed = sum(1 for r in results if r.get("verdict") == "PASS")
        failed = sum(1 for r in results if r.get("verdict") in ("FAIL", "WARN", "ERROR"))
        canal_stats[name] = {"passed": passed, "failed": failed, "total": len(results)}
        for r in results:
            r = dict(r)
            r["canal_group"] = name
            all_results.append(r)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("verdict") == "PASS")
    failed = total - passed
    hallucinations = [r for r in all_results if r.get("hallucination") is True]
    global_rate = _fmt_pct(passed, total)

    donut_path = make_donut(passed, failed, 0, TMP / "donut.png")
    bars_path = make_canal_bars(canal_stats, TMP / "bars.png")

    # ── Styles ─────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=36,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle("CoverSub", fontName="Helvetica", fontSize=13.5, leading=19, textColor=GRAY, spaceAfter=4)
    )
    styles.add(
        ParagraphStyle(
            "CoverBig", fontName="Helvetica-Bold", fontSize=64, leading=64, textColor=INDIGO, alignment=TA_CENTER
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverBigLbl", fontName="Helvetica", fontSize=13, leading=16, textColor=GRAY, alignment=TA_CENTER
        )
    )
    styles.add(
        ParagraphStyle(
            "H1", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK, spaceBefore=4, spaceAfter=10
        )
    )
    styles.add(
        ParagraphStyle(
            "H2",
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=INDIGO_DARK,
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=10.3, leading=15.5, textColor=INK, spaceAfter=8, alignment=TA_LEFT
        )
    )
    styles.add(ParagraphStyle("BodySmall", fontName="Helvetica", fontSize=9, leading=13, textColor=GRAY, spaceAfter=4))
    styles.add(
        ParagraphStyle("Caption", fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=INK, spaceAfter=2)
    )
    styles.add(ParagraphStyle("BoxTitle", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK))
    styles.add(ParagraphStyle("BoxBody", fontName="Helvetica", fontSize=9.6, leading=14, textColor=INK))
    styles.add(
        ParagraphStyle("Eyebrow", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=INDIGO, spaceAfter=2)
    )

    def info_box(title, body, bg=INDIGO_SOFT, border=INDIGO):
        t = Table(
            [[Paragraph(f"<b>{title}</b>", styles["BoxTitle"])], [Paragraph(body, styles["BoxBody"])]],
            colWidths=[160 * mm],
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BOX", (0, 0), (-1, -1), 0.75, border),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                    ("TOPPADDING", (0, 1), (-1, 1), 2),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
                ]
            )
        )
        return t

    # ── Document + gabarits de page ──────────────────────────────────────
    out_path = REPORTS / "rapport_comprehensible.pdf"
    doc = BaseDocTemplate(
        str(out_path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm
    )
    frame_full = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="full")

    def cover_bg(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(INDIGO)
        canvas.rect(0, A4[1] - 10 * mm, A4[0], 10 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(20 * mm, A4[1] - 7 * mm, "PROJET TESTING AGENTIQUE — TECHSHOP")
        canvas.restoreState()

    def page_bg(canvas, _doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.6)
        canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(20 * mm, 11 * mm, "Rapport de test automatique — Assistant TechShop")
        canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[frame_full], onPage=cover_bg),
            PageTemplate(id="normal", frames=[frame_full], onPage=page_bg),
        ]
    )

    story = []

    # ── PAGE DE GARDE ──────────────────────────────────────────────────
    story.append(Spacer(1, 34 * mm))
    story.append(Paragraph("Rapport de test", styles["CoverTitle"]))
    story.append(Paragraph("Assistant conversationnel TechShop", styles["CoverTitle"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Ce document explique, en langage simple, comment nous avons vérifié automatiquement "
            "que l'assistant virtuel du site TechShop répond correctement à ses clients.",
            styles["CoverSub"],
        )
    )
    story.append(Spacer(1, 30))
    story.append(Paragraph(global_rate, styles["CoverBig"]))
    story.append(Paragraph(f"des {total} questions testées ont reçu une bonne réponse", styles["CoverBigLbl"]))
    story.append(Spacer(1, 40))
    story.append(HRFlowable(width="100%", color=LINE, thickness=0.7))
    story.append(Spacer(1, 10))
    meta = [
        ["Date du rapport", datetime.now().strftime("%d/%m/%Y")],
        ["Nombre de questions testées", str(total)],
        ["Façons de tester utilisées", ", ".join(canaux.keys())],
    ]
    meta_table = Table(
        [[Paragraph(f"<b>{k}</b>", styles["BodySmall"]), Paragraph(str(v), styles["BodySmall"])] for k, v in meta],
        colWidths=[60 * mm, 100 * mm],
    )
    meta_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(meta_table)
    story.append(NextPageTemplate("normal"))
    story.append(PageBreak())

    # ── QU'EST-CE QUE CE DOCUMENT ─────────────────────────────────────
    story.append(Paragraph("Qu'est-ce que ce document ?", styles["H1"]))
    story.append(
        Paragraph(
            "TechShop utilise un assistant virtuel (un \u00abrobot\u00bb basé sur l'intelligence artificielle) pour "
            "répondre aux questions des clients : prix, disponibilité des produits, suivi de commande, etc. "
            "Avant de mettre ce robot en service, il faut vérifier qu'il donne de bonnes réponses, tout le temps, "
            "même face à des questions difficiles, mal formulées ou pièges.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Plutôt que de tester le robot \u00abà la main\u00bb (ce qui prendrait des heures et devrait être refait "
            "à chaque changement), nous avons construit un <b>système de test automatique</b> : quatre programmes "
            "informatiques travaillent ensemble, chacun avec un rôle précis, pour poser des questions au robot, "
            "vérifier ses réponses, et produire un rapport — comme celui-ci.",
            styles["Body"],
        )
    )
    story.append(
        info_box(
            "En une phrase",
            f"Sur {total} questions posées au robot TechShop, <b>{passed} ont reçu une bonne réponse</b> "
            f"et {failed} ont révélé un problème à corriger.",
        )
    )

    # ── COMMENT ÇA MARCHE ─────────────────────────────────────────────
    story.append(Paragraph("Comment fonctionne le système de test ?", styles["H1"]))
    story.append(
        Paragraph("Quatre \u00abrobots testeurs\u00bb se relaient, chacun spécialisé dans une tâche :", styles["Body"])
    )
    story.append(Spacer(1, 6))
    story.append(PipelineDiagram(doc.width))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Ce processus peut être rejoué à tout moment — par exemple après une modification du robot — "
            "pour vérifier qu'aucune réponse ne s'est dégradée entre-temps.",
            styles["BodySmall"],
        )
    )

    story.append(PageBreak())

    # ── RÉSULTATS GLOBAUX ──────────────────────────────────────────────
    story.append(Paragraph("Résultats globaux", styles["H1"]))
    story.append(
        Paragraph(
            f"Nous avons posé {total} questions au robot TechShop, en couvrant des situations normales "
            f"(\u00abBonjour\u00bb, \u00abQuel est le prix de ce produit ?\u00bb), des cas limites (questions vides, très "
            f"longues, hors sujet) et des tentatives de piège volontaires (questions en plusieurs langues, "
            f"ironie, tentatives de manipulation du robot).",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 6))
    img_donut = Image(str(donut_path), width=85 * mm, height=85 * mm)
    stat_cards = Table(
        [
            [
                Paragraph(
                    "<b>%d</b>" % total, ParagraphStyle("n1", parent=styles["CoverBig"], fontSize=26, textColor=INK)
                ),
                Paragraph(
                    "<b>%d</b>" % passed, ParagraphStyle("n2", parent=styles["CoverBig"], fontSize=26, textColor=GREEN)
                ),
                Paragraph(
                    "<b>%d</b>" % failed, ParagraphStyle("n3", parent=styles["CoverBig"], fontSize=26, textColor=RED)
                ),
            ],
            [
                Paragraph("questions posées", styles["BodySmall"]),
                Paragraph("bonnes réponses", styles["BodySmall"]),
                Paragraph("à corriger", styles["BodySmall"]),
            ],
        ],
        colWidths=[35 * mm, 35 * mm, 35 * mm],
    )
    stat_cards.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    row = Table([[img_donut, stat_cards]], colWidths=[90 * mm, 80 * mm])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(row)

    # ── PAR CANAL ────────────────────────────────────────────────────
    story.append(Paragraph("Résultats selon la façon de tester", styles["H1"]))
    story.append(
        Paragraph(
            "Le robot a été testé de plusieurs façons différentes, pour s'assurer qu'il fonctionne bien "
            "partout où les clients peuvent le rencontrer :",
            styles["Body"],
        )
    )
    for name, c in canaux.items():
        s = canal_stats[name]
        story.append(Paragraph(f"{name} — {c['subtitle']}", styles["H2"]))
        story.append(Paragraph(c["explanation"], styles["Body"]))
        story.append(
            Paragraph(
                f"<b>{s['passed']} bonnes réponses sur {s['total']}</b> ({_fmt_pct(s['passed'], s['total'])}).",
                styles["Body"],
            )
        )
    story.append(Spacer(1, 6))
    story.append(Image(str(bars_path), width=150 * mm, height=70 * mm))

    story.append(PageBreak())

    # ── EXEMPLES CONCRETS AVEC VRAIES CAPTURES ─────────────────────────
    story.append(Paragraph("Exemples concrets", styles["H1"]))
    story.append(
        Paragraph(
            "Voici deux vraies captures d'écran prises automatiquement pendant les tests, pour illustrer "
            "concrètement ce qui a été vérifié.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 8))

    web_results = canaux.get("Web", {}).get("data", {}).get("scored_results", [])
    good_example = next((r for r in web_results if r.get("verdict") == "PASS"), None)
    bad_example = next((r for r in web_results if r.get("verdict") != "PASS"), None)
    for r, label, color_bg, color_border in [
        (good_example, "Exemple réussi", GREEN_SOFT, GREEN),
        (bad_example, "Exemple à corriger", RED_SOFT, RED),
    ]:
        if not r:
            continue
        shot = SCREENSHOTS / f"{r['id']}.png"
        if not shot.exists():
            continue
        story.append(Paragraph(f"{label} — {r.get('name', r['id'])}", styles["H2"]))
        cap_table = Table(
            [
                [
                    Image(str(shot), width=70 * mm, height=52 * mm),
                    Paragraph(
                        f"<b>Question posée :</b> \u00ab{r.get('input', '')}\u00bb<br/><br/>"
                        f"<b>Ce qui s'est passé :</b> {_plain_reason(r.get('reason', ''), r.get('verdict'))}",
                        styles["BoxBody"],
                    ),
                ]
            ],
            colWidths=[75 * mm, 85 * mm],
        )
        cap_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 1, color_border),
                    ("BACKGROUND", (0, 0), (-1, -1), color_bg),
                    ("LEFTPADDING", (0, 0), (0, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (1, 0), (1, 0), 10),
                ]
            )
        )
        story.append(cap_table)
        story.append(Spacer(1, 10))

    # ── HALLUCINATIONS EXPLIQUÉES ───────────────────────────────────────
    story.append(Paragraph("Qu'est-ce qu'une \u00abhallucination\u00bb ?", styles["H1"]))
    story.append(
        info_box(
            "En langage simple",
            "Une <b>hallucination</b>, c'est quand le robot invente une information qui n'est pas vraie — "
            "par exemple annoncer un prix erroné, dire qu'un produit est disponible alors qu'il ne l'est pas, "
            "ou citer une politique de retour qui n'existe pas. C'est le type d'erreur le plus grave, car un "
            "client pourrait faire confiance à une information fausse. C'est pour cela que, dans notre système, "
            "<b>une seule hallucination suffit à faire échouer un test</b>, même si le reste de la réponse était "
            "bien formulé.",
            bg=AMBER_SOFT if hallucinations else GREEN_SOFT,
            border=AMBER if hallucinations else GREEN,
        )
    )
    if hallucinations:
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                f"<b>{len(hallucinations)} hallucination(s) détectée(s)</b> lors de cette série de tests — "
                f"voir le détail technique dans reports/test_report.md.",
                styles["Body"],
            )
        )
    else:
        story.append(Spacer(1, 6))
        story.append(
            Paragraph("Aucune hallucination n'a été détectée lors de cette série de tests. ✅", styles["Body"])
        )

    story.append(PageBreak())

    # ── RECOMMANDATIONS ─────────────────────────────────────────────────
    story.append(Paragraph("Et maintenant ?", styles["H1"]))
    recos = []
    if failed / total > 0.3 if total else False:
        recos.append(
            "Une part importante des réponses est encore à corriger : nous recommandons de revoir "
            "les cas signalés avant une mise en service auprès de vrais clients."
        )
    if hallucinations:
        recos.append(
            "Les hallucinations détectées doivent être corrigées en priorité : ce sont les erreurs "
            "les plus risquées pour la confiance des clients."
        )
    recos.append(
        "Ce système de test peut être rejoué automatiquement après chaque modification du robot, "
        "pour s'assurer qu'aucune réponse qui fonctionnait auparavant ne se dégrade (on appelle "
        "cela une \u00abrégression\u00bb)."
    )
    recos.append(
        "Un tableau de bord interactif (reports/dashboard.html) permet d'explorer tous les résultats "
        "en détail, avec les captures d'écran de chaque test."
    )
    for r in recos:
        story.append(Paragraph(f"•  {r}", styles["Body"]))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=LINE, thickness=0.7))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Ce rapport a été généré automatiquement par le système de test agentique TechShop "
            "(pipeline Generator -&gt; Executor -&gt; Evaluator -&gt; Reporter). Pour le d\u00e9tail technique complet, "
            "voir reports/test_report.md et reports/dashboard.html.",
            styles["BodySmall"],
        )
    )

    doc.build(story)
    return out_path


if __name__ == "__main__":
    path = build_pdf_report()
    print(f"Rapport PDF généré : {path}")
