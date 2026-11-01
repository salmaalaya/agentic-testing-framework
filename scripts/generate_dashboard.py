"""
scripts/generate_dashboard.py — Dashboard HTML (Plotly + design sur-mesure)

Génère reports/dashboard.html : un fichier unique, autonome (les captures
d'écran réelles sont encodées en base64 à l'intérieur), qui présente :
  - le pipeline des 4 agents Swarm (Generator, Executor, Evaluator, Reporter)
    avec leurs statistiques propres
  - les KPI globaux et les graphiques Plotly (verdicts, scores, latence)
  - une galerie des VRAIES captures Playwright/Appium avec zoom au clic
  - un tableau détaillé filtrable par canal

Usage :
    python scripts/generate_dashboard.py
    python run_pipeline.py --dashboard   # généré automatiquement en fin de pipeline
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
SCREENSHOTS = ROOT / "screenshots"
CORPUS = ROOT / "corpus"
LITE_MAX_SCREENSHOTS = 4

SCORE_FILES = {
    "API": REPORTS / "scored_results.json",
    "Web": REPORTS / "web_scored_results.json",
    "Mobile": REPORTS / "mobile_scored_results.json",
}
EXEC_FILES = {
    "API": REPORTS / "execution_results.json",
    "Web": REPORTS / "web_execution_results.json",
    "Mobile": REPORTS / "mobile_execution_results.json",
}
CORPUS_FILES = {
    "API": CORPUS / "test_cases.json",
    "Web": CORPUS / "web_test_cases.json",
    "Mobile": CORPUS / "mobile_test_cases.json",
}

VERDICT_COLORS = {"PASS": "#34D399", "FAIL": "#F87171", "WARN": "#FBBF24", "ERROR": "#94A3B8"}
VERDICT_LABELS = {"PASS": "Réussi", "FAIL": "Échoué", "WARN": "Alerte", "ERROR": "Erreur technique"}
CANAL_LABELS = {"api": "API (backend)", "web": "Interface Web", "mobile": "Application Mobile"}
CATEGORY_LABELS = {"nominal": "Cas normal", "edge_case": "Cas limite", "adversarial": "Test de robustesse"}

FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&'
    'family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" '
    'rel="stylesheet">'
)


# ── Helpers ──────────────────────────────────────────────────────────────
def _load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _b64_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _esc(s) -> str:
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _badge_class(verdict: str) -> str:
    """Classe CSS ascii-safe pour un verdict (pas d'accents dans les noms de classe)."""
    return {"PASS": "pass", "FAIL": "fail", "WARN": "warn", "ERROR": "error"}.get(verdict, "error")


# ── Chargement des données et construction ───────────────────────────────
def build_dashboard(out_path: Path | None = None, lite: bool = False) -> Path:
    """
    Construit le dashboard HTML à partir des fichiers *_scored_results.json.

    out_path : fichier de sortie (défaut : reports/dashboard.html)
    lite     : Plotly.js chargé via CDN au lieu d'être embarqué, et seulement
               LITE_MAX_SCREENSHOTS captures — fichier ~10x plus léger, utilisé
               pour l'exemple versionné dans docs/sample-output/.
    """
    canaux = {name: _load(path) for name, path in SCORE_FILES.items()}
    canaux = {name: data for name, data in canaux.items() if data}
    if not canaux:
        raise SystemExit(
            "Aucun fichier de scores trouvé dans reports/. "
            "Lance d'abord le pipeline (run_pipeline.py) pour générer des résultats."
        )

    all_results = []
    for canal_name, data in canaux.items():
        for r in data.get("scored_results", []):
            r = dict(r)
            r["canal_group"] = canal_name
            all_results.append(r)
    by_id = {r["id"]: r for r in all_results}

    total = len(all_results)
    verdict_counts_global = {"PASS": 0, "FAIL": 0, "WARN": 0, "ERROR": 0}
    for r in all_results:
        v = r.get("verdict", "ERROR")
        verdict_counts_global[v] = verdict_counts_global.get(v, 0) + 1
    hallucinations = [r for r in all_results if r.get("hallucination") is True]
    global_rate = round(verdict_counts_global["PASS"] / total * 100, 1) if total else 0

    # Stats Generator
    generated = _load(CORPUS / "generated_test_cases.json")
    generated_count = len(generated.get("test_cases", [])) if generated else 0

    # Stats Executor (par canal)
    executor_stats = {}
    for canal_name in canaux:
        exec_data = _load(EXEC_FILES.get(canal_name, Path("__missing__")))
        corpus_data = _load(CORPUS_FILES.get(canal_name, Path("__missing__")))
        corpus_total = len(corpus_data.get("test_cases", [])) if corpus_data else None
        if exec_data:
            results = exec_data.get("results", [])
            errors = sum(1 for r in results if r.get("error") or r.get("status_code", 200) >= 500)
            executed = len(results)
        else:
            executed = len(canaux[canal_name].get("scored_results", []))
            errors = 0
        executor_stats[canal_name] = {
            "executed": executed,
            "corpus_total": corpus_total or executed,
            "errors": errors,
        }

    # Stats Evaluator
    dims = ["pertinence", "exactitude", "coherence"]
    evaluator_avgs = {}
    for d in dims:
        vals = [r.get(d, 0) for r in all_results]
        evaluator_avgs[d] = round(sum(vals) / len(vals), 2) if vals else 0

    # Stats Reporter
    report_md = REPORTS / "test_report.md"
    report_generated_at = (
        datetime.fromtimestamp(report_md.stat().st_mtime).strftime("%d/%m/%Y à %H:%M") if report_md.exists() else "—"
    )

    # Captures d'écran réelles
    gallery_items = []
    if SCREENSHOTS.exists():
        shots = sorted(SCREENSHOTS.glob("*.png"))
        if lite:
            shots = [p for p in shots if "_ERROR" not in p.stem.upper()][:LITE_MAX_SCREENSHOTS]
        for img_path in shots:
            m = re.match(r"^([A-Za-z]+-\d+)", img_path.stem)
            test_id = m.group(1) if m else img_path.stem
            is_error_shot = "_ERROR" in img_path.stem.upper()
            meta = by_id.get(test_id, {})
            verdict = meta.get("verdict") or ("ERROR" if is_error_shot else None)
            gallery_items.append(
                {
                    "id": test_id,
                    "data_uri": _b64_image(img_path),
                    "name": meta.get("name", img_path.stem),
                    "canal": meta.get("canal", "web" if test_id.upper().startswith("WEB") else "mobile"),
                    "verdict": verdict,
                }
            )

    # Graphiques Plotly (thème sombre sur-mesure)
    canal_names = list(canaux.keys())
    plot_font = dict(family="Inter, sans-serif", color="#C7D0E0", size=13)
    common_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=plot_font, margin=dict(t=48, l=40, r=24, b=40)
    )

    fig_stack = go.Figure()
    for verdict, color in VERDICT_COLORS.items():
        counts = [
            sum(1 for r in canaux[c].get("scored_results", []) if r.get("verdict") == verdict) for c in canal_names
        ]
        fig_stack.add_trace(go.Bar(name=VERDICT_LABELS[verdict], x=canal_names, y=counts, marker_color=color))
    fig_stack.update_layout(barmode="stack", title="Verdicts par canal", height=360, **common_layout)
    fig_stack.update_xaxes(gridcolor="#232B3F")
    fig_stack.update_yaxes(gridcolor="#232B3F")

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=[VERDICT_LABELS[k] for k in verdict_counts_global],
                values=list(verdict_counts_global.values()),
                marker_colors=[VERDICT_COLORS[k] for k in verdict_counts_global],
                hole=0.55,
                textfont=dict(color="#0B0F1A", size=13),
            )
        ]
    )
    fig_pie.update_layout(title="Répartition globale", height=360, **common_layout)

    fig_scores = go.Figure()
    for cname in canal_names:
        results = canaux[cname].get("scored_results", [])
        n = len(results) or 1
        avgs = [round(sum(r.get(d, 0) for r in results) / n, 2) for d in dims]
        fig_scores.add_trace(go.Bar(name=cname, x=["Pertinence", "Exactitude", "Cohérence"], y=avgs))
    fig_scores.update_layout(
        barmode="group", title="Scores moyens (sur 5)", yaxis_range=[0, 5], height=360, **common_layout
    )
    fig_scores.update_xaxes(gridcolor="#232B3F")
    fig_scores.update_yaxes(gridcolor="#232B3F")

    fig_latency = go.Figure()
    for cname in canal_names:
        results = canaux[cname].get("scored_results", [])
        fig_latency.add_trace(
            go.Bar(name=cname, x=[r.get("id", "?") for r in results], y=[r.get("latency_ms", 0) for r in results])
        )
    fig_latency.update_layout(title="Latence par cas de test (ms)", height=420, xaxis_tickangle=-45, **common_layout)
    fig_latency.update_xaxes(gridcolor="#1A2136")
    fig_latency.update_yaxes(gridcolor="#232B3F")

    fig_gauges = make_subplots(rows=1, cols=len(canal_names), specs=[[{"type": "indicator"}] * len(canal_names)])
    for i, cname in enumerate(canal_names, start=1):
        rate = canaux[cname].get("summary", {}).get("success_rate_pct", 0)
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=rate,
                title={"text": cname, "font": {"color": "#C7D0E0"}},
                number={"font": {"color": "#E7ECF5"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8B96AD"},
                    "bar": {"color": "#6D5DFB"},
                    "bgcolor": "#131A2B",
                    "bordercolor": "#232B3F",
                    "steps": [
                        {"range": [0, 50], "color": "#2A1620"},
                        {"range": [50, 80], "color": "#2A2410"},
                        {"range": [80, 100], "color": "#12281F"},
                    ],
                },
            ),
            row=1,
            col=i,
        )
    fig_gauges.update_layout(title="Taux de réussite par canal (%)", height=280, **common_layout)

    # Plotly.js est embarqué en dur dans le 1er graphique (True) plutôt que via CDN :
    # le dashboard reste 100% autonome et s'ouvre même sans connexion internet.
    figs = [fig_gauges, fig_stack, fig_pie, fig_scores, fig_latency]
    gauges_html, stack_html, pie_html, scores_html, latency_html = [
        fig.to_html(
            full_html=False,
            include_plotlyjs=(("cdn" if lite else True) if i == 0 else False),
            config={"displaylogo": False},
        )
        for i, fig in enumerate(figs)
    ]

    html = _render_html(
        canal_names=canal_names,
        total=total,
        verdict_counts_global=verdict_counts_global,
        hallucinations=hallucinations,
        global_rate=global_rate,
        generated_count=generated_count,
        executor_stats=executor_stats,
        evaluator_avgs=evaluator_avgs,
        report_generated_at=report_generated_at,
        gallery_items=gallery_items,
        all_results=all_results,
        gauges_html=gauges_html,
        stack_html=stack_html,
        pie_html=pie_html,
        scores_html=scores_html,
        latency_html=latency_html,
    )

    out_path = Path(out_path) if out_path else REPORTS / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ── Rendu HTML ────────────────────────────────────────────────────────────
def _render_html(
    *,
    canal_names,
    total,
    verdict_counts_global,
    hallucinations,
    global_rate,
    generated_count,
    executor_stats,
    evaluator_avgs,
    report_generated_at,
    gallery_items,
    all_results,
    gauges_html,
    stack_html,
    pie_html,
    scores_html,
    latency_html,
) -> str:

    now = datetime.now().strftime("%d/%m/%Y à %H:%M")

    generator_note = (
        f"{generated_count} cas générés automatiquement"
        if generated_count
        else "Étape ignorée (--skip-generator) — corpus manuel utilisé"
    )
    executor_rows = "".join(
        f"""<div class="agent-canal-row">
              <span class="dot {"ok" if s["errors"] == 0 else "bad"}"></span>
              <span class="agent-canal-name">{_esc(c)}</span>
              <span class="agent-canal-stat">{s["executed"]}/{s["corpus_total"]} exécutés{f", {s['errors']} erreur(s)" if s["errors"] else ""}</span>
            </div>"""
        for c, s in executor_stats.items()
    )

    agent_cards = f"""
    <div class="pipeline">
      <div class="pipeline-node">
        <div class="node-card">
          <div class="node-icon">&#10022;</div>
          <div class="node-label">Generator</div>
          <div class="node-model">mistral</div>
          <p class="node-desc">{_esc(generator_note)}</p>
        </div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-node">
        <div class="node-card">
          <div class="node-icon">&#9638;</div>
          <div class="node-label">Executor</div>
          <div class="node-model">mistral</div>
          {executor_rows}
        </div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-node">
        <div class="node-card">
          <div class="node-icon">&#9678;</div>
          <div class="node-label">Evaluator</div>
          <div class="node-model">llama3.2</div>
          <p class="node-desc">Pertinence {evaluator_avgs["pertinence"]}/5 &middot; Exactitude {evaluator_avgs["exactitude"]}/5 &middot; Coh&eacute;rence {evaluator_avgs["coherence"]}/5</p>
          <p class="node-desc {"halluc" if hallucinations else ""}">{len(hallucinations)} hallucination(s) d&eacute;tect&eacute;e(s)</p>
        </div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-node">
        <div class="node-card node-active">
          <div class="node-icon">&#9639;</div>
          <div class="node-label">Reporter</div>
          <div class="node-model">mistral</div>
          <p class="node-desc">Rapport g&eacute;n&eacute;r&eacute; le {report_generated_at}</p>
        </div>
      </div>
    </div>
    """

    if gallery_items:
        cards = []
        for item in gallery_items:
            v = item["verdict"]
            badge_cls = _badge_class(v) if v else "error"
            badge_label = VERDICT_LABELS.get(v, "Capture") if v else "Capture"
            cards.append(f'''
          <figure class="shot" data-canal="{_esc(item["canal"])}" onclick="openLightbox('{item["data_uri"]}','{_esc(item["id"])} \u2014 {_esc(item["name"])}')">
            <img src="{item["data_uri"]}" loading="lazy" alt="{_esc(item["id"])}">
            <figcaption>
              <span class="shot-id">{_esc(item["id"])}</span>
              <span class="badge {badge_cls}">{_esc(badge_label)}</span>
            </figcaption>
          </figure>''')
        gallery_section = f"""
        <section class="section">
          <h2><span class="eyebrow">03</span> Captures d'&eacute;cran r&eacute;elles</h2>
          <p class="section-sub">Chaque capture provient d'une ex&eacute;cution Playwright r&eacute;elle sur l'interface web de TechShop &mdash; clique pour zoomer.</p>
          <div class="gallery">{"".join(cards)}</div>
        </section>"""
        detail_eyebrow = "04"
    else:
        gallery_section = ""
        detail_eyebrow = "03"

    def _nlp_cell(v):
        return f"{v}" if v is not None else '<span class="dim">—</span>'

    table_rows = "".join(
        f'''
      <tr data-canal="{_esc(r.get("canal", ""))}">
        <td class="mono">{_esc(r.get("id"))}</td>
        <td>{_esc(r.get("name"))}</td>
        <td>{_esc(CANAL_LABELS.get(r.get("canal"), r.get("canal")))}</td>
        <td>{_esc(CATEGORY_LABELS.get(r.get("category"), r.get("category")))}</td>
        <td><span class="badge {_badge_class(r.get("verdict"))}">{_esc(VERDICT_LABELS.get(r.get("verdict"), r.get("verdict")))}</span></td>
        <td class="mono">{r.get("score_moyen", 0)}/5</td>
        <td class="mono">{int(r.get("latency_ms", 0))} ms</td>
        <td class="mono">{_nlp_cell(r.get("bleu_score"))}</td>
        <td class="mono">{_nlp_cell(r.get("rouge_l_score"))}</td>
        <td class="mono">{_nlp_cell(r.get("nli_entailment_score"))}</td>
        <td class="reason">{_esc(r.get("reason", ""))}</td>
      </tr>'''
        for r in all_results
    )

    filter_buttons = "".join(
        f'<button class="filter-btn" data-filter="{c}" onclick="filterTable(\'{c}\')">{c}</button>' for c in canal_names
    )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard &middot; Testing Agentique TechShop</title>
{FONT_LINKS}
<style>
:root {{
  --bg: #0B0F1A; --surface: #10182A; --surface-2: #16203A; --border: #212B45;
  --text: #E7ECF5; --text-dim: #8B96AD; --accent: #6D5DFB; --accent-soft: #241D5C;
  --pass: #34D399; --fail: #F87171; --warn: #FBBF24; --error: #94A3B8; --halluc: #FB7185;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: radial-gradient(circle at 15% 0%, #16204A 0%, var(--bg) 45%) var(--bg);
  color: var(--text); font-family: 'Inter', sans-serif; line-height: 1.5;
}}
.mono {{ font-family: 'JetBrains Mono', monospace; }}
h1, h2, h3, .node-label {{ font-family: 'Space Grotesk', sans-serif; }}

.hero {{ padding: 56px 48px 40px; border-bottom: 1px solid var(--border); }}
.hero-top {{ display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:24px; }}
.hero-tag {{ display:inline-block; padding:4px 12px; border-radius:999px; background:var(--accent-soft); color:#B4A9FF;
  font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:600; margin-bottom:14px; }}
.hero h1 {{ font-size: 2.1rem; margin: 0 0 6px 0; font-weight:700; letter-spacing:-0.01em; }}
.hero p.sub {{ color: var(--text-dim); margin:0; font-size:0.95rem; }}
.hero-kpi {{ text-align:right; }}
.hero-kpi .big {{ font-size:3rem; font-weight:700; font-family:'Space Grotesk',sans-serif;
  background:linear-gradient(90deg,#8F7CFF,#22D3EE); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.hero-kpi .lbl {{ color:var(--text-dim); font-size:0.85rem; }}

.pipeline {{ display:flex; align-items:stretch; gap:6px; margin-top:36px; flex-wrap:wrap; }}
.pipeline-node {{ flex:1; min-width:190px; }}
.pipeline-arrow {{ display:flex; align-items:center; color: var(--text-dim); font-size:1.4rem; padding: 0 2px; }}
.node-card {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:18px; height:100%; }}
.node-card.node-active {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), 0 0 24px -4px rgba(109,93,251,.5); }}
.node-icon {{ font-size:1.3rem; color:var(--accent); margin-bottom:6px; }}
.node-label {{ font-weight:600; font-size:1rem; }}
.node-model {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-dim); margin-bottom:10px; }}
.node-desc {{ font-size:0.82rem; color:var(--text-dim); margin:4px 0 0 0; }}
.node-desc.halluc {{ color: var(--halluc); font-weight:600; }}
.agent-canal-row {{ display:flex; align-items:center; gap:8px; font-size:0.8rem; margin-top:8px; }}
.agent-canal-name {{ color:var(--text); min-width:52px; }}
.agent-canal-stat {{ color:var(--text-dim); }}
.dot {{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }}
.dot.ok {{ background: var(--pass); box-shadow: 0 0 6px var(--pass); }}
.dot.bad {{ background: var(--fail); box-shadow: 0 0 6px var(--fail); }}

.section {{ padding: 40px 48px; border-bottom:1px solid var(--border); }}
.section h2 {{ font-size:1.3rem; margin:0 0 4px 0; display:flex; align-items:center; gap:12px; }}
.eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--accent); border:1px solid var(--accent-soft);
  padding:2px 8px; border-radius:6px; }}
.section-sub {{ color:var(--text-dim); font-size:0.88rem; margin:6px 0 24px 0; }}

.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:14px; }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:18px 20px; }}
.kpi .val {{ font-size:1.9rem; font-weight:700; font-family:'Space Grotesk',sans-serif; }}
.kpi .lbl {{ color:var(--text-dim); font-size:0.8rem; margin-top:2px; }}
.pass-c {{ color: var(--pass); }} .fail-c {{ color: var(--fail); }} .warn-c {{ color: var(--warn); }} .halluc-c {{ color: var(--halluc); }}

.charts-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:8px; }}
.chart-card {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:6px; }}
.full {{ grid-column: 1 / -1; }}

.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:16px; }}
.shot {{ margin:0; background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden;
  cursor:zoom-in; transition: transform .15s ease, border-color .15s ease; }}
.shot:hover {{ transform: translateY(-3px); border-color: var(--accent); }}
.shot img {{ width:100%; display:block; aspect-ratio: 16/11; object-fit:cover; object-position: top; background:#fff; }}
.shot figcaption {{ display:flex; justify-content:space-between; align-items:center; padding:10px 12px; }}
.shot-id {{ font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--text-dim); }}

.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.74rem; font-weight:600; white-space:nowrap; }}
.badge.pass {{ background:#0F2E22; color:var(--pass); }}
.badge.fail {{ background:#341717; color:var(--fail); }}
.badge.warn {{ background:#332B0C; color:var(--warn); }}
.badge.error {{ background:#1E2434; color:var(--error); }}

.table-wrap {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
th {{ text-align:left; padding:12px 14px; color:var(--text-dim); font-weight:600; font-size:0.75rem;
  text-transform:uppercase; letter-spacing:0.03em; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--surface); }}
td {{ padding:11px 14px; border-bottom:1px solid var(--border); vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
td.reason {{ color:var(--text-dim); max-width:280px; }}
.dim {{ color:var(--text-dim); }}
.filters {{ margin-bottom:16px; display:flex; gap:8px; }}
.filter-btn {{ background:var(--surface-2); border:1px solid var(--border); color:var(--text); padding:6px 14px;
  border-radius:999px; font-size:0.8rem; cursor:pointer; font-family:'Inter',sans-serif; }}
.filter-btn.active {{ background:var(--accent); border-color:var(--accent); color:white; }}

#lightbox {{ display:none; position:fixed; inset:0; background:rgba(4,6,14,.92); z-index:999; align-items:center;
  justify-content:center; flex-direction:column; padding:32px; cursor:zoom-out; }}
#lightbox img {{ max-width:min(900px,90vw); max-height:78vh; border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,.6); }}
#lightbox .cap {{ color:var(--text); margin-top:16px; font-family:'JetBrains Mono',monospace; font-size:0.85rem; }}

footer {{ padding:28px 48px; color:var(--text-dim); font-size:0.8rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;}}

@media (max-width: 900px) {{
  .hero {{ padding:36px 20px; }} .section {{ padding:28px 20px; }}
  .charts-grid {{ grid-template-columns:1fr; }} .hero-kpi {{ text-align:left; }}
  .hero-top {{ flex-direction:column; }}
}}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-top">
    <div>
      <span class="hero-tag">Pipeline Swarm &middot; Ollama local</span>
      <h1>Testing Agentique &mdash; TechShop</h1>
      <p class="sub">G&eacute;n&eacute;r&eacute; le {now} &middot; {total} sc&eacute;narios ex&eacute;cut&eacute;s sur {len(canal_names)} canal(aux) : {", ".join(canal_names)}</p>
    </div>
    <div class="hero-kpi">
      <div class="big">{global_rate}%</div>
      <div class="lbl">taux de r&eacute;ussite global</div>
    </div>
  </div>
  {agent_cards}
</header>

<section class="section">
  <h2><span class="eyebrow">01</span> Vue d'ensemble</h2>
  <p class="section-sub">R&eacute;sultats agr&eacute;g&eacute;s sur l'ensemble des canaux test&eacute;s.</p>
  <div class="kpis">
    <div class="kpi"><div class="val">{total}</div><div class="lbl">Tests totaux</div></div>
    <div class="kpi"><div class="val pass-c">{verdict_counts_global["PASS"]}</div><div class="lbl">R&eacute;ussis</div></div>
    <div class="kpi"><div class="val fail-c">{verdict_counts_global["FAIL"]}</div><div class="lbl">&Eacute;chou&eacute;s</div></div>
    <div class="kpi"><div class="val warn-c">{verdict_counts_global["WARN"] + verdict_counts_global["ERROR"]}</div><div class="lbl">Alertes / erreurs</div></div>
    <div class="kpi"><div class="val halluc-c">{len(hallucinations)}</div><div class="lbl">Hallucinations</div></div>
  </div>
</section>

<section class="section">
  <h2><span class="eyebrow">02</span> Graphiques</h2>
  <p class="section-sub">D&eacute;tail visuel par canal, par dimension de score et par latence.</p>
  <div class="charts-grid">
    <div class="chart-card full">{gauges_html}</div>
    <div class="chart-card">{stack_html}</div>
    <div class="chart-card">{pie_html}</div>
    <div class="chart-card">{scores_html}</div>
    <div class="chart-card full">{latency_html}</div>
  </div>
</section>

{gallery_section}

<section class="section">
  <h2><span class="eyebrow">{detail_eyebrow}</span> D&eacute;tail des cas de test</h2>
  <p class="section-sub">Filtre par canal pour explorer chaque sc&eacute;nario individuellement. Les colonnes BLEU / ROUGE-L / NLI sont informatives (n'affectent pas le verdict) &mdash; &laquo;&nbsp;&mdash;&nbsp;&raquo; si non disponibles pour ce cas.</p>
  <div class="filters">
    <button class="filter-btn active" data-filter="all" onclick="filterTable('all')">Tous</button>
    {filter_buttons}
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>ID</th><th>Nom</th><th>Canal</th><th>Cat&eacute;gorie</th><th>Verdict</th><th>Score</th><th>Latence</th><th>BLEU</th><th>ROUGE-L</th><th>NLI</th><th>D&eacute;tail</th></tr></thead>
      <tbody id="results-body">{table_rows}</tbody>
    </table>
  </div>
</section>

<footer>
  <span>Testing Agentique TechShop &mdash; pipeline Generator &rarr; Executor &rarr; Evaluator &rarr; Reporter (Swarm + Ollama)</span>
  <span>Voir aussi : reports/test_report.md &middot; reports/RAPPORT_COMPLET_PROJET.md</span>
</footer>

<div id="lightbox" onclick="closeLightbox()">
  <img id="lightbox-img" src="" alt="">
  <div class="cap" id="lightbox-cap"></div>
</div>

<script>
function openLightbox(src, caption) {{
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox-cap').textContent = caption;
  document.getElementById('lightbox').style.display = 'flex';
}}
function closeLightbox() {{ document.getElementById('lightbox').style.display = 'none'; }}
function filterTable(canal) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b.dataset.filter === canal));
  document.querySelectorAll('#results-body tr').forEach(tr => {{
    tr.style.display = (canal === 'all' || tr.dataset.canal === canal) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Génère le dashboard HTML (Plotly) des résultats.")
    parser.add_argument("--out", type=Path, default=None, help="Fichier de sortie (défaut : reports/dashboard.html)")
    parser.add_argument(
        "--lite", action="store_true", help="Plotly via CDN + quelques captures seulement (fichier léger)"
    )
    args = parser.parse_args()
    path = build_dashboard(out_path=args.out, lite=args.lite)
    print(f"Dashboard généré : {path}")
