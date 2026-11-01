# Sample output

Files produced by one real run of the pipeline on 2026-11-01
(`--canal api` and `--canal web`, curated corpus, `mistral` + `llama3.2` on CPU).
Reports are written in French, like the rest of the pipeline output.

| File | Produced by | Content |
|---|---|---|
| [`test_report.md`](test_report.md) | Reporter agent | Per-case table (latency, scores, hallucination flag, verdict, reason), BLEU/ROUGE-L, per-channel summaries |
| [`final_summary.json`](final_summary.json) | Reporter agent | Machine-readable summary of the API channel |
| [`dashboard.html`](dashboard.html) | `scripts/generate_dashboard.py --lite` | Interactive Plotly dashboard (API + Web). The lite build loads Plotly from a CDN and embeds 4 screenshots; the default build is fully offline (~15 MB) |
| [`rapport_comprehensible.pdf`](rapport_comprehensible.pdf) | `scripts/generate_pdf_report.py` | Plain-language report for non-technical readers |

GitHub shows `dashboard.html` as source: download it and open it in a browser.
