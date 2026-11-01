"""
scripts/compare_versions.py — Détection automatique de régressions
entre deux versions de l'agent cible.

Exigence du projet :
  "Critère de succès : le système détecte automatiquement les
   régressions entre deux versions de l'agent cible."

Principe :
  1. Tu lances le pipeline complet une 1ère fois sur la version A de
     target_app (ex: avant une modif) → tu archives le snapshot.
  2. Tu modifies l'agent (nouvelle version B), tu relances le pipeline.
  3. Ce script compare les deux snapshots cas par cas et détecte :
       - PASS → FAIL   (régression franche)
       - score_moyen qui chute de plus de REGRESSION_THRESHOLD points
       - hallucination : False → True (régression critique)
       - latence qui augmente de plus de LATENCY_REGRESSION_PCT %
     et signale aussi les AMÉLIORATIONS (FAIL → PASS) pour ne pas juste
     lister des problèmes.

Usage :
    # Archiver la version actuelle des résultats comme baseline
    python scripts/compare_versions.py --snapshot baseline

    # ... modifier target_app, relancer run_pipeline.py ...

    # Comparer les résultats courants à la baseline
    python scripts/compare_versions.py --compare baseline

    # Comparer deux baselines nommées entre elles
    python scripts/compare_versions.py --compare baseline --against v2
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
SNAPSHOTS_DIR = ROOT / "reports" / "snapshots"

SCORE_FILES = ["scored_results.json", "web_scored_results.json", "mobile_scored_results.json"]

REGRESSION_SCORE_THRESHOLD = 0.5  # points sur 5 : score_moyen qui chute de plus de ça = régression
LATENCY_REGRESSION_PCT = 50  # % d'augmentation de latence considéré comme régression


def snapshot(name: str) -> None:
    """Archive les fichiers scored_results.json actuels sous reports/snapshots/<name>/."""
    dest = SNAPSHOTS_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in SCORE_FILES:
        src = REPORTS / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
            copied.append(fname)

    meta = {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": copied,
    }
    (dest / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not copied:
        print("⚠ Aucun fichier de scores trouvé dans reports/. Lance le pipeline d'abord.")
        return
    print(f"✓ Snapshot '{name}' créé avec {len(copied)} fichier(s) : {', '.join(copied)}")


def _load_results_by_id(dir_path: Path) -> dict[str, dict]:
    """Charge tous les scored_results.json d'un dossier (ou de reports/ si dir_path == REPORTS)
    et retourne un dict {test_id: résultat scoré}."""
    by_id: dict[str, dict] = {}
    for fname in SCORE_FILES:
        path = dir_path / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("scored_results", []):
            by_id[r.get("id", "?")] = r
    return by_id


def diff_results(old: dict[str, dict], new: dict[str, dict]) -> dict:
    """
    Compare two {test_id: scored_result} mappings case by case (pure function, no I/O).

    Returns a dict with: regressions (list of {id, canal, reasons}),
    improvements (list of {id, detail}), unchanged (int), new_only and
    missing (sorted lists of test ids), common_ids (sorted list).
    """
    regressions = []
    improvements = []
    unchanged = 0

    common_ids = set(old) & set(new)
    for tid in sorted(common_ids):
        o, n = old[tid], new[tid]
        reasons = []

        # 1. Verdict PASS → FAIL (régression franche)
        if o.get("verdict") == "PASS" and n.get("verdict") == "FAIL":
            reasons.append(f"verdict PASS→FAIL ({n.get('reason', '?')})")
        elif o.get("verdict") == "FAIL" and n.get("verdict") == "PASS":
            improvements.append({"id": tid, "detail": "verdict FAIL→PASS"})

        # 2. Nouvelle hallucination
        if not o.get("hallucination") and n.get("hallucination"):
            reasons.append("nouvelle hallucination détectée")

        # 3. Chute de score_moyen
        delta_score = (o.get("score_moyen", 0) or 0) - (n.get("score_moyen", 0) or 0)
        if delta_score >= REGRESSION_SCORE_THRESHOLD:
            reasons.append(f"score_moyen {o.get('score_moyen')}→{n.get('score_moyen')} (Δ-{round(delta_score, 2)})")

        # 4. Latence dégradée
        old_lat = o.get("latency_ms", 0) or 0
        new_lat = n.get("latency_ms", 0) or 0
        if old_lat > 0 and new_lat > old_lat * (1 + LATENCY_REGRESSION_PCT / 100):
            pct = round((new_lat - old_lat) / old_lat * 100)
            reasons.append(f"latence {old_lat}ms→{new_lat}ms (+{pct}%)")

        if reasons:
            regressions.append({"id": tid, "canal": n.get("canal", "?"), "reasons": reasons})
        else:
            unchanged += 1

    new_only = sorted(set(new) - set(old))
    missing = sorted(set(old) - set(new))

    return {
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "new_only": new_only,
        "missing": missing,
        "common_ids": sorted(common_ids),
    }


def compare(baseline_name: str, current_name: str | None = None) -> dict:
    """
    Compare une baseline archivée aux résultats courants (reports/) par défaut,
    ou à une autre baseline archivée si --against est fourni.
    """
    baseline_dir = SNAPSHOTS_DIR / baseline_name
    if not baseline_dir.exists():
        raise SystemExit(
            f"Baseline '{baseline_name}' introuvable dans {SNAPSHOTS_DIR}. "
            f"Crée-la d'abord avec --snapshot {baseline_name}."
        )

    current_dir = SNAPSHOTS_DIR / current_name if current_name else REPORTS
    if current_name and not current_dir.exists():
        raise SystemExit(f"Baseline '{current_name}' introuvable dans {SNAPSHOTS_DIR}.")

    old = _load_results_by_id(baseline_dir)
    new = _load_results_by_id(current_dir)

    if not old or not new:
        raise SystemExit("Impossible de comparer : l'une des deux versions n'a aucun résultat scoré.")

    diff = diff_results(old, new)
    regressions = diff["regressions"]
    improvements = diff["improvements"]

    report = {
        "baseline": baseline_name,
        "current": current_name or "reports/ (courant)",
        "compared_at": datetime.now().isoformat(timespec="seconds"),
        "total_common_tests": len(diff["common_ids"]),
        "regressions_count": len(regressions),
        "improvements_count": len(improvements),
        "unchanged_count": diff["unchanged"],
        "regressions": regressions,
        "improvements": improvements,
        "new_test_ids": diff["new_only"],
        "missing_test_ids": diff["missing"],
        "verdict": "REGRESSION_DETECTEE" if regressions else "AUCUNE_REGRESSION",
    }

    out_path = REPORTS / "regression_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_report(report)
    print(f"\n📄 Rapport détaillé sauvegardé dans {out_path.relative_to(ROOT)}")
    return report


def _print_report(report: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  COMPARAISON : {report['baseline']}  →  {report['current']}")
    print(f"{'=' * 70}")
    print(f"  Tests comparés (présents dans les 2 versions) : {report['total_common_tests']}")
    print(f"  Inchangés   : {report['unchanged_count']}")
    print(f"  Améliorés   : {report['improvements_count']}")
    print(f"  Régressions : {report['regressions_count']}")

    if report["regressions"]:
        print("\n  🔴 RÉGRESSIONS DÉTECTÉES :")
        for r in report["regressions"]:
            print(f"    - {r['id']} [{r['canal']}] : {'; '.join(r['reasons'])}")
    else:
        print("\n  ✅ Aucune régression détectée entre les deux versions.")

    if report["improvements"]:
        print("\n  🟢 Améliorations :")
        for r in report["improvements"]:
            print(f"    - {r['id']} : {r['detail']}")

    if report["new_test_ids"]:
        print(f"\n  ℹ Nouveaux cas de test (absents de la baseline) : {', '.join(report['new_test_ids'])}")
    if report["missing_test_ids"]:
        print(
            f"  ℹ Cas de test disparus (présents dans la baseline seulement) : {', '.join(report['missing_test_ids'])}"
        )

    print(f"\n  VERDICT GLOBAL : {report['verdict']}")
    print(f"{'=' * 70}\n")


def main():
    """CLI : --snapshot <nom> archive les scores courants, --compare <nom> détecte les régressions (exit 1 si régression)."""
    parser = argparse.ArgumentParser(
        description="Détection de régressions entre deux versions de l'agent cible (target_app).",
        epilog="""
Exemples :
  python scripts/compare_versions.py --snapshot baseline
  python scripts/compare_versions.py --compare baseline
  python scripts/compare_versions.py --compare v1 --against v2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot", metavar="NOM", help="Archive les résultats scorés courants (reports/) sous ce nom."
    )
    parser.add_argument(
        "--compare", metavar="BASELINE", help="Compare une baseline archivée aux résultats courants (ou à --against)."
    )
    parser.add_argument(
        "--against",
        metavar="NOM",
        default=None,
        help="Compare à une autre baseline archivée plutôt qu'aux résultats courants.",
    )
    args = parser.parse_args()

    if not args.snapshot and not args.compare:
        parser.print_help()
        raise SystemExit(1)

    if args.snapshot:
        snapshot(args.snapshot)

    if args.compare:
        result = compare(args.compare, args.against)
        raise SystemExit(1 if result["verdict"] == "REGRESSION_DETECTEE" else 0)


if __name__ == "__main__":
    main()
