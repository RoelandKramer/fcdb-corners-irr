"""
Inter-Rater Reliability analysis for the corner-role labelling study.

After everyone has finished labelling, collect their CSVs into the
``labels/`` folder (one per rater, named ``labels_<RaterName>.csv``)
and run this script.

It computes:
  - Pairwise Cohen's kappa for every pair of raters (per side + overall)
  - Fleiss' kappa across all raters
  - Per-class agreement on the major roles
  - Marking-assignment agreement (when both raters agree a defender is
    MAN, do they agree on which attacker they mark?)

Output: a printed summary, plus ``labels/irr_report.csv`` with the
pairwise kappa table.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

THIS_DIR   = Path(__file__).parent
LABELS_DIR = THIS_DIR / "labels"
REPORT_OUT = LABELS_DIR / "irr_report.csv"


# ---------------------------------------------------------------------------
def load_all_labels() -> pd.DataFrame:
    rows = []
    for f in sorted(LABELS_DIR.glob("labels_*.csv")):
        df = pd.read_csv(f)
        rows.append(df)
    if not rows:
        sys.exit(f"No label CSVs found in {LABELS_DIR}.")
    out = pd.concat(rows, ignore_index=True)
    # Drop empty role rows (player wasn't labelled)
    out = out[out["role"].astype(str).str.strip() != ""].copy()
    return out


def _wide_role_matrix(df: pd.DataFrame, side: str | None = None) -> pd.DataFrame:
    """Return a wide table: rows = (corner_id_int, player_team, jersey),
    columns = raters, values = role label."""
    sub = df if side is None else df[df["player_team"] == side]
    sub = sub[~sub["role"].isin(["DON'T KNOW", ""])].copy()
    wide = sub.pivot_table(
        index=["corner_id_int", "player_team", "jersey"],
        columns="rater",
        values="role",
        aggfunc="last",
    )
    return wide


def fleiss_kappa(matrix: np.ndarray) -> float:
    """Fleiss' kappa for an N×k table of per-item rater counts."""
    N, k = matrix.shape
    n = matrix.sum(axis=1)[0]
    if n < 2:
        return float("nan")
    p = matrix.sum(axis=0) / (N * n)
    P_bar  = ((matrix ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_mean = P_bar.mean()
    Pe = (p ** 2).sum()
    if 1 - Pe == 0:
        return 1.0
    return float((P_mean - Pe) / (1 - Pe))


def compute_fleiss(wide: pd.DataFrame) -> float:
    """Build the count matrix for Fleiss and call fleiss_kappa."""
    wide_full = wide.dropna(how="any")
    if wide_full.empty:
        return float("nan")
    categories = sorted({v for col in wide_full.columns for v in wide_full[col].dropna().unique()})
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    M = np.zeros((len(wide_full), len(categories)), dtype=int)
    for i, (_, row) in enumerate(wide_full.iterrows()):
        for v in row.values:
            M[i, cat_to_idx[v]] += 1
    return fleiss_kappa(M)


def pairwise_kappa_table(wide: pd.DataFrame) -> pd.DataFrame:
    raters = list(wide.columns)
    rows = []
    for r1, r2 in combinations(raters, 2):
        common = wide[[r1, r2]].dropna()
        if common.empty:
            rows.append({"rater_1": r1, "rater_2": r2, "n": 0, "kappa": float("nan")})
            continue
        k = cohen_kappa_score(common[r1], common[r2])
        rows.append({"rater_1": r1, "rater_2": r2, "n": len(common), "kappa": k})
    return pd.DataFrame(rows)


def per_class_kappa(wide: pd.DataFrame, classes: list[str]) -> dict:
    """For each class, treat it as one-vs-rest and compute mean pairwise kappa."""
    raters = list(wide.columns)
    out = {}
    for cls in classes:
        ks = []
        for r1, r2 in combinations(raters, 2):
            common = wide[[r1, r2]].dropna()
            if common.empty: continue
            a = (common[r1] == cls).astype(int)
            b = (common[r2] == cls).astype(int)
            if a.nunique() < 2 and b.nunique() < 2:
                continue
            ks.append(cohen_kappa_score(a, b))
        out[cls] = float(np.mean(ks)) if ks else float("nan")
    return out


def marking_agreement(df: pd.DataFrame) -> dict:
    """Among pairs where both raters labelled a defender MAN, how often do
    they agree on the marked attacker's jersey?"""
    def_man = df[(df["player_team"] == "DEF") & (df["role"] == "MAN")].copy()
    wide = def_man.pivot_table(
        index=["corner_id_int", "jersey"],
        columns="rater",
        values="marks",
        aggfunc="last",
    )
    raters = list(wide.columns)
    totals, agreed = 0, 0
    for r1, r2 in combinations(raters, 2):
        common = wide[[r1, r2]].dropna()
        for _, row in common.iterrows():
            totals += 1
            if int(row[r1]) == int(row[r2]):
                agreed += 1
    return {"pairs_both_MAN": totals,
            "agreed_on_attacker": agreed,
            "agreement_rate": agreed / totals if totals else float("nan")}


# ---------------------------------------------------------------------------
def main():
    df = load_all_labels()
    raters = sorted(df["rater"].unique())
    print(f"Loaded {len(df)} role labels from {len(raters)} raters: "
          f"{', '.join(raters)}\n")

    # ------------------------------------------------------------------
    # Defender side
    # ------------------------------------------------------------------
    wide_def = _wide_role_matrix(df, side="DEF")
    print("=== Defender role agreement ===")
    print(f"  Rated (corner, jersey) pairs: {len(wide_def)}")
    pw_def = pairwise_kappa_table(wide_def)
    print("\n  Pairwise Cohen's kappa:")
    print(pw_def.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fk = compute_fleiss(wide_def)
    print(f"\n  Fleiss' kappa (all raters): {fk:.3f}")
    pcs = per_class_kappa(wide_def, ["MAN", "ZONAL", "SHORT", "COUNTER"])
    print("\n  Per-class kappa (one-vs-rest, averaged across pairs):")
    for k, v in pcs.items():
        print(f"    {k:8s}: {v:.3f}")

    # ------------------------------------------------------------------
    # Attacker side
    # ------------------------------------------------------------------
    wide_att = _wide_role_matrix(df, side="ATT")
    print("\n=== Attacker role agreement ===")
    print(f"  Rated (corner, jersey) pairs: {len(wide_att)}")
    pw_att = pairwise_kappa_table(wide_att)
    print("\n  Pairwise Cohen's kappa:")
    print(pw_att.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fk = compute_fleiss(wide_att)
    print(f"\n  Fleiss' kappa (all raters): {fk:.3f}")
    att_classes = ["TARGET","DECOY","STATIC","SECOND_BALL","BLOCK_GK","BLOCK_DEF"]
    pcs = per_class_kappa(wide_att, att_classes)
    print("\n  Per-class kappa (one-vs-rest, averaged across pairs):")
    for k, v in pcs.items():
        print(f"    {k:11s}: {v:.3f}")

    # ------------------------------------------------------------------
    # Marking agreement
    # ------------------------------------------------------------------
    print("\n=== Marking-assignment agreement (when both raters say MAN) ===")
    m = marking_agreement(df)
    print(f"  Pairs where both said MAN: {m['pairs_both_MAN']}")
    print(f"  Agreed on attacker:        {m['agreed_on_attacker']}")
    print(f"  Agreement rate:            {m['agreement_rate']:.1%}")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    pw_def["side"] = "DEF"
    pw_att["side"] = "ATT"
    report = pd.concat([pw_def, pw_att], ignore_index=True)
    report.to_csv(REPORT_OUT, index=False)
    print(f"\nReport saved -> {REPORT_OUT}")
    print("\nReference: kappa > 0.80 = almost perfect; 0.60-0.80 substantial; "
          "0.40-0.60 moderate; <0.40 fair-or-worse.")


if __name__ == "__main__":
    main()
