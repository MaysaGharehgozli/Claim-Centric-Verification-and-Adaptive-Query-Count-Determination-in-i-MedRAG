# # -*- coding: utf-8 -*-
# """
# significance_test.py
# ====================
# Significance tests for baseline i-MedRAG vs claim-judged adaptive i-MedRAG.

# Usage:
#     python significance_test.py path/to/dataset_comparison.json
#     python significance_test.py path/to/dataset_comparison.json --drop-none

# Requires: numpy, pandas, scipy
# Tests:
#   1) Accuracy : exact McNemar test (paired binary) + Wilson CI + bootstrap CI of the difference
#   2) Queries  : paired Wilcoxon signed-rank + paired t-test + Cohen's dz + bootstrap CI
#   3) Relation : is #queries associated with correctness? (Mann-Whitney U, Spearman) per method
# """
# import sys
# import json
# import argparse
# import numpy as np
# import pandas as pd
# from scipy import stats


# def wilson_ci(k, n, z=1.96):
#     p = k / n
#     den = 1 + z * z / n
#     ctr = (p + z * z / (2 * n)) / den
#     half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
#     return ctr - half, ctr + half


# def mcnemar_exact(b, c):
#     """Two-sided exact McNemar p-value from discordant counts b, c."""
#     n = b + c
#     if n == 0:
#         return 1.0
#     return min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


# def bootstrap_ci(x, n_boot=5000, seed=0):
#     rng = np.random.default_rng(seed)
#     x = np.asarray(x, dtype=float)
#     means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
#     return np.percentile(means, [2.5, 97.5])


# def load(path):
#     with open(path, encoding="utf-8") as f:
#         data = json.load(f)
#     rows = []
#     for r in data:
#         rows.append(
#             dict(
#                 id=r["question_id"],
#                 gold=r["expected_answer_idx"],
#                 base_pred=r["baseline"]["prediction"],
#                 judged_pred=r["claim_judged"]["prediction"],
#                 base_q=r["baseline"]["queries_used"],
#                 judged_q=r["claim_judged"]["queries_used"],
#             )
#         )
#     df = pd.DataFrame(rows)
#     df["base_ok"] = (df.base_pred == df.gold).astype(int)   # None counts as wrong
#     df["judged_ok"] = (df.judged_pred == df.gold).astype(int)
#     return df


# def analyze(df, label):
#     n = len(df)
#     print("\n" + "=" * 70)
#     print(f"{label}  (n = {n})")
#     print("=" * 70)

#     # ---------------- 1) Accuracy ----------------
#     print("\n[1] ACCURACY (paired, exact McNemar)")
#     for name, col in [("baseline", "base_ok"), ("judged  ", "judged_ok")]:
#         k = df[col].sum()
#         lo, hi = wilson_ci(k, n)
#         print(f"  {name}: {k}/{n} = {k / n:.3f}   95% Wilson CI [{lo:.3f}, {hi:.3f}]")

#     both = int(((df.base_ok == 1) & (df.judged_ok == 1)).sum())
#     b = int(((df.base_ok == 1) & (df.judged_ok == 0)).sum())   # baseline only correct
#     c = int(((df.base_ok == 0) & (df.judged_ok == 1)).sum())   # judged only correct
#     neither = int(((df.base_ok == 0) & (df.judged_ok == 0)).sum())
#     print(f"  both correct={both}, baseline-only={b}, judged-only={c}, both wrong={neither}")
#     p_mc = mcnemar_exact(b, c)
#     diff = (df.judged_ok - df.base_ok).values
#     lo, hi = bootstrap_ci(diff)
#     print(f"  accuracy difference (judged - baseline) = {diff.mean():+.3f}   "
#           f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
#     print(f"  McNemar exact p = {p_mc:.4f}  ->  "
#           f"{'SIGNIFICANT' if p_mc < 0.05 else 'not significant'} at alpha=0.05")

#     # ---------------- 2) Number of queries ----------------
#     print("\n[2] NUMBER OF QUERIES (paired)")
#     print(f"  baseline: mean={df.base_q.mean():.2f}  sd={df.base_q.std():.2f}  "
#           f"range=[{df.base_q.min()}, {df.base_q.max()}]")
#     print(f"  judged  : mean={df.judged_q.mean():.2f}  sd={df.judged_q.std():.2f}  "
#           f"range=[{df.judged_q.min()}, {df.judged_q.max()}]")
#     dq = (df.base_q - df.judged_q).values
#     lo, hi = bootstrap_ci(dq)
#     print(f"  mean saving = {dq.mean():.2f} queries ({100 * dq.mean() / df.base_q.mean():.1f}%)   "
#           f"bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
#     t = stats.ttest_rel(df.base_q, df.judged_q)
#     print(f"  paired t-test        : t={t.statistic:.2f}, p={t.pvalue:.3g}")
#     try:
#         w = stats.wilcoxon(df.base_q, df.judged_q)
#         print(f"  Wilcoxon signed-rank : W={w.statistic:.1f}, p={w.pvalue:.3g}")
#     except ValueError as e:
#         print(f"  Wilcoxon not computable: {e}")
#     sd = dq.std(ddof=1)
#     if sd > 0:
#         print(f"  Cohen's dz = {dq.mean() / sd:.2f}")

#     # ---------------- 3) Queries vs correctness ----------------
#     print("\n[3] IS #QUERIES ASSOCIATED WITH CORRECTNESS? (within each method)")
#     for name, q, ok in [("baseline", "base_q", "base_ok"), ("judged", "judged_q", "judged_ok")]:
#         right = df.loc[df[ok] == 1, q]
#         wrong = df.loc[df[ok] == 0, q]
#         line = f"  {name}: mean q (correct)={right.mean():.2f} [n={len(right)}]"
#         if len(wrong):
#             line += f", (wrong)={wrong.mean():.2f} [n={len(wrong)}]"
#             try:
#                 u = stats.mannwhitneyu(right, wrong, alternative="two-sided")
#                 line += f" | Mann-Whitney p={u.pvalue:.4f}"
#             except ValueError:
#                 pass
#         print(line)
#         if df[q].nunique() > 1 and df[ok].nunique() > 1:
#             rho = stats.spearmanr(df[q], df[ok])
#             print(f"     Spearman rho={rho.statistic:+.3f}, p={rho.pvalue:.4f}")

#     print("\n  judged: accuracy by number of queries")
#     print(df.groupby("judged_q")["judged_ok"].agg(n="count", correct="sum", acc="mean").round(3).to_string())


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("json_path", help="dataset_comparison.json")
#     ap.add_argument("--drop-none", action="store_true",
#                     help="also run analysis excluding questions where a method gave no prediction")
#     args = ap.parse_args()

#     df = load(args.json_path)
#     analyze(df, "ALL QUESTIONS (missing prediction counted as wrong)")
#     if args.drop_none:
#         valid = df[df.base_pred.notna() & df.judged_pred.notna()].copy()
#         analyze(valid, "VALID PREDICTIONS ONLY")

#     df.to_csv("per_question_results.csv", index=False)
#     print("\nSaved per-question table -> per_question_results.csv")


# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
"""
significance_test.py
====================
Significance tests for baseline i-MedRAG vs claim-judged adaptive i-MedRAG,
pooled over several runs (e.g. 5 parts x 100 questions = 500).

Usage (pass the 5 dataset_comparison.json files, in any order):
    python Significance_test.py RUN1-100/dataset_comparison.json RUN101-200/dataset_comparison.json ...
    python Significance_test.py "claim_comparison_adaptiveRUN*/dataset_comparison.json" --drop-none

You can also pass the run FOLDERS; the script then looks for dataset_comparison.json inside.

Requires: numpy, pandas, scipy
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


def bootstrap_ci(x, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def resolve_paths(args_paths):
    """Expand wildcards; if an entry is a folder, use dataset_comparison.json inside it."""
    paths = []
    for p in args_paths:
        matches = sorted(glob.glob(p)) or [p]
        for m in matches:
            if os.path.isdir(m):
                found = sorted(glob.glob(os.path.join(m, "**", "dataset_comparison.json"),
                                         recursive=True))
                if not found:
                    sys.exit(f"No dataset_comparison.json found in folder: {m}")
                paths.extend(found)
            else:
                paths.append(m)
    return paths


def load(paths):
    rows = []
    for part_idx, path in enumerate(paths, 1):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  part {part_idx}: {len(data):4d} questions  <- {path}")
        for r in data:
            rows.append(
                dict(
                    part=part_idx,
                    id=r["question_id"],
                    gold=r["expected_answer_idx"],
                    base_pred=r["baseline"]["prediction"],
                    judged_pred=r["claim_judged"]["prediction"],
                    base_q=r["baseline"]["queries_used"],
                    judged_q=r["claim_judged"]["queries_used"],
                )
            )
    df = pd.DataFrame(rows)
    df["base_ok"] = (df.base_pred == df.gold).astype(int)   # None counts as wrong
    df["judged_ok"] = (df.judged_pred == df.gold).astype(int)

    dup = int(df.duplicated(subset=["id"]).sum())
    if dup:
        print(f"  WARNING: {dup} duplicate question_id values "
              f"(ids may restart in each part, or runs overlap). Rows are NOT dropped.")
    return df


def analyze(df, label):
    n = len(df)
    print("\n" + "=" * 70)
    print(f"{label}  (n = {n})")
    print("=" * 70)

    # ---------------- 1) Accuracy ----------------
    print("\n[1] ACCURACY (paired, exact McNemar)")
    for name, col in [("baseline", "base_ok"), ("judged  ", "judged_ok")]:
        k = df[col].sum()
        lo, hi = wilson_ci(k, n)
        print(f"  {name}: {k}/{n} = {k / n:.3f}   95% Wilson CI [{lo:.3f}, {hi:.3f}]")

    both = int(((df.base_ok == 1) & (df.judged_ok == 1)).sum())
    b = int(((df.base_ok == 1) & (df.judged_ok == 0)).sum())   # baseline only correct
    c = int(((df.base_ok == 0) & (df.judged_ok == 1)).sum())   # judged only correct
    neither = int(((df.base_ok == 0) & (df.judged_ok == 0)).sum())
    print(f"  both correct={both}, baseline-only={b}, judged-only={c}, both wrong={neither}")
    p_mc = mcnemar_exact(b, c)
    diff = (df.judged_ok - df.base_ok).values
    lo, hi = bootstrap_ci(diff)
    print(f"  accuracy difference (judged - baseline) = {diff.mean():+.3f}   "
          f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  McNemar exact p = {p_mc:.4f}  ->  "
          f"{'SIGNIFICANT' if p_mc < 0.05 else 'not significant'} at alpha=0.05")

    # ---------------- 2) Number of queries ----------------
    print("\n[2] NUMBER OF QUERIES (paired)")
    print(f"  baseline: mean={df.base_q.mean():.2f}  sd={df.base_q.std():.2f}  "
          f"range=[{df.base_q.min()}, {df.base_q.max()}]")
    print(f"  judged  : mean={df.judged_q.mean():.2f}  sd={df.judged_q.std():.2f}  "
          f"range=[{df.judged_q.min()}, {df.judged_q.max()}]")
    dq = (df.base_q - df.judged_q).values
    lo, hi = bootstrap_ci(dq)
    print(f"  mean saving = {dq.mean():.2f} queries ({100 * dq.mean() / df.base_q.mean():.1f}%)   "
          f"bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
    t = stats.ttest_rel(df.base_q, df.judged_q)
    print(f"  paired t-test        : t={t.statistic:.2f}, p={t.pvalue:.3g}")
    try:
        w = stats.wilcoxon(df.base_q, df.judged_q)
        print(f"  Wilcoxon signed-rank : W={w.statistic:.1f}, p={w.pvalue:.3g}")
    except ValueError as e:
        print(f"  Wilcoxon not computable: {e}")
    sd = dq.std(ddof=1)
    if sd > 0:
        print(f"  Cohen's dz = {dq.mean() / sd:.2f}")

    # ---------------- 3) Queries vs correctness ----------------
    print("\n[3] IS #QUERIES ASSOCIATED WITH CORRECTNESS? (within each method)")
    for name, q, ok in [("baseline", "base_q", "base_ok"), ("judged", "judged_q", "judged_ok")]:
        right = df.loc[df[ok] == 1, q]
        wrong = df.loc[df[ok] == 0, q]
        line = f"  {name}: mean q (correct)={right.mean():.2f} [n={len(right)}]"
        if len(wrong):
            line += f", (wrong)={wrong.mean():.2f} [n={len(wrong)}]"
            try:
                u = stats.mannwhitneyu(right, wrong, alternative="two-sided")
                line += f" | Mann-Whitney p={u.pvalue:.4f}"
            except ValueError:
                pass
        print(line)
        if df[q].nunique() > 1 and df[ok].nunique() > 1:
            rho = stats.spearmanr(df[q], df[ok])
            print(f"     Spearman rho={rho.statistic:+.3f}, p={rho.pvalue:.4f}")

    print("\n  judged: accuracy by number of queries")
    print(df.groupby("judged_q")["judged_ok"].agg(n="count", correct="sum", acc="mean").round(3).to_string())


def per_part_summary(df):
    print("\n" + "=" * 70)
    print("PER-PART SUMMARY (consistency check across the runs)")
    print("=" * 70)
    g = df.groupby("part").agg(n=("id", "count"),
                               base_acc=("base_ok", "mean"),
                               judged_acc=("judged_ok", "mean"),
                               base_q=("base_q", "mean"),
                               judged_q=("judged_q", "mean")).round(3)
    print(g.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+",
                    help="dataset_comparison.json files (or run folders / glob patterns)")
    ap.add_argument("--drop-none", action="store_true",
                    help="also run analysis excluding questions where a method gave no prediction")
    args = ap.parse_args()

    paths = resolve_paths(args.json_paths)
    print(f"Loading {len(paths)} file(s):")
    df = load(paths)

    if len(paths) > 1:
        per_part_summary(df)

    analyze(df, "ALL QUESTIONS (missing prediction counted as wrong)")
    if args.drop_none:
        valid = df[df.base_pred.notna() & df.judged_pred.notna()].copy()
        analyze(valid, "VALID PREDICTIONS ONLY")

    df.to_csv("per_question_results.csv", index=False)
    print("\nSaved per-question table -> per_question_results.csv")


if __name__ == "__main__":
    main()