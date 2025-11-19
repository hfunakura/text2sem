import argparse
from pathlib import Path
from typing import Dict, List
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_tsv",
        default="./results/exp_fol_t5base_ep50_seed5/predictions/sick_test_master_valid_with_predicted_formula.tsv",
    )
    parser.add_argument(
        "--report_tsv",
        default="./results/exp_fol_t5base_ep50_seed5/predictions/detailed_report.tsv",
    )
    parser.add_argument(
        "--wff_jsonl",
        default="./analysis/openrouter_labels/fol_t5base_ep50_seed5_wff_labels_langchain_single_raw.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/exp_fol_t5base_ep50_seed5/predictions/analysis",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=8,
    )
    return parser.parse_args()


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: True if str(x).lower() == "true" else False)


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "count": float(len(df)),
        "is_exact_match_mean": float(df["is_exact_match"].mean()) if (len(df) > 0 and "is_exact_match" in df.columns) else np.nan,
        "is_prover_correct_mean": float(df["is_prover_correct"].mean()) if (len(df) > 0 and "is_prover_correct" in df.columns) else np.nan,
        "drs_fscore_mean": float(df["drs_fscore"].mean()) if (len(df) > 0 and "drs_fscore" in df.columns) else np.nan,
    }


def dataframe_from_metrics_dict(metrics: Dict[str, Dict[str, float]], index_name: str) -> pd.DataFrame:
    records = []
    for k, v in metrics.items():
        row = {index_name: k}
        row.update(v)
        records.append(row)
    return pd.DataFrame.from_records(records)


def ensure_bool_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["cc", "pp", "pss", "has_neg", "has_or", "has_all"]:
        if col in df.columns:
            df[col] = to_bool_series(df[col])
    for col in ["is_exact_match", "is_prover_correct"]:
        if col in df.columns:
            df[col] = to_bool_series(df[col])
    if "drs_fscore" in df.columns:
        df["drs_fscore"] = to_float_series(df["drs_fscore"])
    if "complexity" in df.columns:
        df["complexity"] = to_float_series(df["complexity"])
    return df


def add_length_bins(df: pd.DataFrame) -> pd.DataFrame:
    if "text" in df.columns:
        df["token_length"] = df["text"].fillna("").str.split().map(len)
        try:
            df["length_bin"] = pd.qcut(df["token_length"], q=3, duplicates="drop")
        except Exception:
            df["length_bin"] = pd.cut(df["token_length"], bins=3)
    return df


def annotate_bars(ax: plt.Axes, fmt: str = "{:.1%}") -> None:
    for p in ax.patches:
        h = p.get_height()
        if not np.isfinite(h) or h <= 0:
            continue
        ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def humanize_category(name: str) -> str:
    if not isinstance(name, str):
        return name
    base = name
    for suf in ["Mismatch", "Error", "Issue", "Problem", "Violation"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", base)
    tokens = s.replace("_", " ").split()
    if not tokens:
        return s
    first = tokens[0][:1].upper() + tokens[0][1:].lower()
    rest = [t.lower() for t in tokens[1:]]
    return " ".join([first] + rest)


def analyze(input_tsv: Path, report_tsv: Path, output_dir: Path, wff_jsonl: Path, num_bins: int) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_context("paper", rc={
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_tsv, sep="\t", dtype=str)
    rep = pd.read_csv(report_tsv, sep="\t", dtype=str)

    metrics_cols = [
        c
        for c in ["id", "is_exact_match", "is_prover_correct", "drs_precision", "drs_recall", "drs_fscore"]
        if c in rep.columns
    ]
    if "id" in df.columns and len(metrics_cols) > 1:
        missing_metrics = any((c not in df.columns) for c in metrics_cols if c != "id")
        if missing_metrics:
            df = df.merge(rep[metrics_cols], on="id", how="left")

    df = ensure_bool_numeric(df)
    df = add_length_bins(df)

    baseline = compute_metrics(df)
    pd.DataFrame([baseline]).to_csv(output_dir / "baseline_summary.tsv", sep="\t", index=False)

    single_cols_struct = ["cc", "pp", "pss"]
    single_cols_logic = ["has_neg", "has_or", "has_all"]

    single_struct_metrics: Dict[str, Dict[str, float]] = {}
    for col in single_cols_struct:
        if col in df.columns:
            m_true = compute_metrics(df[df[col] == True])
            m_false = compute_metrics(df[df[col] == False])
            single_struct_metrics[f"{col}=True"] = m_true
            single_struct_metrics[f"{col}=False"] = m_false
    dataframe_from_metrics_dict(single_struct_metrics, "slice").to_csv(
        output_dir / "slice_single_struct.tsv", sep="\t", index=False
    )

    single_logic_metrics: Dict[str, Dict[str, float]] = {}
    for col in single_cols_logic:
        if col in df.columns:
            m_true = compute_metrics(df[df[col] == True])
            m_false = compute_metrics(df[df[col] == False])
            single_logic_metrics[f"{col}=True"] = m_true
            single_logic_metrics[f"{col}=False"] = m_false
    dataframe_from_metrics_dict(single_logic_metrics, "slice").to_csv(
        output_dir / "slice_single_logic.tsv", sep="\t", index=False
    )

    interactions = [
        ("cc", "has_or"),
        ("pp", "has_all"),
        ("pss", "has_neg"),
    ]
    for a, b in interactions:
        if a in df.columns and b in df.columns:
            metrics: Dict[str, Dict[str, float]] = {}
            for va in [False, True]:
                for vb in [False, True]:
                    sub = df[(df[a] == va) & (df[b] == vb)]
                    metrics[f"{a}={va},{b}={vb}"] = compute_metrics(sub)
            out_df = dataframe_from_metrics_dict(metrics, "slice")
            out_df.to_csv(output_dir / f"interaction_{a}_{b}.tsv", sep="\t", index=False)

    rows = ["cc", "pp", "pss"]
    cols = ["has_neg", "has_or", "has_all"]
    heat = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if r in df.columns and c in df.columns and "is_exact_match" in df.columns:
                sub = df[(df[r] == True) & (df[c] == True)]
                heat[i, j] = sub["is_exact_match"].mean() if len(sub) > 0 else np.nan
    heat_df = pd.DataFrame(heat, index=rows, columns=cols)
    plt.figure(figsize=(6.5, 4))
    ax = sns.heatmap(heat_df, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", square=True, linewidths=0.5, linecolor="white", cbar_kws={"label": "Exact Match"})
    plt.title("Exact Match by Structural x Logical Phenomena")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_exact_match.png", dpi=300, bbox_inches="tight")
    plt.close()

    if set(single_cols_struct).issubset(df.columns) and "is_exact_match" in df.columns:
        tmp = []
        for col in single_cols_struct:
            tmp.append({"category": col, "value": float(df[df[col] == True]["is_exact_match"].mean())})
        plot_df = pd.DataFrame(tmp)
        plt.figure(figsize=(5.2, 3.5))
        ax = sns.barplot(data=plot_df, x="category", y="value", palette="Set2")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.ylim(0, 1)
        plt.ylabel("Exact Match")
        plt.title("Exact Match for cc/pp/pss=True")
        annotate_bars(ax)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "bar_single_struct_exact.png", dpi=300, bbox_inches="tight")
        plt.close()

    if set(single_cols_logic).issubset(df.columns) and "is_exact_match" in df.columns:
        tmp = []
        for col in single_cols_logic:
            tmp.append({"category": col, "value": float(df[df[col] == True]["is_exact_match"].mean())})
        plot_df = pd.DataFrame(tmp)
        plt.figure(figsize=(5.2, 3.5))
        ax = sns.barplot(data=plot_df, x="category", y="value", palette="Set2")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.ylim(0, 1)
        plt.ylabel("Exact Match")
        plt.title("Exact Match for has_*=True")
        annotate_bars(ax)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "bar_single_logic_exact.png", dpi=300, bbox_inches="tight")
        plt.close()

    if "length_bin" in df.columns and "drs_fscore" in df.columns:
        length_stats = df.groupby("length_bin").apply(compute_metrics).apply(pd.Series)
        length_stats.reset_index().rename(columns={"index": "length_bin"}).to_csv(
            output_dir / "length_bin_summary.tsv", sep="\t", index=False
        )
        plt.figure(figsize=(6, 3.6))
        ax = sns.boxplot(data=df, x="length_bin", y="drs_fscore", palette="Set2")
        plt.title("DRS F-score by Length Bin")
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "box_drs_by_length.png", dpi=300, bbox_inches="tight")
        plt.close()

    if "is_exact_match" in df.columns:
        failures_em = df[df["is_exact_match"] == False].copy()
        cols_out: List[str] = [
            "id",
            "split",
            "text",
            "formula",
            "predicted_formula",
            "cc",
            "pp",
            "pss",
            "has_neg",
            "has_or",
            "has_all",
            "is_exact_match",
            "is_prover_correct",
            "drs_fscore",
            "complexity",
        ]
        cols_out = [c for c in cols_out if c in df.columns]
        failures_em[cols_out].to_csv(output_dir / "failures_exact_mismatch.tsv", sep="\t", index=False)

    if "is_prover_correct" in df.columns:
        failures_pc = df[df["is_prover_correct"] == False].copy()
        cols_out2 = [
            c
            for c in [
                "id",
                "split",
                "text",
                "formula",
                "predicted_formula",
                "cc",
                "pp",
                "pss",
                "has_neg",
                "has_or",
                "has_all",
                "is_exact_match",
                "is_prover_correct",
                "drs_fscore",
                "complexity",
            ]
            if c in df.columns
        ]
        failures_pc[cols_out2].to_csv(output_dir / "failures_prover_incorrect.tsv", sep="\t", index=False)

    if set(["cc", "pp", "pss"]).issubset(df.columns) and "is_prover_correct" in df.columns:
        rows_cat = ["cc", "pp", "pss"]
        label_map = {"cc": "CC", "pp": "PP", "pss": "PSS"}
        records = []
        for col in rows_cat:
            for flag in [True, False]:
                sub = df[df[col] == flag]
                value = float(sub["is_prover_correct"].mean()) if len(sub) > 0 else np.nan
                records.append({"category": col, "category_label": label_map.get(col, col), "flag": "Present" if flag else "Absent", "value": value})
        plot_df2 = pd.DataFrame(records)
        plt.figure(figsize=(6.2, 3.8))
        present_color = sns.color_palette("Set2")[0]
        absent_color = "#B0B0B0"
        flag_palette = {"Present": present_color, "Absent": absent_color}
        ax = sns.barplot(data=plot_df2, x="category_label", y="value", hue="flag", order=[label_map[c] for c in rows_cat], palette=flag_palette)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.ylim(0, 1)
        plt.xlabel("Category")
        plt.ylabel("Prover Accuracy")
        plt.legend(title=None, loc="upper right", frameon=True)
        annotate_bars(ax)
        sns.despine()
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(output_dir / "bar_prover_struct_true_false.png", dpi=300, bbox_inches="tight")
        plt.close()

    if "complexity" in df.columns and df["complexity"].notna().any():
        comp_df = df[np.isfinite(df["complexity"])].copy()
        comp_df = comp_df.sort_values("complexity").reset_index(drop=True)
        n = len(comp_df)
        if n > 0:
            comp_df["bin_index"] = (np.floor(np.arange(n) * num_bins / n).astype(int) + 1)
        else:
            comp_df["bin_index"] = []
        agg_oct = (
            comp_df.groupby("bin_index")
            .agg(
                prover_accuracy=("is_prover_correct", "mean"),
                drs_fscore=("drs_fscore", "mean"),
                count=("id", "count"),
                min_complexity=("complexity", "min"),
                max_complexity=("complexity", "max"),
            )
            .reset_index()
        )
        full_idx = pd.DataFrame({"bin_index": list(range(1, num_bins + 1))})
        agg_oct = full_idx.merge(agg_oct, on="bin_index", how="left")
        agg_oct = agg_oct.sort_values("bin_index")
        agg_oct.to_csv(output_dir / "complexity_octile_summary.tsv", sep="\t", index=False)

        long_df = agg_oct.melt(
            id_vars=["bin_index"],
            value_vars=["prover_accuracy", "drs_fscore"],
            var_name="metric",
            value_name="value",
        )
        metric_label_map = {"prover_accuracy": "Prover Accuracy", "drs_fscore": "Dmatch F1"}
        long_df["metric_label"] = long_df["metric"].map(metric_label_map)

        plt.figure(figsize=(6.4, 3.8))
        ax = sns.lineplot(data=long_df, x="bin_index", y="value", hue="metric_label", marker="o")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.ylim(0, 1)
        plt.xlabel(f"Formula Complexity ({num_bins} Bins)")
        plt.ylabel("Score")
        plt.xticks(ticks=list(range(1, num_bins + 1)))
        plt.legend(title=None, loc="upper right", frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "lines_by_complexity_octiles.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6.4, 3.8))
        ax = sns.barplot(data=long_df, x="bin_index", y="value", hue="metric_label", palette="Set2")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.ylim(0, 1)
        plt.xlabel(f"Formula Complexity ({num_bins} Bins)")
        plt.ylabel("Score")
        plt.xticks(ticks=list(range(1, num_bins + 1)))
        plt.legend(title=None, loc="upper right", frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "bars_by_complexity_octiles.png", dpi=300, bbox_inches="tight")
        plt.close()

    if wff_jsonl.exists():
        wff_df = pd.read_json(wff_jsonl, lines=True)
        if "raw" in wff_df.columns:
            wff_df["category"] = wff_df["raw"].map(lambda x: x.get("category") if isinstance(x, dict) else np.nan)
        counts = (
            wff_df.dropna(subset=["category"])
            .groupby("category")
            .size()
            .reset_index(name="count")
            .sort_values(["count", "category"], ascending=[False, True])
        )
        counts.to_csv(output_dir / "wff_category_counts.tsv", sep="\t", index=False)
        if len(counts) > 0:
            display_map = {
                "QuantifierCountMismatch": "Quantifier count",
                "QuantifierOrderMismatch": "Quantifier order",
                "VariableNameMismatch": "Variable name",
                "VariableBindingMismatch": "Variable binding",
                "PredicateCountMismatch": "Predicate count",
                "PredicateNameMismatch": "Predicate name",
                "ArgumentCountMismatch": "Argument count",
                "OperatorMismatch": "Operator",
                "ParenthesisMismatch": "Parentheses",
                "ScopeError": "Scope",
                "TypeMismatch": "Type",
                "NegationMismatch": "Negation",
                "ConjunctionMismatch": "Conjunction",
                "DisjunctionMismatch": "Disjunction",
                "ImplicationMismatch": "Implication",
                "EquivalenceMismatch": "Equivalence",
                "FreeVariableError": "Free variable",
                "UnboundVariableError": "Unbound variable",
                "SyntaxError": "Syntax",
            }
            counts_plot = counts.copy()
            counts_plot["category_display"] = counts_plot["category"].map(lambda c: display_map.get(c, humanize_category(c)))
            h = max(2.5, 0.3 * len(counts_plot) + 1)
            plt.figure(figsize=(6.4, h))
            ax = sns.barplot(data=counts_plot, x="count", y="category_display", orient="h", palette="Set2")
            plt.xlabel("Count")
            plt.ylabel("Misprediction Category")
            max_count = counts_plot["count"].max()
            total = counts_plot["count"].sum()
            pad = max_count * 0.02
            ax.set_xlim(0, max_count * 1.15)
            for patch, (_, row) in zip(ax.patches, counts_plot.iterrows()):
                y = patch.get_y() + patch.get_height() / 2.0
                x = row["count"] + pad
                label = f"{int(row['count'])} ({row['count']/total*100:.1f}%)"
                ax.text(x, y, label, va="center", ha="left", fontsize=8)
            sns.despine()
            plt.tight_layout()
            plt.savefig(output_dir / "bar_wff_category_counts.png", dpi=300, bbox_inches="tight")
            plt.close()


def main() -> int:
    args = parse_args()
    analyze(Path(args.input_tsv), Path(args.report_tsv), Path(args.output_dir), Path(args.wff_jsonl), args.num_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())