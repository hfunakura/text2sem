#!/usr/bin/env python3
import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def find_predictions_tsv(experiment_dir: str) -> Optional[str]:
    predictions_dir = os.path.join(experiment_dir, "predictions")
    cand = []
    if os.path.isdir(predictions_dir):
        cand = sorted(glob(os.path.join(predictions_dir, "*.tsv")))
    if not cand:
        return None
    basename = os.path.basename(os.path.normpath(experiment_dir)) + ".tsv"
    preferred = [p for p in cand if os.path.basename(p) == basename]
    if preferred:
        return preferred[0]
    return cand[0]


def safe_mean(series: pd.Series) -> Optional[float]:
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return None
    return float(s.mean())


def compute_metrics_from_tsv(tsv_path: str) -> Dict[str, Optional[float]]:
    df = pd.read_csv(tsv_path, sep="\t")
    metrics = {
        "n": int(len(df)),
        "is_exact_match": None,
        "is_prover_correct": None,
        "drs_precision": None,
        "drs_recall": None,
        "drs_fscore": None,
    }
    if "is_exact_match" in df.columns:
        metrics["is_exact_match"] = safe_mean(df["is_exact_match"])  # proportion
    if "is_prover_correct" in df.columns:
        metrics["is_prover_correct"] = safe_mean(df["is_prover_correct"])  # proportion
    if "drs_precision" in df.columns:
        metrics["drs_precision"] = safe_mean(df["drs_precision"])  # average
    if "drs_recall" in df.columns:
        metrics["drs_recall"] = safe_mean(df["drs_recall"])  # average
    if "drs_fscore" in df.columns:
        metrics["drs_fscore"] = safe_mean(df["drs_fscore"])  # average
    return metrics


def parse_checkpoint_step(path: str) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)", path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def find_trainer_state_json(experiment_dir: str) -> Optional[str]:
    paths = sorted(glob(os.path.join(experiment_dir, "checkpoint-*", "trainer_state.json")))
    if paths:
        with_steps: List[Tuple[int, str]] = []
        for p in paths:
            step = parse_checkpoint_step(p)
            if step is not None:
                with_steps.append((step, p))
        if with_steps:
            with_steps.sort()
            return with_steps[-1][1]
        return paths[-1]
    direct = os.path.join(experiment_dir, "trainer_state.json")
    if os.path.isfile(direct):
        return direct
    any_found = sorted(glob(os.path.join(experiment_dir, "**", "trainer_state.json"), recursive=True))
    if any_found:
        return any_found[-1]
    return None


def extract_loss_history(trainer_state_path: str) -> pd.DataFrame:
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    history = state.get("log_history", [])
    rows: List[Dict[str, Any]] = []
    for item in history:
        if isinstance(item, dict) and ("loss" in item) and ("step" in item):
            rows.append({"step": item["step"], "loss": item["loss"]})
    return pd.DataFrame(rows)


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def build_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "experiment",
        "n",
        "is_exact_match",
        "is_prover_correct",
        "drs_precision",
        "drs_recall",
        "drs_fscore",
        "predictions",
        "loss_figure",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        line = [
            r.get("experiment", ""),
            str(r.get("n", "")),
            format_float(r.get("is_exact_match")),
            format_float(r.get("is_prover_correct")),
            format_float(r.get("drs_precision")),
            format_float(r.get("drs_recall")),
            format_float(r.get("drs_fscore")),
            r.get("predictions", ""),
            r.get("loss_figure", ""),
        ]
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines) + "\n"


def build_group_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    # Group by experiment name without trailing _seed\d+
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        name = r.get("experiment", "")
        base = re.sub(r"_seed\d+$", "", name)
        group_map.setdefault(base, []).append(r)

    metrics_keys = [
        "is_exact_match",
        "is_prover_correct",
        "drs_precision",
        "drs_recall",
        "drs_fscore",
    ]

    lines: List[str] = []
    for base, items in sorted(group_map.items()):
        # skip groups with single item (still show mean=that value, std=0)
        lines.append(f"\n## {base} (group summary)\n")
        # build table header
        lines.append("| metric | mean | std | count |")
        lines.append("| --- | --- | --- | --- |")
        for k in metrics_keys:
            vals: List[float] = []
            for it in items:
                v = it.get(k)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                mean_v = sum(vals) / len(vals)
                # std (population std)
                if len(vals) > 1:
                    var = sum((x - mean_v) ** 2 for x in vals) / len(vals)
                    std_v = var ** 0.5
                else:
                    std_v = 0.0
                lines.append(
                    f"| {k} | {mean_v:.6f} | {std_v:.6f} | {len(vals)} |"
                )
            else:
                lines.append(f"| {k} |  |  | 0 |")
    lines.append("")
    return "\n".join(lines)


def plot_and_save_loss(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        return
    ensure_parent_dir(out_png)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=df, x="step", y="loss")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment directories like results/exp_fol_t5mini_ep50",
    )
    parser.add_argument(
        "--output-md",
        default="results/metrics_summary.md",
        help="Output Markdown file path",
    )
    parser.add_argument(
        "--loss-fig-name",
        default="training_loss.png",
        help="Filename for loss figure saved under each experiment (in figures/)",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for exp_dir in args.experiments:
        exp_dir = os.path.normpath(exp_dir)
        exp_name = os.path.basename(exp_dir)
        pred_path = find_predictions_tsv(exp_dir)
        metrics: Dict[str, Any] = {
            "experiment": exp_name,
            "n": "",
            "is_exact_match": None,
            "is_prover_correct": None,
            "drs_precision": None,
            "drs_recall": None,
            "drs_fscore": None,
            "predictions": pred_path or "",
            "loss_figure": "",
        }
        if pred_path and os.path.isfile(pred_path):
            try:
                m = compute_metrics_from_tsv(pred_path)
                metrics.update({
                    "n": m.get("n", ""),
                    "is_exact_match": m.get("is_exact_match"),
                    "is_prover_correct": m.get("is_prover_correct"),
                    "drs_precision": m.get("drs_precision"),
                    "drs_recall": m.get("drs_recall"),
                    "drs_fscore": m.get("drs_fscore"),
                })
            except Exception:
                pass
        ts_path = find_trainer_state_json(exp_dir)
        if ts_path and os.path.isfile(ts_path):
            try:
                loss_df = extract_loss_history(ts_path)
                out_png = os.path.join(exp_dir, "figures", args.loss_fig_name)
                plot_and_save_loss(loss_df, out_png)
                metrics["loss_figure"] = out_png
            except Exception:
                pass
        rows.append(metrics)

    ensure_parent_dir(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Metrics Summary\n\n")
        f.write(build_markdown_table(rows))
        f.write("\n")
        f.write(build_group_summary(rows))

    print(f"Saved summary to: {args.output_md}")
    for r in rows:
        if r.get("loss_figure"):
            print(f"Saved loss figure: {r['loss_figure']}")


if __name__ == "__main__":
    main() 