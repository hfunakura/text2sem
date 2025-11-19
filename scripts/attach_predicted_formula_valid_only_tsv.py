import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_csv",
        default="./ccg2lambda/data/sick_test.csv",
    )
    parser.add_argument(
        "--report_tsv",
        default="./results/exp_fol_t5base_ep50_seed5/predictions/detailed_report.tsv",
    )
    parser.add_argument(
        "--output_tsv",
        default="./results/exp_fol_t5base_ep50_seed5/predictions/sick_test_master_valid_with_predicted_formula.tsv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    test_csv_path = Path(args.test_csv)
    report_tsv_path = Path(args.report_tsv)
    output_tsv_path = Path(args.output_tsv)

    if not test_csv_path.exists():
        print(f"Input test CSV not found: {test_csv_path}", file=sys.stderr)
        return 1
    if not report_tsv_path.exists():
        print(f"Input report TSV not found: {report_tsv_path}", file=sys.stderr)
        return 1

    test_df = pd.read_csv(test_csv_path, dtype=str)
    report_df = pd.read_csv(report_tsv_path, sep="\t", dtype=str)

    if "id" not in test_df.columns or "correctness" not in test_df.columns:
        print("Columns 'id' or 'correctness' missing in test CSV", file=sys.stderr)
        return 1
    if "id" not in report_df.columns or "predicted_formula" not in report_df.columns:
        print("Required columns missing in report TSV ('id', 'predicted_formula')", file=sys.stderr)
        return 1

    filtered_df = test_df[test_df["correctness"] == "valid"].copy()

    joined_df = filtered_df.merge(
        report_df[["id", "predicted_formula"]], how="left", on="id"
    )

    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_csv(output_tsv_path, index=False, sep="\t")

    print(f"Wrote: {output_tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 