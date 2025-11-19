"""
Evaluator for semantic parsing models.

This module provides comprehensive evaluation functionality for semantic parsing
models, including detailed analysis and comparison capabilities.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import torch
from tqdm import tqdm
import re

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.semantic_parser import T5SemanticParser
from src.training.metrics import MetricsCalculator, EvaluationResult, METRICS_REGISTRY
from src.utils.config import load_config, Config
from src.models.model_config import ModelConfig


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""

    pass

def generate_predictions_for_csv(
    parser: T5SemanticParser, texts: List[str], batch_size: int, device: str
) -> List[str]:
    """Generate predictions for a list of texts"""
    predictions = []

    with tqdm(
        total=len(texts) // batch_size + (1 if len(texts) % batch_size else 0),
        desc="Generating predictions",
    ) as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_predictions = parser.generate_batch(batch_texts)

            predictions.extend(batch_predictions)
            pbar.update(1)

    return predictions


def evaluate_checkpoint_with_csv(
    main_config: "Config",
    checkpoint_path: str,
    csv_path: str,
    batch_size: int,
    device: str,
    save_detailed_report: Optional[str] = None,
):
    """
    Evaluates a model checkpoint against a single specific CSV file.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use generic model config compatible with semantic_parser
    model_conf = ModelConfig()  # default values
    parser = T5SemanticParser.load_model(Path(checkpoint_path), model_conf)
    parser.to(device)
    parser.eval()

    # Load data
    df = pd.read_csv(csv_path)
    texts = df["text"].fillna("").astype(str).tolist()
    gold_formulas = df["formula"].fillna("").astype(str).tolist()
    dataset_name = Path(csv_path).stem

    # Generate predictions
    predictions = generate_predictions_for_csv(parser, texts, batch_size, device)

    # --- Evaluation ---
    metrics_to_run = ["exact_match", "prover", "drs_match"]
    print(
        f"--- Evaluating against {Path(csv_path).name} (Metrics: {', '.join(metrics_to_run)}) ---"
    )

    calculator = MetricsCalculator()
    metric_args = {
        "prover": {
            "vampire_path": main_config.evaluation.dependencies.get("vampire_path")
        },
        "drs_match": {
            "drs_parsing_path": main_config.evaluation.dependencies.get(
                "drs_parsing_path"
            )
        },
    }
    for metric_name in metrics_to_run:
        args = metric_args.get(metric_name, {})
        args = {k: v for k, v in args.items() if v is not None}
        metric_instance = METRICS_REGISTRY.create_metric(metric_name, **args)
        calculator.add_metric(metric_name, metric_instance)

    metrics, per_sample_details = calculator.compute_metrics(
        predictions=predictions,
        references=gold_formulas,
        dataset=dataset_name,
        metrics=metrics_to_run,
    )

    stats = calculator.get_stats()
    stats["num_samples"] = len(predictions)

    result = EvaluationResult(metrics=metrics, stats=stats, dataset=dataset_name)

    print("\nDataset:", result.dataset)
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.4f}")

    # --- Detailed Report (Optional) ---
    if save_detailed_report:
        try:
            detailed_df = df.copy()
            # More robust cleaning: remove all whitespace including newlines and tabs
            predictions_clean = []
            for pred in predictions:
                if isinstance(pred, str):
                    # Remove all types of whitespace and normalize to single space
                    cleaned = " ".join(pred.split())
                else:
                    cleaned = str(pred)
                predictions_clean.append(cleaned)
            detailed_df["predicted_formula"] = predictions_clean

            if "exact_match" in per_sample_details:
                detailed_df["is_exact_match"] = per_sample_details["exact_match"]
            if "prover" in per_sample_details:
                detailed_df["is_prover_correct"] = per_sample_details["prover"]
            if "drs_match" in per_sample_details:
                drs_results = per_sample_details["drs_match"]
                detailed_df["drs_precision"] = [
                    res.get("precision", 0.0) for res in drs_results
                ]
                detailed_df["drs_recall"] = [
                    res.get("recall", 0.0) for res in drs_results
                ]
                detailed_df["drs_fscore"] = [
                    res.get("fscore", 0.0) for res in drs_results
                ]

            report_path = Path(save_detailed_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            # Use TSV format to avoid CSV structure issues with complex formulas
            detailed_df.to_csv(report_path, index=False, encoding="utf-8", sep="\t")
            print(f"\nDetailed evaluation report saved to: {report_path}")

        except Exception as e:
            print(f"Error saving detailed report: {e}")

    # --- Save Overall Report ---
    output_dir = Path(main_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"{dataset_name}_evaluation_report.json"
    result.save_json(report_file)
    print(f"\nEvaluation report saved to: {report_file}")


def main():
    """Main function to run the evaluation from command line."""
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the specific CSV file for evaluation.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (e.g., 'cuda', 'cuda:0', or 'cpu')",
    )
    parser.add_argument(
        "--save-detailed-report",
        type=str,
        default=None,
        help="Path to save the detailed per-sample report (TSV format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the main configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    main_config = load_config(args.config)

    print("=" * 60)
    print("AUTOMATED CHECKPOINT EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CSV File: {args.csv}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output Directory: {main_config.evaluation.output_dir}")
    print("=" * 60)

    try:
        evaluate_checkpoint_with_csv(
            checkpoint_path=args.checkpoint,
            csv_path=args.csv,
            batch_size=args.batch_size,
            device=args.device,
            save_detailed_report=args.save_detailed_report,
            main_config=main_config,
        )
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
