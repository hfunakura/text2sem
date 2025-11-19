"""
Metrics for semantic parsing evaluation.

This module provides a pluggable and extensible evaluation metrics system
for semantic parsing tasks, supporting dynamic metric registration and computation.
"""

import json
import re
import time
import threading
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Type, Tuple
from dataclasses import dataclass, field
import csv
import random
import math
import subprocess
import tempfile
from nltk.sem.logic import Expression
import pandas as pd


class MetricsError(Exception):
    """Exception raised for metrics-related errors"""

    pass


@dataclass
class EvaluationResult:
    """Container for evaluation results"""

    metrics: Dict[str, float]
    stats: Dict[str, Any]
    dataset: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(
            {
                "metrics": self.metrics,
                "stats": self.stats,
                "dataset": self.dataset,
                "timestamp": self.timestamp,
            },
            indent=2,
        )

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save result to JSON file"""
        file_path = Path(file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        except PermissionError:
            raise MetricsError("Permission denied: Cannot write to file")
        except Exception as e:
            raise MetricsError(f"Error saving JSON: {e}")

    def save_csv(self, file_path: Union[str, Path]) -> None:
        """Save result to CSV file"""
        file_path = Path(file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                for metric, value in self.metrics.items():
                    writer.writerow([metric, value])

                for stat, value in self.stats.items():
                    writer.writerow([f"stat_{stat}", value])

        except PermissionError:
            raise MetricsError("Permission denied: Cannot write to file")
        except Exception as e:
            raise MetricsError(f"Error saving CSV: {e}")

    def is_better_than(
        self, other: "EvaluationResult", metric: str = "exact_match"
    ) -> bool:
        """Compare with another result"""
        if metric not in self.metrics or metric not in other.metrics:
            raise MetricsError(f"Metric '{metric}' not found in results")
        return self.metrics[metric] > other.metrics[metric]


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics"""

    def __init__(self, name: str, **kwargs):
        """
        Initialize base metric.

        Args:
            name: Name of the metric
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.reset()

    @abstractmethod
    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[Union[float, Dict[str, float]], List[Any]]:
        """
        Compute the metric score and per-sample results.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            A tuple containing:
            - The aggregate metric score (float or Dict[str, float])
            - A list of per-sample results.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for fresh computation"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the metric computation"""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class DrsMatchMetric(BaseMetric):
    """DRS Match metric based on D-match score"""

    def __init__(self, drs_parsing_path: str, **kwargs):
        """
        Initialize DrsMatchMetric.

        Args:
            drs_parsing_path: Path to DRS_parsing directory
        """
        super().__init__("drs_match", **kwargs)

        drs_path = Path(drs_parsing_path)
        if not drs_path.exists():
            print(f"Warning: DRS_parsing path not found: {drs_parsing_path}")
            print("DRS evaluation will return 0.0 scores")
            self.drs_parsing_available = False
        else:
            self.drs_parsing_available = True

        self.drs_parsing_path = drs_parsing_path

        ccg2lambda_scripts_path = str(
            Path(__file__).parent.parent.parent / "ccg2lambda" / "scripts"
        )
        sys.path.insert(0, ccg2lambda_scripts_path)
        try:
            from nltk2drs import convert_to_drs
            from drs2clf import convert_to_clausal_forms
            from nltk.sem.logic import Expression

            self.convert_to_drs = convert_to_drs
            self.convert_to_clausal_forms = convert_to_clausal_forms
            self.lexpr = Expression.fromstring
        except ImportError as e:
            print(f"Warning: Could not import DRS conversion functions: {e}")
            self.drs_parsing_available = False
        finally:
            sys.path.pop(0)

        self.reset()

    def reset(self) -> None:
        """Reset metric state"""
        self._total_samples = 0
        self._precision_sum = 0.0
        self._recall_sum = 0.0
        self._fscore_sum = 0.0
        self._lock = threading.Lock()

    def _is_clf(self, formula_str: str) -> bool:
        """Heuristically judge whether the given string already represents a
        clausal form (CLF). If so, we can skip DRS/FOL conversion.
        This is a lightweight check based on typical patterns such as
        box variable prefixes (b1, b2, …) and clause keywords like
        ' REF ', ' IMP ', etc.
        """
        if not formula_str or pd.isna(formula_str):
            return False
        s = formula_str.strip()
        if re.match(r"^[bB]\d+\s", s):
            return True
        clause_tokens = [" REF ", " IMP ", " NOT ", " NEGATION ", " ATTR "]
        return any(tok in s for tok in clause_tokens)

    def _formula_to_cf_str(self, formula_str: str) -> str:
        """DRS/FOL文字列をClausal Form文字列に変換"""
        if not formula_str or pd.isna(formula_str):
            return ""
        try:
            if re.match(r"\[[a-zA-Z0-9, ]*\]", formula_str.strip()):
                drs_expr = self.dp(formula_str)
            else:
                drs_expr = self.convert_to_drs(self.lexpr(formula_str))

            clausal_forms = self.convert_to_clausal_forms(drs_expr)
            return "\n".join(clausal_forms)
        except Exception:
            return ""

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        if not predictions:
            return {
                "drs_match_precision": 0.0,
                "drs_match_recall": 0.0,
                "drs_match_fscore": 0.0,
            }, []

        if len(predictions) != len(references):
            raise MetricsError(
                f"Length mismatch: predictions={len(predictions)}, references={len(references)}"
            )

        if not self.drs_parsing_available:
            num_samples = len(predictions)
            per_sample_results = [
                {"precision": 0.0, "recall": 0.0, "fscore": 0.0}
                for _ in range(num_samples)
            ]

            with self._lock:
                self._total_samples += num_samples
                self._precision_sum += 0.0
                self._recall_sum += 0.0
                self._fscore_sum += 0.0

            return {
                "drs_match_precision": 0.0,
                "drs_match_recall": 0.0,
                "drs_match_fscore": 0.0,
            }, per_sample_results

        num_samples = len(predictions)
        total_precision = 0.0
        total_recall = 0.0
        total_fscore = 0.0
        per_sample_results = []

        for pred_str, ref_str in zip(predictions, references):
            precision = 0.0
            recall = 0.0
            fscore = 0.0

            if self._is_clf(pred_str) and self._is_clf(ref_str):
                def _split_clf(s: str):
                    if "\n" in s:
                        return [cl.strip() for cl in s.split("\n") if cl.strip()]
                    if " SEP " in s:
                        return [cl.strip() for cl in s.split(" SEP ") if cl.strip()]
                    return [s.strip()]

                pred_clf = _split_clf(pred_str)
                ref_clf = _split_clf(ref_str)

                precision, recall, fscore = self._calculate_dmatch_scores(
                    pred_clf, ref_clf
                )
            else:
                try:
                    pred_expr = self.lexpr(pred_str)
                    ref_expr = self.lexpr(ref_str)

                    pred_drs = self.convert_to_drs(pred_expr)
                    ref_drs = self.convert_to_drs(ref_expr)

                    if pred_drs and ref_drs:
                        pred_clf = self.convert_to_clausal_forms(pred_drs)
                        ref_clf = self.convert_to_clausal_forms(ref_drs)

                        if pred_clf and ref_clf:
                            precision, recall, fscore = self._calculate_dmatch_scores(
                                pred_clf, ref_clf
                            )
                        else:
                            precision = recall = fscore = 0.0
                    else:
                        precision = recall = fscore = 0.0

                except Exception:
                    precision = recall = fscore = 0.0

            total_precision += precision
            total_recall += recall
            total_fscore += fscore

            per_sample_results.append(
                {"precision": precision, "recall": recall, "fscore": fscore}
            )

        agg_scores = {
            "drs_match_precision": total_precision / num_samples
            if num_samples > 0
            else 0.0,
            "drs_match_recall": total_recall / num_samples if num_samples > 0 else 0.0,
            "drs_match_fscore": total_fscore / num_samples if num_samples > 0 else 0.0,
        }

        with self._lock:
            self._total_samples += num_samples
            self._precision_sum += total_precision
            self._recall_sum += total_recall
            self._fscore_sum += total_fscore

        return agg_scores, per_sample_results

    def _calculate_dmatch_scores(
        self, pred_clf: list, ref_clf: list
    ) -> tuple[float, float, float]:
        """
        Calculate D-match precision, recall, and F-score for clausal forms.

        Args:
            pred_clf: Predicted clausal form list
            ref_clf: Reference clausal form list

        Returns:
            Tuple of (precision, recall, fscore)
        """
        try:
            pred_clauses = self._parse_clausal_form_list(pred_clf)
            ref_clauses = self._parse_clausal_form_list(ref_clf)

            if not pred_clauses and not ref_clauses:
                return 0.0, 0.0, 0.0  # Both empty
            elif not pred_clauses:
                return 0.0, 0.0, 0.0  # No prediction clauses
            elif not ref_clauses:
                return 0.0, 0.0, 0.0  # No reference clauses

            matching_clauses = pred_clauses.intersection(ref_clauses)

            precision = (
                len(matching_clauses) / len(pred_clauses) if pred_clauses else 0.0
            )
            recall = len(matching_clauses) / len(ref_clauses) if ref_clauses else 0.0

            if precision + recall > 0:
                fscore = 2 * precision * recall / (precision + recall)
            else:
                fscore = 0.0

            return precision, recall, fscore

        except Exception:
            return 0.0, 0.0, 0.0

    def _parse_clausal_form_list(self, clf_list: list) -> set:
        """
        Parse a clausal form list into a set of normalized clauses.

        Args:
            clf_list: List of clausal form strings

        Returns:
            Set of normalized clause strings
        """
        if not clf_list or not isinstance(clf_list, list):
            return set()

        try:
            clauses = set()
            for clf_item in clf_list:
                if isinstance(clf_item, str):
                    lines = [
                        line.strip() for line in clf_item.split("\n") if line.strip()
                    ]
                    for line in lines:
                        if (
                            line
                            and not line.startswith("%")
                            and not line.startswith("#")
                        ):
                            normalized = " ".join(line.split())
                            if normalized:
                                clauses.add(normalized)
                else:
                    normalized = " ".join(str(clf_item).split())
                    if normalized:
                        clauses.add(normalized)

            return clauses

        except Exception:
            return set()

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "average_precision": self._precision_sum / self._total_samples
                if self._total_samples > 0
                else 0.0,
                "average_recall": self._recall_sum / self._total_samples
                if self._total_samples > 0
                else 0.0,
                "average_fscore": self._fscore_sum / self._total_samples
                if self._total_samples > 0
                else 0.0,
            }


class ProverMetric(BaseMetric):
    """Theorem Prover based metric"""

    def __init__(self, vampire_path: str, **kwargs):
        """
        Initialize ProverMetric.

        Args:
            vampire_path: Path to vampire executable
        """
        super().__init__("prover", **kwargs)
        if not Path(vampire_path).exists() or not Path(vampire_path).is_file():
            raise MetricsError(f"Vampire executable not found: {vampire_path}")
        self.vampire_path = vampire_path

        ccg2lambda_scripts_path = str(
            Path(__file__).parent.parent.parent / "ccg2lambda" / "scripts"
        )
        sys.path.insert(0, ccg2lambda_scripts_path)
        try:
            from nltk2tptp import convert_to_tptp_proof

            self.convert_to_tptp_proof = convert_to_tptp_proof
        finally:
            sys.path.pop(0)

        self.lexpr = Expression.fromstring
        self.reset()

    def reset(self) -> None:
        """Reset metric state"""
        self._total_samples = 0
        self._correct_proofs = 0
        self._lock = threading.Lock()

    def _is_theorem(self, premises, conclusion) -> bool:
        """Vampireを使って含意関係が証明できるか判定する"""
        if premises is None or conclusion is None:
            return False

        temp_dir = tempfile.mkdtemp(prefix="vampire_")
        try:
            inference = [premises, conclusion]
            fols = self.convert_to_tptp_proof(inference)

            tptp_file_path = Path(temp_dir) / "problem.p"
            with open(tptp_file_path, "w", encoding="utf-8") as f:
                for fol in fols:
                    f.write(fol + "\n")

            command = [
                self.vampire_path,
                "-t",
                "7",
                "--mode",
                "casc",
                str(tptp_file_path),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )

            return "% Refutation found. Thanks to Tanya!" in result.stdout
        except Exception:
            return False

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[Dict[str, float], List[bool]]:
        if not predictions:
            return {"prover_accuracy": 0.0}, []

        if len(predictions) != len(references):
            raise MetricsError(
                f"Length mismatch: predictions={len(predictions)}, references={len(references)}"
            )

        correct = 0
        total = len(predictions)
        per_sample_results = []

        for pred_str, ref_str in zip(predictions, references):
            is_correct = False
            try:
                p_expr = self.lexpr(pred_str)
                g_expr = self.lexpr(ref_str)

                if self._is_theorem(g_expr, p_expr) and self._is_theorem(
                    p_expr, g_expr
                ):
                    correct += 1
                    is_correct = True
            except Exception:
                try:
                    pred_fixed = pred_str.strip()
                    ref_fixed = ref_str.strip()

                    for formula_str in [pred_fixed, ref_fixed]:
                        open_count = formula_str.count("(")
                        close_count = formula_str.count(")")
                        if close_count > open_count:
                            extra_close = close_count - open_count
                            formula_str = formula_str.rstrip(")" * extra_close)
                        elif open_count > close_count:
                            extra_open = open_count - close_count
                            formula_str += ")" * extra_open

                    p_expr = self.lexpr(pred_fixed)
                    g_expr = self.lexpr(ref_fixed)

                    if self._is_theorem(g_expr, p_expr) and self._is_theorem(
                        p_expr, g_expr
                    ):
                        correct += 1
                        is_correct = True
                except Exception:
                    is_correct = False

            per_sample_results.append(is_correct)

        with self._lock:
            self._total_samples += total
            self._correct_proofs += correct

        accuracy = correct / total if total > 0 else 0.0
        return {"prover_accuracy": accuracy}, per_sample_results

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "correct_proofs": self._correct_proofs,
            }


class ExactMatchMetric(BaseMetric):
    """Exact Match metric for semantic parsing evaluation"""

    def __init__(
        self, normalize_whitespace: bool = True, case_sensitive: bool = True, **kwargs
    ):
        """
        Initialize ExactMatchMetric.

        Args:
            normalize_whitespace: Whether to normalize whitespace
            case_sensitive: Whether comparison is case sensitive
        """
        super().__init__("exact_match", **kwargs)
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive
        self.reset()

    def reset(self) -> None:
        """Reset metric state"""
        self._total_samples = 0
        self._correct_samples = 0
        self._lock = threading.Lock()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (minimal: strip only, optional case-fold)"""
        if text is None:
            return ""

        text = str(text).strip()

        if not self.case_sensitive:
            text = text.lower()

        return text

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[float, List[bool]]:
        """
        Compute Exact Match score.

        Args:
            predictions: List of predicted logical formulas
            references: List of reference logical formulas

        Returns:
            Exact Match score (0.0 to 1.0)
        """
        if not predictions:
            raise MetricsError("Empty predictions list")

        if len(predictions) != len(references):
            raise MetricsError(
                f"Length mismatch: predictions={len(predictions)}, "
                f"references={len(references)}"
            )

        correct = 0
        total = len(predictions)
        per_sample_results = []

        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)

            is_correct = pred_norm == ref_norm
            per_sample_results.append(is_correct)
            if is_correct:
                correct += 1

        with self._lock:
            self._total_samples += total
            self._correct_samples += correct

        return (correct / total if total > 0 else 0.0), per_sample_results

    def get_stats(self) -> Dict[str, int]:
        """Get detailed statistics"""
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "correct_samples": self._correct_samples,
            }


class BLEUMetric(BaseMetric):
    """BLEU score metric (dummy implementation for demonstration)"""

    def __init__(self, n_grams: int = 4, **kwargs):
        """
        Initialize BLEU metric.

        Args:
            n_grams: Maximum n-gram order
        """
        super().__init__("bleu", **kwargs)
        self.n_grams = n_grams
        self.reset()

    def reset(self) -> None:
        """Reset metric state"""
        self._total_samples = 0
        self._total_bleu = 0.0
        self._lock = threading.Lock()

    def _dummy_bleu_score(self, prediction: str, reference: str) -> float:
        """Dummy BLEU computation for demonstration"""
        pred_tokens = str(prediction).split()
        ref_tokens = str(reference).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        overlap = len(set(pred_tokens) & set(ref_tokens))
        total = max(len(pred_tokens), len(ref_tokens))

        base_score = overlap / total if total > 0 else 0.0
        noise = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_score + noise))

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[float, List[float]]:
        """
        Compute BLEU score.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            Average BLEU score (0.0 to 1.0)
        """
        if not predictions:
            raise MetricsError("Empty predictions list")

        if len(predictions) != len(references):
            raise MetricsError(
                f"Length mismatch: predictions={len(predictions)}, "
                f"references={len(references)}"
            )

        with self._lock:
            total_bleu = 0.0
            total = len(predictions)
            per_sample_results = []

            for pred, ref in zip(predictions, references):
                bleu_score = self._dummy_bleu_score(pred, ref)
                total_bleu += bleu_score
                per_sample_results.append(bleu_score)

            self._total_samples += total
            self._total_bleu += total_bleu

            return total_bleu / total if total > 0 else 0.0, per_sample_results

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "total_bleu": self._total_bleu,
                "n_grams": self.n_grams,
            }


class ROUGEMetric(BaseMetric):
    """ROUGE score metric (dummy implementation for demonstration)"""

    def __init__(self, rouge_type: str = "rouge-l", **kwargs):
        """
        Initialize ROUGE metric.

        Args:
            rouge_type: Type of ROUGE metric (rouge-1, rouge-2, rouge-l)
        """
        super().__init__("rouge", **kwargs)
        self.rouge_type = rouge_type
        self.reset()

    def reset(self) -> None:
        """Reset metric state"""
        self._total_samples = 0
        self._total_rouge = 0.0
        self._lock = threading.Lock()

    def _dummy_rouge_score(self, prediction: str, reference: str) -> float:
        """Dummy ROUGE computation for demonstration"""
        pred_str = str(prediction).lower()
        ref_str = str(reference).lower()

        if not pred_str or not ref_str:
            return 0.0

        common_chars = sum(1 for c in pred_str if c in ref_str)
        total_chars = len(pred_str) + len(ref_str)

        if total_chars == 0:
            return 0.0

        precision = common_chars / len(pred_str) if pred_str else 0.0
        recall = common_chars / len(ref_str) if ref_str else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        noise = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, f1 + noise))

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[float, List[float]]:
        """
        Compute ROUGE score.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            Average ROUGE score (0.0 to 1.0)
        """
        if not predictions:
            raise MetricsError("Empty predictions list")

        if len(predictions) != len(references):
            raise MetricsError(
                f"Length mismatch: predictions={len(predictions)}, "
                f"references={len(references)}"
            )

        with self._lock:
            total_rouge = 0.0
            total = len(predictions)
            per_sample_results = []

            for pred, ref in zip(predictions, references):
                rouge_score = self._dummy_rouge_score(pred, ref)
                total_rouge += rouge_score
                per_sample_results.append(rouge_score)

            self._total_samples += total
            self._total_rouge += total_rouge

            return total_rouge / total if total > 0 else 0.0, per_sample_results

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "total_rouge": self._total_rouge,
                "rouge_type": self.rouge_type,
            }


class SemanticSimilarityMetric(BaseMetric):
    """Custom semantic similarity metric (dummy implementation for demonstration)"""

    def __init__(self, similarity_threshold: float = 0.8, **kwargs):
        """
        Initialize Semantic Similarity metric.

        Args:
            similarity_threshold: Threshold for considering predictions similar
        """
        super().__init__("semantic_similarity", **kwargs)
        self.similarity_threshold = similarity_threshold
        self.reset()

    def reset(self) -> None:
        """Reset metric state"""
        self._total_samples = 0
        self._similar_samples = 0
        self._total_similarity = 0.0
        self._lock = threading.Lock()

    def _dummy_semantic_similarity(self, prediction: str, reference: str) -> float:
        """Dummy semantic similarity computation for demonstration"""
        pred_str = str(prediction).lower()
        ref_str = str(reference).lower()

        if not pred_str or not ref_str:
            return 0.0

        def edit_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return edit_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        max_len = max(len(pred_str), len(ref_str))
        if max_len == 0:
            return 1.0

        distance = edit_distance(pred_str, ref_str)
        similarity = 1.0 - (distance / max_len)

        similarity = similarity * math.exp(-0.1 * distance / max_len)

        return max(0.0, min(1.0, similarity))

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Tuple[float, List[float]]:
        """
        Compute semantic similarity score.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            Average semantic similarity score (0.0 to 1.0)
        """
        if not predictions:
            raise MetricsError("Empty predictions list")

        if len(predictions) != len(references):
            raise MetricsError(
                f"Length mismatch: predictions={len(predictions)}, "
                f"references={len(references)}"
            )

        with self._lock:
            total_similarity = 0.0
            similar_count = 0
            total = len(predictions)
            per_sample_results = []

            for pred, ref in zip(predictions, references):
                similarity = self._dummy_semantic_similarity(pred, ref)
                total_similarity += similarity
                per_sample_results.append(similarity)

                if similarity >= self.similarity_threshold:
                    similar_count += 1

            self._total_samples += total
            self._similar_samples += similar_count
            self._total_similarity += total_similarity

            return total_similarity / total if total > 0 else 0.0, per_sample_results

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "similar_samples": self._similar_samples,
                "total_similarity": self._total_similarity,
                "similarity_threshold": self.similarity_threshold,
                "similarity_rate": self._similar_samples / self._total_samples
                if self._total_samples > 0
                else 0.0,
            }


class MetricsRegistry:
    """Registry for managing available metrics"""

    def __init__(self):
        """Initialize the metrics registry"""
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._register_default_metrics()

    def _register_default_metrics(self):
        """Register default metrics"""
        self.register("exact_match", ExactMatchMetric)
        self.register("bleu", BLEUMetric)
        self.register("rouge", ROUGEMetric)
        self.register("semantic_similarity", SemanticSimilarityMetric)
        self.register("drs_match", DrsMatchMetric)
        self.register("prover", ProverMetric)

    def register(self, name: str, metric_class: Type[BaseMetric]):
        """
        Register a new metric class.

        Args:
            name: Name of the metric
            metric_class: Metric class that inherits from BaseMetric
        """
        if not issubclass(metric_class, BaseMetric):
            raise MetricsError("Metric class must inherit from BaseMetric")

        self._metrics[name] = metric_class

    def get(self, name: str) -> Type[BaseMetric]:
        """
        Get a metric class by name.

        Args:
            name: Name of the metric

        Returns:
            Metric class
        """
        if name not in self._metrics:
            raise MetricsError(f"Unknown metric: {name}")

        return self._metrics[name]

    def list_metrics(self) -> List[str]:
        """Get list of available metric names"""
        return list(self._metrics.keys())

    def create_metric(self, name: str, **kwargs) -> BaseMetric:
        """
        Create an instance of a metric.

        Args:
            name: Name of the metric
            **kwargs: Configuration parameters for the metric

        Returns:
            Metric instance
        """
        metric_class = self.get(name)
        return metric_class(**kwargs)


METRICS_REGISTRY = MetricsRegistry()


class MetricsCalculator:
    """Enhanced calculator for multiple evaluation metrics with pluggable architecture"""

    def __init__(self, metrics_config: Optional[Dict[str, Dict]] = None):
        """
        Initialize MetricsCalculator with configurable metrics.

        Args:
            metrics_config: Dictionary mapping metric names to their configurations
                           Example: {"exact_match": {"case_sensitive": False}, "bleu": {"n_grams": 4}}
        """
        self.metrics = {}
        self._performance_stats = {}

        if metrics_config is None:
            metrics_config = {"exact_match": {}}

        for metric_name, config in metrics_config.items():
            try:
                metric = METRICS_REGISTRY.create_metric(metric_name, **config)
                self.metrics[metric_name] = metric
            except MetricsError as e:
                raise MetricsError(f"Failed to initialize metric '{metric_name}': {e}")

    def add_metric(self, name: str, metric: BaseMetric):
        """
        Add a metric instance to the calculator.

        Args:
            name: Name to register the metric under
            metric: Metric instance
        """
        self.metrics[name] = metric

    def remove_metric(self, name: str):
        """
        Remove a metric from the calculator.

        Args:
            name: Name of the metric to remove
        """
        if name in self.metrics:
            del self.metrics[name]

    def compute_metrics(
        self,
        predictions: List[Any],
        references: List[Any],
        dataset: str = "unknown",
        metrics: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, List[Any]]]:
        """
        Compute all configured metrics.

        Args:
            predictions: List of predicted outputs.
            references: List of reference outputs.
            dataset: Name of the dataset being evaluated.
            metrics: Optional list of metric names to compute. If None, all are computed.

        Returns:
            A tuple containing:
            - A dictionary of aggregate metric names to scores.
            - A dictionary of metric names to lists of per-sample results.
        """
        start_time = time.time()

        computed_metrics: Dict[str, float] = {}
        per_sample_details: Dict[str, List[Any]] = {}

        metrics_to_run = metrics or self.metrics.keys()

        for name in metrics_to_run:
            if name not in self.metrics:
                print(f"Warning: Metric '{name}' not found, skipping.")
                continue

            metric = self.metrics[name]
            try:
                agg_score, per_sample_results = metric.compute(predictions, references)
                per_sample_details[name] = per_sample_results

                if isinstance(agg_score, dict):
                    computed_metrics.update(agg_score)
                else:
                    computed_metrics[name] = agg_score

            except Exception as e:
                print(f"Error computing metric {name}: {e}")

        end_time = time.time()
        self._performance_stats["total_time_seconds"] = end_time - start_time
        self._performance_stats["samples_per_second"] = (
            len(predictions) / (end_time - start_time) if end_time > start_time else 0
        )

        return computed_metrics, per_sample_details

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all metrics"""
        stats = {}
        for metric_name, metric in self.metrics.items():
            metric_stats = metric.get_stats()
            for key, value in metric_stats.items():
                stats[f"{metric_name}_{key}"] = value

        if "exact_match_total_samples" in stats:
            stats["total_samples"] = stats["exact_match_total_samples"]
            stats["correct_samples"] = stats["exact_match_correct_samples"]
        elif stats:
            first_metric = next(iter(self.metrics.keys()))
            if f"{first_metric}_total_samples" in stats:
                stats["total_samples"] = stats[f"{first_metric}_total_samples"]

        return stats

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self._performance_stats.copy()

    def reset(self) -> None:
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()
        self._performance_stats = {}
        self._performance_stats["total_time_seconds"] = 0

    def list_available_metrics(self) -> List[str]:
        """Get list of metrics available in the registry"""
        return METRICS_REGISTRY.list_metrics()

    def list_active_metrics(self) -> List[str]:
        """Get list of currently active metrics in this calculator"""
        return list(self.metrics.keys())


def compute_exact_match(
    predictions: List[str],
    references: List[str],
    normalize_whitespace: bool = True,
    case_sensitive: bool = True,
) -> float:
    """
    Convenience function to compute Exact Match score.

    Args:
        predictions: List of predicted logical formulas
        references: List of reference logical formulas
        normalize_whitespace: Whether to normalize whitespace
        case_sensitive: Whether comparison is case sensitive

    Returns:
        Exact Match score (0.0 to 1.0)
    """
    metric = ExactMatchMetric(
        normalize_whitespace=normalize_whitespace, case_sensitive=case_sensitive
    )
    agg_score, _ = metric.compute(predictions, references)
    return agg_score


def evaluate_dataset(
    predictions: List[str],
    references: List[str],
    dataset_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    metrics_config: Optional[Dict[str, Dict]] = None,
) -> EvaluationResult:
    """
    Evaluate a dataset and return comprehensive results.

    Args:
        predictions: List of predicted logical formulas
        references: List of reference logical formulas
        dataset_name: Name of the dataset
        output_dir: Directory to save results (optional)
        metrics_config: Configuration for metrics (optional)

    Returns:
        EvaluationResult object
    """
    calculator = MetricsCalculator(metrics_config)

    metrics, _ = calculator.compute_metrics(predictions, references, dataset_name)
    stats = calculator.get_stats()

    result = EvaluationResult(metrics=metrics, stats=stats, dataset=dataset_name)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_file = output_dir / f"{dataset_name}_results.json"
        result.save_json(json_file)

        csv_file = output_dir / f"{dataset_name}_results.csv"
        result.save_csv(csv_file)

    return result
