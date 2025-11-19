import unicodedata
import re
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, min_text_length: int = 1, max_text_length: int = 512):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    def normalize_text(self, text: Any) -> str:
        """Normalize text with lowercase conversion and whitespace cleanup."""
        if text is None or pd.isna(text):
            return ""

        if not isinstance(text, str):
            return ""

        normalized = unicodedata.normalize("NFKC", text)

        normalized = normalized.lower()

        normalized = re.sub(r"–", "-", normalized)  # en dash to hyphen
        normalized = re.sub(r"—", "-", normalized)  # em dash to hyphen
        normalized = re.sub(r"…", "...", normalized)  # ellipsis to three dots

        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def is_valid_formula(self, formula: Any) -> bool:
        """Check if formula is valid (only rejects obviously invalid cases)."""
        if formula is None or pd.isna(formula):
            return False

        if not isinstance(formula, str):
            return False

        return bool(formula.strip())

    def clean_data(
        self, df: pd.DataFrame, excluded_file_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Clean data with text normalization and formula preservation."""
        if df.empty:
            return df.copy()

        cleaned_df = df.copy()
        excluded_rows = []

        for idx, row in df.iterrows():
            text = row.get("text", "")
            formula = row.get("formula", "")

            if not self.is_valid_formula(formula):
                row_dict = row.to_dict()
                row_dict["exclusion_reason"] = "invalid_formula"
                excluded_rows.append(row_dict)
                continue

            if text == "" or text is None or pd.isna(text):
                row_dict = row.to_dict()
                row_dict["exclusion_reason"] = "empty_text"
                excluded_rows.append(row_dict)
                continue

            normalized_text = self.normalize_text(text)

            cleaned_df.at[idx, "text"] = normalized_text

        if excluded_rows:
            excluded_ids = [row["id"] for row in excluded_rows]
            cleaned_df = cleaned_df[~cleaned_df["id"].isin(excluded_ids)]

        if excluded_file_path and excluded_rows:
            excluded_df = pd.DataFrame(excluded_rows)
            excluded_df.to_csv(excluded_file_path, index=False)

        return cleaned_df.reset_index(drop=True)

    def preprocess_dataset(
        self, df: pd.DataFrame, excluded_file_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess dataset and return statistics."""
        original_count = len(df)
        original_texts = df["text"].tolist() if "text" in df.columns else []
        original_formulas = df["formula"].tolist() if "formula" in df.columns else []

        cleaned_df = self.clean_data(df, excluded_file_path)

        processed_count = len(cleaned_df)
        removed_count = original_count - processed_count
        retention_rate = processed_count / original_count if original_count > 0 else 0.0

        text_changes = 0
        if "text" in cleaned_df.columns and original_texts:
            for i, (orig, new) in enumerate(
                zip(original_texts[:processed_count], cleaned_df["text"])
            ):
                if self.normalize_text(orig) != orig:
                    text_changes += 1

        formula_changes = 0

        stats = {
            "original_count": original_count,
            "processed_count": processed_count,
            "removed_count": removed_count,
            "retention_rate": retention_rate,
            "text_changes": text_changes,
            "formula_changes": formula_changes,
        }

        logger.info(
            f"Preprocessing completed: {processed_count}/{original_count} samples retained ({retention_rate:.2%})"
        )

        return cleaned_df, stats


def preprocess_formula_data(
    data: pd.DataFrame, min_text_length: int = 1, max_text_length: int = 512
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Preprocess formula data from DataFrame."""
    preprocessor = DataPreprocessor(min_text_length, max_text_length)

    data_list = data.to_dict("records")
    processed_data = []
    original_count = len(data_list)

    exclusion_stats = {
        "empty_text": 0,
        "invalid_formula": 0,
        "text_too_long": 0,
        "text_too_short": 0,
    }

    for item in data_list:
        text = item.get("text", "")
        formula = item.get("formula", "")

        normalized_text = preprocessor.normalize_text(text)

        if len(normalized_text) < min_text_length:
            if not normalized_text:
                exclusion_stats["empty_text"] += 1
            else:
                exclusion_stats["text_too_short"] += 1
            continue

        if len(normalized_text) > max_text_length:
            exclusion_stats["text_too_long"] += 1
            continue

        if not preprocessor.is_valid_formula(formula):
            exclusion_stats["invalid_formula"] += 1
            continue

        processed_item = item.copy()
        processed_item["text"] = normalized_text
        processed_item["formula"] = (
            formula.strip() if isinstance(formula, str) else formula
        )
        processed_data.append(processed_item)

    result_df = pd.DataFrame(processed_data)

    processed_count = len(result_df)
    removed_count = original_count - processed_count

    stats = {
        "original_count": original_count,
        "processed_count": processed_count,
        "removed_count": removed_count,
        **exclusion_stats,
    }

    return result_df, stats
