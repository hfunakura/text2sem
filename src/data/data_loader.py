"""CSV data loader for Neural Semantic Parsing project.

This module provides functionality to load and combine multiple CSV files
containing semantic parsing data.
"""

from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class CSVDataLoader:
    """CSV data loader for semantic parsing datasets."""

    def __init__(
        self,
        input_column: str = "text",
        target_column: str = "formula",
        output_dir: str = "results/outputs",
    ):
        """Initialize the CSV data loader.

        Args:
            input_column: The name of the input text column.
            target_column: The name of the target formula column.
            output_dir: Directory to save logs for dropped rows.
        """
        self.input_column = input_column
        self.target_column = target_column
        self.required_columns = ["id", self.input_column, self.target_column]
        self.output_dir = Path(output_dir)

    def load_single_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a single CSV file, dropping rows with missing essential data.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {file_path}: {e}")

        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file_path}: {missing_columns}"
            )

        missing_mask = df[[self.input_column, self.target_column]].isnull().any(axis=1)
        if missing_mask.any():
            dropped_rows = df[missing_mask]
            dropped_ids = dropped_rows["id"].tolist()

            logger.warning(
                f"Found {len(dropped_ids)} rows with missing data in '{self.input_column}' or "
                f"'{self.target_column}' in {file_path}. These rows will be dropped."
            )

            self.output_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.output_dir / f"{Path(file_path).stem}_dropped_ids.log"
            with open(log_file, "a", encoding="utf-8") as f:
                for row_id in dropped_ids:
                    f.write(f"{row_id}\n")

            df = df.dropna(subset=[self.input_column, self.target_column])

        df = df.astype(
            {"id": "str", self.input_column: "str", self.target_column: "str"}
        )

        if df[self.required_columns].isnull().any().any():
            logger.warning(f"Found missing values in {file_path}")

        return df

    def load_multiple_csvs(self, file_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """Load and combine multiple CSV files.

        Args:
            file_paths: List of paths to CSV files

        Returns:
            Combined DataFrame

        Raises:
            ValueError: If no files provided or loading fails
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        dataframes = []
        for file_path in file_paths:
            df = self.load_single_csv(file_path)
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)

        duplicate_ids = combined_df[combined_df["id"].duplicated()]
        if not duplicate_ids.empty:
            logger.warning(f"Found {len(duplicate_ids)} duplicate IDs")
            combined_df = combined_df.drop_duplicates(subset=["id"], keep="first")

        return combined_df

    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """Validate data integrity.

        Args:
            df: DataFrame to validate

        Returns:
            True if data is valid
        """
        if not all(col in df.columns for col in self.required_columns):
            return False

        if df.empty:
            return False

        expected_types = {
            "id": "object",
            self.input_column: "object",
            self.target_column: "object",
        }
        for col, expected_type in expected_types.items():
            if col in df.columns and df[col].dtype != expected_type:
                return False

        return True


def load_formula_datasets(
    data_dir: Union[str, Path] = "data/formulas_clean"
) -> pd.DataFrame:
    """Load all formula datasets from the specified directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Combined DataFrame with all datasets
    """
    data_dir = Path(data_dir)

    csv_files = [
        data_dir / "MED_formulas.csv",
        data_dir / "FraCaS_formulas.csv",
        data_dir / "CAD_formulas.csv",
    ]

    loader = CSVDataLoader()
    return loader.load_multiple_csvs(csv_files)


def load_csv_data(
    file_path: str,
    encoding: str = "utf-8",
    required_columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    loader = CSVDataLoader()
    df = loader.load_single_csv(file_path)

    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    return df.to_dict("records")


def load_multiple_csv_files(
    file_paths: List[str],
    encoding: str = "utf-8",
    required_columns: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    results = {}

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        results[file_name] = load_csv_data(file_path, encoding, required_columns)

    return results
