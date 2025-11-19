import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer
from typing import Dict, List, Optional, Union
import os
import random
from pathlib import Path

from src.data.data_loader import CSVDataLoader


class DatasetBuilder:
    def __init__(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        input_column: str = "text",
        target_column: str = "formula",
        output_dir: str = "results/outputs",
    ):
        self.tokenizer = tokenizer
        self.input_column = input_column
        self.target_column = target_column
        self.output_dir = output_dir
        self.data_loader = CSVDataLoader(
            input_column=self.input_column,
            target_column=self.target_column,
            output_dir=self.output_dir,
        )

    def load_dataset_from_csv(
        self, csv_path: str, text_column: str = "text", target_column: str = "formula"
    ) -> Dataset:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = self.data_loader.load_single_csv(csv_path)

        if df.empty:
            raise ValueError("Empty dataset after cleaning")

        rename_map = {}
        if text_column != self.input_column:
            rename_map[text_column] = self.input_column
        if target_column != self.target_column:
            rename_map[target_column] = self.target_column
        if rename_map:
            df = df.rename(columns=rename_map)

        if "id" not in df.columns:
            df["id"] = range(len(df))

        return Dataset.from_pandas(df)

    def create_dataset_dict(
        self,
        csv_file: str,
        split_ratios: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
    ) -> DatasetDict:
        if split_ratios is None:
            split_ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}

        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        for split_name, ratio in split_ratios.items():
            if ratio < 0:
                raise ValueError("Split ratios must be positive")

        if len(split_ratios) < 2:
            raise ValueError("At least 2 splits are required")

        default_splits = {"train", "validation", "test"}
        provided_splits = set(split_ratios.keys())

        if provided_splits.intersection(default_splits) and not default_splits.issubset(
            provided_splits
        ):
            missing_splits = default_splits - provided_splits
            raise ValueError(f"Required splits missing: {missing_splits}")

        dataset = self.load_dataset_from_csv(csv_file)

        return self.split_dataset(dataset, split_ratios, random_seed)

    def create_dataset_dict_from_files(
        self,
        file_paths: Dict[str, str],
        text_column: str = "text",
        target_column: str = "formula",
    ) -> DatasetDict:
        dataset_dict = {}

        for split_name, file_path in file_paths.items():
            dataset = self.load_dataset_from_csv(file_path, text_column, target_column)
            dataset_dict[split_name] = dataset

        return DatasetDict(dataset_dict)

    def tokenize_dataset(
        self,
        dataset: Union[Dataset, DatasetDict, pd.DataFrame],
        max_length: int = 512,
        target_max_length: int = 128,
    ) -> Union[Dataset, DatasetDict]:
        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained("google/t5-base")

        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset)

        def tokenize_function(examples):
            def to_str_list(items):
                if not isinstance(items, list):
                    items = [items]
                return [str(item) if pd.notna(item) else "" for item in items]

            texts = to_str_list(examples[self.input_column])
            formulas = to_str_list(examples[self.target_column])

            model_inputs = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    formulas,
                    max_length=target_max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                )

            model_inputs["labels"] = labels["input_ids"]

            batch_size = len(texts)
            for key, value in model_inputs.items():
                if isinstance(value, list):
                    if (
                        batch_size == 1
                        and len(value) > 1
                        and not isinstance(value[0], list)
                    ):
                        model_inputs[key] = [value]
                    elif len(value) != batch_size:
                        if len(value) < batch_size:
                            model_inputs[key] = value + [value[-1]] * (
                                batch_size - len(value)
                            )
                        else:
                            model_inputs[key] = value[:batch_size]

            return model_inputs

        if isinstance(dataset, DatasetDict):
            tokenized = dataset.map(tokenize_function, batched=True)
            for split_name in tokenized.keys():
                columns_to_remove = [
                    col
                    for col in [self.input_column, self.target_column]
                    if col in tokenized[split_name].column_names
                ]
                if columns_to_remove:
                    tokenized[split_name] = tokenized[split_name].remove_columns(
                        columns_to_remove
                    )
            return tokenized
        else:
            tokenized = dataset.map(tokenize_function, batched=True)
            columns_to_remove = [
                col
                for col in [self.input_column, self.target_column]
                if col in tokenized.column_names
            ]
            if columns_to_remove:
                tokenized = tokenized.remove_columns(columns_to_remove)
            return tokenized

    def split_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        split_ratios: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
    ) -> DatasetDict:
        if split_ratios is None:
            split_ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}

        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset)

        random.seed(random_seed)

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        total_size = len(dataset)
        dataset_dict = {}
        start_idx = 0

        split_names = list(split_ratios.keys())
        for i, split_name in enumerate(split_names):
            if i == len(split_names) - 1:
                split_indices = indices[start_idx:]
            else:
                split_size = int(total_size * split_ratios[split_name])
                split_indices = indices[start_idx : start_idx + split_size]
                start_idx += split_size

            dataset_dict[split_name] = dataset.select(split_indices)

        return DatasetDict(dataset_dict)

    def build_complete_dataset(
        self,
        data_source: Union[str, Dict[str, str], pd.DataFrame],
        tokenize: bool = True,
        max_length: int = 512,
        target_max_length: int = 128,
        split_ratios: Optional[Dict[str, float]] = None,
        text_column: str = "text",
        target_column: str = "formula",
        remove_columns: Optional[List[str]] = None,
        random_seed: int = 42,
        **kwargs,  # Accept additional parameters for backward compatibility
    ) -> DatasetDict:
        if "train_ratio" in kwargs or "val_ratio" in kwargs or "test_ratio" in kwargs:
            if split_ratios is not None:
                raise ValueError(
                    "Cannot specify both split_ratios and individual ratio parameters"
                )
            split_ratios = {
                "train": kwargs.get("train_ratio", 0.8),
                "validation": kwargs.get("val_ratio", 0.1),
                "test": kwargs.get("test_ratio", 0.1),
            }

        max_length = kwargs.get("max_length", max_length)
        random_seed = kwargs.get("random_seed", random_seed)

        if isinstance(data_source, str):
            data_path = Path(data_source)
            if data_path.is_dir():
                split_dirs = ["train", "val", "test"]
                if all((data_path / split_dir).exists() for split_dir in split_dirs):
                    print(f"事前分割済みデータディレクトリを使用: {data_path}")
                    dataset_dict = self._load_presplit_data(data_path)
                else:
                    presplit_path = self._detect_presplit_data(
                        data_path,
                        split_ratios or {"train": 0.8, "validation": 0.1, "test": 0.1},
                        random_seed,
                    )

                    if presplit_path:
                        print(f"事前分割済みデータを発見: {presplit_path}")
                        dataset_dict = self._load_presplit_data(presplit_path)
                    else:
                        print(f"事前分割済みデータが見つからないため、元データから分割します: {data_path}")
                        csv_files = list(data_path.glob("*.csv"))
                        if not csv_files:
                            raise ValueError(
                                f"No CSV files found in directory: {data_source}"
                            )

                        loader = CSVDataLoader(
                            input_column=text_column, target_column=target_column
                        )
                        combined_df = loader.load_multiple_csvs(csv_files)

                        if combined_df.empty:
                            raise ValueError("No valid data found in CSV files")

                        if "dataset_source" not in combined_df.columns:
                            combined_df["dataset_source"] = data_path.stem.lower()

                        dataset_dict = self.split_dataset(
                            combined_df, split_ratios, random_seed
                        )
            else:
                dataset = self.load_dataset_from_csv(
                    data_source, text_column, target_column
                )
                dataset_dict = self.create_dataset_dict(
                    dataset, split_ratios, random_seed
                )
        elif isinstance(data_source, dict):
            dataset_dict = self.create_dataset_dict_from_files(
                data_source, text_column, target_column
            )
        elif isinstance(data_source, pd.DataFrame):
            dataset_dict = self.split_dataset(data_source, split_ratios, random_seed)
        else:
            raise ValueError("Invalid data_source type")

        if tokenize and self.tokenizer is not None:
            dataset_dict = self.tokenize_dataset(
                dataset_dict, max_length, target_max_length
            )

            if remove_columns is None:
                columns_to_remove = [text_column, target_column]
            else:
                columns_to_remove = remove_columns

            for split_name in dataset_dict.keys():
                existing_columns = dataset_dict[split_name].column_names
                safe_columns_to_remove = [
                    col
                    for col in columns_to_remove
                    if col in existing_columns and not col.lower().endswith("id")
                ]
                if safe_columns_to_remove:
                    dataset_dict[split_name] = dataset_dict[split_name].remove_columns(
                        safe_columns_to_remove
                    )

        return dataset_dict

    def _detect_presplit_data(
        self, base_path: Path, split_ratios: Dict[str, float], random_seed: int
    ) -> Optional[Path]:
        """
        事前分割済みデータのディレクトリを自動検出する

        Args:
            base_path: ベースパス（例: data/formulas_clean）
            split_ratios: 分割比率
            random_seed: ランダムシード

        Returns:
            事前分割済みディレクトリのパスまたはNone
        """
        parent_dir = base_path.parent

        train_ratio = split_ratios.get("train", 0.8)
        val_ratio = split_ratios.get("validation", 0.1)
        test_ratio = split_ratios.get("test", 0.1)
        expected_dir_name = f"{random_seed}_{train_ratio}-{val_ratio}-{test_ratio}"

        candidate_paths = [
            parent_dir / expected_dir_name,  # data/42_0.7-0.15-0.15
            base_path.parent / expected_dir_name,  # data/42_0.7-0.15-0.15
            Path("data") / expected_dir_name,  # data/42_0.7-0.15-0.15
        ]

        for candidate_path in candidate_paths:
            if candidate_path.exists() and candidate_path.is_dir():
                required_splits = ["train", "val", "test"]
                if all((candidate_path / split).exists() for split in required_splits):
                    has_data = True
                    for split in required_splits:
                        split_dir = candidate_path / split
                        csv_files = list(split_dir.glob("*.csv"))
                        if not csv_files:
                            has_data = False
                            break

                    if has_data:
                        return candidate_path

        return None

    def _load_presplit_data(self, presplit_path: Path) -> DatasetDict:
        """
        事前分割済みデータを読み込む

        Args:
            presplit_path: 事前分割済みディレクトリのパス

        Returns:
            DatasetDict
        """
        dataset_dict = {}

        split_mapping = {"train": "train", "val": "validation", "test": "test"}

        loader = CSVDataLoader(
            input_column=self.input_column, target_column=self.target_column
        )

        for dir_name, split_name in split_mapping.items():
            split_dir = presplit_path / dir_name
            csv_files = list(split_dir.glob("*.csv"))

            if not csv_files:
                raise ValueError(f"No CSV files found in {split_dir}")

            combined_df = loader.load_multiple_csvs(csv_files)

            if not combined_df.empty:
                dataset_dict[split_name] = Dataset.from_pandas(combined_df)

        return DatasetDict(dataset_dict)


def build_semantic_parsing_dataset(
    data_source: Union[str, Dict[str, str], pd.DataFrame],
    tokenizer: Optional[T5Tokenizer] = None,
    max_length: int = 512,
    target_max_length: int = 128,
    split_ratios: Optional[Dict[str, float]] = None,
    text_column: str = "text",
    target_column: str = "formula",
    random_seed: int = 42,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
) -> DatasetDict:
    builder = DatasetBuilder(
        tokenizer, input_column=text_column, target_column=target_column
    )

    has_custom_params = (
        train_ratio is not None
        or val_ratio is not None
        or test_ratio is not None
        or max_length != 512
        or target_max_length != 128
        or split_ratios is not None
        or text_column != "text"
        or target_column != "formula"
        or random_seed != 42
    )

    if train_ratio is not None or val_ratio is not None or test_ratio is not None:
        kwargs = {}
        if train_ratio is not None:
            kwargs["train_ratio"] = train_ratio
        if val_ratio is not None:
            kwargs["val_ratio"] = val_ratio
        if test_ratio is not None:
            kwargs["test_ratio"] = test_ratio
        if max_length != 512:  # Only pass if different from default
            kwargs["max_length"] = max_length
        if random_seed != 42:  # Only pass if different from default
            kwargs["random_seed"] = random_seed

        return builder.build_complete_dataset(data_source, **kwargs)
    elif not has_custom_params:
        return builder.build_complete_dataset(data_source)
    else:
        return builder.build_complete_dataset(
            data_source,
            tokenize=True,
            max_length=max_length,
            target_max_length=target_max_length,
            split_ratios=split_ratios,
            text_column=text_column,
            target_column=target_column,
            random_seed=random_seed,
        )
