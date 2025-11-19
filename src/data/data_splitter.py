import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(
        self,
        source_folder: str,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ):
        self.source_folder = Path(source_folder)
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self._validate_inputs()

        np.random.seed(seed)

    def _validate_inputs(self) -> None:
        if any(
            ratio < 0 for ratio in [self.train_ratio, self.val_ratio, self.test_ratio]
        ):
            raise ValueError("全ての比率は0以上である必要があります")

        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"比率の合計は1.0である必要があります。現在の合計: {self.train_ratio + self.val_ratio + self.test_ratio}"
            )

        if not self.source_folder.exists():
            raise FileNotFoundError(f"ソースフォルダが見つかりません: {self.source_folder}")

        if not self.source_folder.is_dir():
            raise NotADirectoryError(f"指定されたパスはディレクトリではありません: {self.source_folder}")

    def _find_csv_files(self) -> List[Path]:
        csv_files = list(self.source_folder.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"CSVファイルが見つかりません: {self.source_folder}")

        logger.info(f"{len(csv_files)}個のCSVファイルを発見: {[f.name for f in csv_files]}")
        return csv_files

    def _extract_dataset_name(self, filename: str) -> str:
        name_mapping = {
            "FraCaS_formulas.csv": "fracas",
            "MED_formulas.csv": "med",
            "CAD_formulas.csv": "cad",
            "SICK_formulas.csv": "sick",
        }

        if filename in name_mapping:
            return name_mapping[filename]

        return Path(filename).stem.lower()

    def _split_single_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(df) == 0:
            raise ValueError("空のデータセットです")

        if len(df) == 1:
            return (
                df.copy(),
                pd.DataFrame(columns=df.columns),
                pd.DataFrame(columns=df.columns),
            )

        if len(df) == 2:
            train_df = df.iloc[:1].copy()
            val_df = df.iloc[1:2].copy()
            test_df = pd.DataFrame(columns=df.columns)
            return train_df, val_df, test_df

        df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        n_total = len(df_shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train : n_train + n_val].copy()
        test_df = df_shuffled.iloc[n_train + n_val :].copy()

        logger.info(
            f"データセット分割完了: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def generate_output_folder_name(self) -> str:
        return f"{self.seed}_{self.train_ratio}-{self.val_ratio}-{self.test_ratio}"

    def split_datasets(self, output_dir: str, overwrite: bool = False) -> str:
        output_path = Path(output_dir)

        if output_path.exists() and not overwrite:
            existing_files = list(output_path.rglob("*.csv"))
            if existing_files:
                raise FileExistsError(
                    f"出力ディレクトリに既存のファイルがあります: {output_path}. overwrite=Trueを設定してください。"
                )

        output_path.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "val", "test"]:
            (output_path / split_name).mkdir(exist_ok=True)

        csv_files = self._find_csv_files()

        for csv_file in csv_files:
            logger.info(f"処理中: {csv_file.name}")

            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                logger.warning(f"空のファイルをスキップ: {csv_file.name}")
                continue

            if len(df) == 0:
                logger.warning(f"空のファイルをスキップ: {csv_file.name}")
                continue

            dataset_name = self._extract_dataset_name(csv_file.name)

            train_df, val_df, test_df = self._split_single_dataset(df)

            train_file = output_path / "train" / f"{dataset_name}-train.csv"
            val_file = output_path / "val" / f"{dataset_name}-val.csv"
            test_file = output_path / "test" / f"{dataset_name}-test.csv"

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

            logger.info(
                f"保存完了: {dataset_name} -> train({len(train_df)}), val({len(val_df)}), test({len(test_df)})"
            )

        logger.info(f"全データセットの分割が完了しました: {output_path}")
        return str(output_path)


def split_datasets(
    source_folder: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    output_base_dir: Optional[str] = None,
) -> str:
    splitter = DataSplitter(source_folder, seed, train_ratio, val_ratio, test_ratio)

    if output_base_dir is None:
        output_base_dir = "data"

    output_base_path = Path(output_base_dir)
    output_folder_name = splitter.generate_output_folder_name()
    output_dir = output_base_path / output_folder_name

    return splitter.split_datasets(str(output_dir), overwrite=True)
