import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.config import Config
from src.utils.reproducibility import set_seed
from src.data.dataset_builder import DatasetBuilder
from src.training.trainer import SemanticParsingTrainer
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description="Train a semantic parsing model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config)

    set_seed(config.seed)

    print(f"Using configuration from: {args.config}")
    print(f"Target column: {config.data.target_column}")
    print(f"Output directory: {config.training.output_dir}")

    model_name = config.model.name
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    data_config = config.data
    input_column = data_config.input_column
    target_column = data_config.target_column

    dataset_builder = DatasetBuilder(
        tokenizer,
        input_column=input_column,
        target_column=target_column,
        output_dir=config.training.output_dir,
    )

    train_file_path = os.path.join(data_config.source, data_config.train_file)
    validation_file_path = os.path.join(data_config.source, data_config.validation_file)

    datasets = dataset_builder.create_dataset_dict_from_files(
        file_paths={"train": train_file_path, "validation": validation_file_path},
        text_column=input_column,
        target_column=target_column,
    )

    tokenized_datasets = dataset_builder.tokenize_dataset(
        datasets,
        max_length=config.model.max_length,
        target_max_length=config.model.max_length,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    trainer = SemanticParsingTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"Training on {len(train_dataset)} samples")
    print(f"Evaluating on {len(eval_dataset)} samples")

    if (
        hasattr(train_dataset, "column_names")
        and "dataset_source" in train_dataset.column_names
    ):
        import pandas as pd

        train_df = pd.DataFrame(train_dataset)
        dataset_counts = train_df["dataset_source"].value_counts()
        print("Dataset composition:")
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} samples")

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
