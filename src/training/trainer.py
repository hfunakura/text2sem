"""Training functionality using Hugging Face Trainer."""

import os
import time
from typing import Dict, Any, Optional
import torch
import psutil
import numpy as np
import shutil
from transformers import (
    Seq2SeqTrainer as Trainer,
    Seq2SeqTrainingArguments as TrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from pydantic import BaseModel, field_validator

from src.utils.config import Config
from src.training.callbacks import (
    LoggingCallback,
    EarlyStoppingCallback,
    MetricsCallback,
    CheckpointCallback,
)
from src.training.metrics import ExactMatchMetric
from src.utils.logging_utils import get_logger


class TrainerConfig(BaseModel):
    """Configuration for trainer."""

    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_exact_match"
    greater_is_better: bool = True
    early_stopping_patience: int = 3

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate output directory."""
        if not v or not v.strip():
            raise ValueError("Output directory cannot be empty")
        return v

    @field_validator("num_train_epochs")
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        """Validate number of epochs."""
        if v <= 0:
            raise ValueError("Number of epochs must be positive")
        return v

    @field_validator("per_device_train_batch_size", "per_device_eval_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class SemanticParsingTrainer:
    """Trainer for semantic parsing models using Hugging Face Trainer."""

    def __init__(
        self,
        config: Config,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        train_dataset: Any,
        eval_dataset: Any,
        compute_metrics: Optional[callable] = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            model: T5 model for training
            tokenizer: T5 tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics: Function to compute metrics
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.logger = get_logger(__name__)

        self._validate_config()

        training_args = self._create_training_arguments()

        if compute_metrics is None:
            compute_metrics = self._compute_metrics

        callbacks = self._create_callbacks()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        self.logger.info("SemanticParsingTrainer initialized successfully")

    def _validate_config(self) -> None:
        """Validate training configuration."""
        training_config = self.config.training

        if not training_config.output_dir:
            raise ValueError("Output directory must be specified")

        if training_config.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if training_config.batch_size <= 0:
            raise ValueError("Training batch size must be positive")

        os.makedirs(training_config.output_dir, exist_ok=True)

    def _create_training_arguments(self) -> TrainingArguments:
        """Create TrainingArguments from configuration."""
        training_config = self.config.training

        use_cuda = torch.cuda.is_available() and not getattr(
            training_config, "use_cpu", False
        )

        if use_cuda:
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, using CPU")
            elif getattr(training_config, "use_cpu", False):
                self.logger.info("CPU usage forced by configuration")
            else:
                self.logger.info("Using CPU")

        args = {
            "output_dir": training_config.output_dir,
            "num_train_epochs": training_config.num_epochs,
            "per_device_train_batch_size": getattr(
                training_config, "per_device_train_batch_size", getattr(training_config, "batch_size", 8)
            ),
            "per_device_eval_batch_size": getattr(
                training_config, "per_device_eval_batch_size", getattr(training_config, "eval_batch_size", 16)
            ),
            "learning_rate": getattr(training_config, "learning_rate", 5e-5),
            "warmup_steps": getattr(training_config, "warmup_steps", 500),
            "weight_decay": getattr(training_config, "weight_decay", 0.0),
            "logging_dir": getattr(training_config, "logging_dir", None),
            "logging_steps": getattr(training_config, "logging_steps", 100),
            "save_strategy": getattr(training_config, "save_strategy", "steps"),
            "save_steps": getattr(training_config, "save_steps", 500),
            "save_total_limit": getattr(training_config, "save_total_limit", None),
            "load_best_model_at_end": False,
            "metric_for_best_model": getattr(
                training_config, "metric_for_best_model", None
            ),
            "greater_is_better": getattr(training_config, "greater_is_better", None),
            "fp16": use_cuda and getattr(training_config, "fp16", False),
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": getattr(
                training_config, "gradient_accumulation_steps", 1
            ),
            "seed": getattr(self.config, "seed", 42),
            "data_seed": getattr(self.config, "seed", 42),
        }


        return TrainingArguments(**args)

    def _create_callbacks(self) -> list:
        """Create training callbacks."""
        callbacks = []

        log_dir = os.path.join(self.config.training.output_dir, "logs")
        callbacks.append(LoggingCallback(log_dir=log_dir))

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Default patience
                early_stopping_threshold=0.001,
            )
        )

        metrics_file = os.path.join(self.config.training.output_dir, "metrics.json")
        callbacks.append(MetricsCallback(metrics_file=metrics_file))

        callbacks.append(
            CheckpointCallback(
                checkpoint_dir=self.config.training.output_dir,
                save_steps=self.config.training.save_steps,
                max_checkpoints=3,
            )
        )

        return callbacks

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if len(predictions.shape) == 3:
            predictions = predictions.argmax(axis=-1)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        try:
            decoded_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )

            decoded_labels = [label.strip() for label in decoded_labels]
            decoded_preds = [pred.strip() for pred in decoded_preds]

            exact_match_metric = ExactMatchMetric()
            exact_match = exact_match_metric.compute(decoded_preds, decoded_labels)

            return {"exact_match": exact_match}
        except Exception as e:
            self.logger.error(f"Error in metrics computation: {e}")
            return {"exact_match": 0.0}

    def _is_main_process(self) -> bool:
        """Check if the current process is the main process."""
        try:
            if hasattr(self.trainer, "is_world_process_zero"):
                return bool(self.trainer.is_world_process_zero())
        except Exception:
            pass
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        return rank in (None, "0")

    def _cleanup_checkpoints_keep_latest(self) -> None:
        """Clean up old checkpoint files to keep only the latest one."""
        output_dir = self.config.training.output_dir
        try:
            if not output_dir or not os.path.isdir(output_dir):
                return
            entries = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(output_dir, d))
            ]
            if not entries:
                return
            def step_num(name: str) -> int:
                try:
                    return int(name.split("-")[-1])
                except Exception:
                    return -1
            entries.sort(key=step_num)
            latest = entries[-1]
            latest_path = os.path.join(output_dir, latest)
            for d in entries[:-1]:
                path = os.path.join(output_dir, d)
                try:
                    shutil.rmtree(path)
                    self.logger.info(f"Removed old checkpoint: {path}")
                except Exception as e:
                    self.logger.error(f"Failed to remove checkpoint {path}: {e}")
            self.logger.info(f"Kept latest checkpoint: {latest_path}")
        except Exception as e:
            self.logger.error(f"Checkpoint cleanup failed: {e}")

    def train(self) -> Any:
        """Start training process."""
        self.logger.info("Starting training...")

        start_time = time.time()

        try:
            result = self.trainer.train()

            end_time = time.time()
            training_time = end_time - start_time

            self.logger.info(f"Training completed in {training_time:.2f} seconds")

            if hasattr(result, "training_loss") and result.training_loss is not None:
                try:
                    loss_value = float(result.training_loss)
                    self.logger.info(f"Final training loss: {loss_value:.4f}")
                except (ValueError, TypeError):
                    self.logger.info(f"Final training loss: {result.training_loss}")
            else:
                self.logger.info("Training loss not available")

            if self._is_main_process():
                self._cleanup_checkpoints_keep_latest()

            return result

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on evaluation dataset."""
        self.logger.info("Starting evaluation...")

        try:
            result = self.trainer.evaluate()

            self.logger.info("Evaluation completed")
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{key}: {value:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")

        try:
            self.trainer.save_model(checkpoint_path)
            self.logger.info("Checkpoint saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            self.model.load_state_dict(
                torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
            )

            self.logger.info("Checkpoint loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    def resume_from_checkpoint(self, checkpoint_path: str) -> Any:
        """Resume training from checkpoint."""
        self.logger.info(f"Resuming training from {checkpoint_path}")

        try:
            result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
            self.logger.info("Training resumed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to resume training: {str(e)}")
            raise

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()

        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
        }

    def predict(self, text: str, max_length: int = None) -> str:
        """
        入力文から論理式を生成する推論メソッド
        Args:
            text: 入力文
            max_length: 生成最大長（省略時はconfigのmax_length）
        Returns:
            生成された論理式（文字列）
        """
        if max_length is None:
            max_length = self.config.model.max_length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self.config.training.use_cpu
            else "cpu"
        )
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=1,
            )
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred.strip()
