"""Training callbacks for semantic parsing."""

import os
import json
import time
import shutil
from typing import Dict, Any, List, Optional
from transformers.trainer_callback import TrainerCallback
from transformers import TrainerState, TrainerControl, TrainingArguments

from src.utils.logging_utils import get_logger

class LoggingCallback(TrainerCallback):
    """Callback for detailed training logging."""

    def __init__(self, log_dir: str):
        """Initialize logging callback.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = get_logger(__name__)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of training."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Output directory: {args.output_dir}")
        self.logger.info(f"Number of epochs: {args.num_train_epochs}")
        self.logger.info(f"Training batch size: {args.per_device_train_batch_size}")
        self.logger.info(f"Evaluation batch size: {args.per_device_eval_batch_size}")
        self.logger.info(f"Learning rate: {args.learning_rate}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total steps: {state.global_step}")
        self.logger.info(f"Final epoch: {state.epoch}")

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of each epoch."""
        self.logger.info(f"Starting epoch {state.epoch}")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each epoch."""
        self.logger.info(f"Completed epoch {state.epoch}")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called when logging training metrics."""
        if logs:
            self.logger.info(f"Training Progress - Step {state.global_step}")
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")


class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on evaluation metrics."""

    def __init__(
        self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0
    ):
        """Initialize early stopping callback.

        Args:
            early_stopping_patience: Number of evaluations to wait for improvement
            early_stopping_threshold: Minimum improvement threshold
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0
        self.logger = get_logger(__name__)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation."""
        if not logs:
            return

        metric_name = args.metric_for_best_model
        if metric_name not in logs:
            return

        current_metric = logs[metric_name]

        if self.best_metric is None:
            self.best_metric = current_metric
            self.patience_counter = 0
            self.logger.info(f"Initial {metric_name}: {current_metric:.4f}")
            return

        improved = False
        if args.greater_is_better:
            if current_metric > self.best_metric + self.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric
        else:
            if current_metric < self.best_metric - self.early_stopping_threshold:
                improved = True
                self.best_metric = current_metric

        if improved:
            self.patience_counter = 0
            self.logger.info(f"New best {metric_name}: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            self.logger.info(
                f"No improvement in {metric_name}: {current_metric:.4f} "
                f"(patience: {self.patience_counter}/{self.early_stopping_patience})"
            )

            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info("Early stopping triggered!")
                control.should_training_stop = True


class MetricsCallback(TrainerCallback):
    """Callback for recording and saving training metrics."""

    def __init__(self, metrics_file: str):
        """Initialize metrics callback.

        Args:
            metrics_file: Path to save metrics JSON file
        """
        self.metrics_file = metrics_file
        self.metrics_history: List[Dict[str, Any]] = []
        self.logger = get_logger(__name__)

        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation."""
        if logs:
            metrics_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": time.time(),
                **logs,
            }

            self.metrics_history.append(metrics_entry)
            self.logger.info(f"Recorded metrics for step {state.global_step}")

            self.save_metrics()

    def save_metrics(self) -> None:
        """Save metrics history to file."""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            self.logger.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all evaluations."""
        if not self.metrics_history:
            return {}

        metric_names = set()
        for entry in self.metrics_history:
            for key in entry.keys():
                if key not in ["step", "epoch", "timestamp"]:
                    metric_names.add(key)

        aggregated = {}
        for metric_name in metric_names:
            values = [
                entry[metric_name]
                for entry in self.metrics_history
                if metric_name in entry and isinstance(entry[metric_name], (int, float))
            ]

            if values:
                aggregated[f"best_{metric_name}"] = max(values)
                aggregated[f"avg_{metric_name}"] = sum(values) / len(values)
                aggregated[f"final_{metric_name}"] = values[-1]

        return aggregated


class CheckpointCallback(TrainerCallback):
    """Callback for managing model checkpoints."""

    def __init__(
        self, checkpoint_dir: str, save_steps: int = 500, max_checkpoints: int = 3
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_steps: Steps between checkpoint saves
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.best_metric = None
        self.best_checkpoint_step = None
        self.logger = get_logger(__name__)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when saving a checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint-{state.global_step}"
        )

        self.logger.info(f"Saving checkpoint at step {state.global_step}")


    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation to track best checkpoint."""
        if not logs:
            return

        metric_name = args.metric_for_best_model
        if metric_name not in logs:
            return

        current_metric = logs[metric_name]

        if (
            self.best_metric is None
            or (args.greater_is_better and current_metric > self.best_metric)
            or (not args.greater_is_better and current_metric < self.best_metric)
        ):

            self.best_metric = current_metric
            self.best_checkpoint_step = state.global_step

            self.logger.info(
                f"New best checkpoint at step {state.global_step} "
                f"with {metric_name}: {current_metric:.4f}"
            )

    def cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        try:
            checkpoint_dirs = [
                d
                for d in os.listdir(self.checkpoint_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(self.checkpoint_dir, d))
            ]

            checkpoint_dirs.sort(
                key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0
            )

            while len(checkpoint_dirs) > self.max_checkpoints:
                oldest_checkpoint = checkpoint_dirs.pop(0)
                checkpoint_path = os.path.join(self.checkpoint_dir, oldest_checkpoint)

                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                    self.logger.info(f"Removed old checkpoint: {oldest_checkpoint}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {str(e)}")

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        if self.best_checkpoint_step is None:
            return None

        return os.path.join(
            self.checkpoint_dir, f"checkpoint-{self.best_checkpoint_step}"
        )
