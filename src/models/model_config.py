"""Model configuration for Neural Semantic Parsing project.

This module provides configuration classes for semantic parsing models.
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for semantic parsing models."""

    model_name: str = "google/t5-base"
    max_input_length: int = 512
    max_output_length: int = 512
    num_beams: int = 4
    early_stopping: bool = True
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be positive")
        if self.max_output_length <= 0:
            raise ValueError("max_output_length must be positive")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
