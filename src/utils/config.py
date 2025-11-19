import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict


class ConfigError(Exception):
    """Configuration related errors"""
    pass

class ModelConfig(BaseModel):
    """Model configuration"""

    name: str = Field(..., description="Model name")
    max_length: int = Field(512, ge=1, le=2048, description="Maximum sequence length")

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v


class TrainingConfig(BaseModel):
    """Training configuration"""

    batch_size: int = Field(8, ge=1, le=128, description="Training batch size")
    eval_batch_size: int = Field(16, ge=1, le=128, description="Evaluation batch size")
    learning_rate: float = Field(5e-5, gt=0, le=1e-2, description="Learning rate")
    num_epochs: int = Field(10, ge=1, le=100, description="Number of training epochs")
    num_train_epochs: int | None = Field(
        None,
        ge=1,
        le=100,
        description="(Alias) Number of training epochs; overrides num_epochs if set",
    )
    warmup_steps: int = Field(500, ge=0, description="Number of warmup steps")
    save_steps: int = Field(500, ge=1, description="Save checkpoint every N steps")
    eval_steps: int = Field(500, ge=1, description="Evaluate every N steps")
    output_dir: str = Field("outputs", description="Output directory")
    logging_dir: str = Field("logs", description="Logging directory")
    use_cpu: bool = Field(
        False, description="Force CPU usage even if CUDA is available"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1e-2:
            raise ValueError("Learning rate must be between 0 and 0.01")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if self.num_train_epochs is not None:
            object.__setattr__(self, "num_epochs", self.num_train_epochs)
        else:
            object.__setattr__(self, "num_train_epochs", self.num_epochs)


class DataConfig(BaseModel):
    """Data configuration"""

    source: str = Field(..., description="Path to training data CSV or directory")
    train_file: str = Field(..., description="Name of the training CSV file")
    validation_file: str = Field(..., description="Name of the validation CSV file")
    input_column: str = Field("text", description="Name of the input text column")
    target_column: str = Field(
        "formula", description="Name of the target formula column"
    )
    split_ratios: Optional[Dict[str, float]] = Field(
        default=None, description="Data split ratios (train, validation, test)"
    )

    @field_validator("split_ratios")
    @classmethod
    def validate_split_ratios(cls, v):
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("split_ratios must be a dictionary")

        required_keys = {"train", "validation", "test"}
        if not required_keys.issubset(v.keys()):
            missing = required_keys - v.keys()
            raise ValueError(f"Missing required split ratios: {missing}")

        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        for key, ratio in v.items():
            if ratio < 0:
                raise ValueError(
                    f"Split ratio for {key} must be non-negative, got {ratio}"
                )

        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""

    metrics: List[str] = Field(
        default=["exact_match"], description="Metrics to compute"
    )
    metrics_config: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Configuration for each metric (metric_name -> config_dict)",
    )
    output_dir: str = Field(
        default="results/evaluation",
        description="Output directory for evaluation results",
    )
    save_predictions: bool = Field(default=True, description="Save predictions to file")
    normalize_whitespace: bool = Field(
        default=True, description="Normalize whitespace in comparisons"
    )
    case_sensitive: bool = Field(default=False, description="Case sensitive comparison")
    dependencies: Optional[Dict[str, str]] = Field(
        default=None, description="Paths to external dependencies"
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
        if not v:
            raise ValueError("At least one metric must be specified")

        valid_metrics = {
            "exact_match",
            "bleu",
            "rouge",
            "semantic_similarity",
            "drs_match",
            "prover",
        }
        invalid_metrics = set(v) - valid_metrics
        if invalid_metrics:
            print(f"Warning: Unknown metrics specified: {invalid_metrics}")

        return v

    @field_validator("metrics_config")
    @classmethod
    def validate_metrics_config(cls, v):
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("metrics_config must be a dictionary")

        return v


class Config(BaseModel):
    """Main configuration class"""

    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    evaluation: Optional[EvaluationConfig] = Field(default_factory=EvaluationConfig)
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading config file: {e}")

        if not config_data:
            raise ConfigError("Config file is empty")

        config_data = cls._apply_env_overrides(config_data)

        try:
            return cls(**config_data)
        except ValidationError as e:
            for error in e.errors():
                if error["type"] == "missing":
                    field_name = ".".join(str(loc) for loc in error["loc"])
                    raise ConfigError(f"Missing required field: {field_name}")
            raise ConfigError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigError(f"Error creating config: {e}")

    @staticmethod
    def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_prefix = "NSP_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            config_key = env_key[len(env_prefix) :].lower()

            if "_" in config_key:
                parts = config_key.split("_")
                if len(parts) >= 2:
                    section = parts[0]
                    field = "_".join(parts[1:])

                    if section not in config_data:
                        config_data[section] = {}

                    try:
                        if env_value.isdigit() or (
                            env_value.startswith("-") and env_value[1:].isdigit()
                        ):
                            config_data[section][field] = int(env_value)
                        elif "." in env_value:
                            config_data[section][field] = float(env_value)
                        else:
                            config_data[section][field] = env_value
                    except ValueError:
                        config_data[section][field] = env_value

        return config_data

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump()

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigError(f"Error saving config file: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.model_dump()

    def update(self, **kwargs) -> "Config":
        """Create a new config with updated values"""
        config_dict = self.model_dump()

        for key, value in kwargs.items():
            if "." in key:
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value

        return Config(**config_dict)


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Config:
    """Helper to load Config from YAML file with a default path."""
    return Config.from_file(config_path)
