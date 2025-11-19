import os
import sys
import random
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

class ReproducibilityError(Exception):
    """Reproducibility related errors"""

    pass


def set_seed(seed: int) -> None:
    """Set seed for all random number generators"""
    if not isinstance(seed, int):
        raise ReproducibilityError("Seed must be an integer")

    if seed < 0:
        raise ReproducibilityError("Seed must be non-negative")

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information"""
    import importlib.metadata

    packages = {}
    try:
        for dist in importlib.metadata.distributions():
            packages[dist.metadata["name"].lower()] = dist.version
    except Exception:
        import pkg_resources

        for dist in pkg_resources.working_set:
            packages[dist.project_name.lower()] = dist.version

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "packages": packages,
    }

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cudnn_version"] = torch.backends.cudnn.version()
        env_info["cuda_device_count"] = torch.cuda.device_count()
        env_info["cuda_device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    return env_info


def save_environment_info(file_path: Union[str, Path]) -> None:
    """Save environment information to JSON file"""
    file_path = Path(file_path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        env_info = get_environment_info()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2, sort_keys=True)

    except PermissionError:
        raise ReproducibilityError("Permission denied: Cannot write environment file")
    except Exception as e:
        raise ReproducibilityError(f"Error saving environment info: {e}")


def calculate_hash(data: Union[str, bytes]) -> str:
    """Calculate SHA-256 hash of data"""
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of file"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise ReproducibilityError(f"File not found: {file_path}")

    hash_sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
    except Exception as e:
        raise ReproducibilityError(f"Error calculating file hash: {e}")

    return hash_sha256.hexdigest()


def ensure_reproducibility(config) -> None:
    """Ensure complete reproducibility setup"""
    seed = getattr(config, "seed", 42)
    set_seed(seed)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_file = output_dir / "environment.json"
    save_environment_info(env_file)

    config_hash = calculate_hash(str(config.model_dump()))
    hash_file = output_dir / "config_hash.txt"

    try:
        with open(hash_file, "w") as f:
            f.write(config_hash)
    except Exception as e:
        raise ReproducibilityError(f"Error saving config hash: {e}")


def verify_reproducibility(config, reference_dir: Union[str, Path]) -> bool:
    """Verify reproducibility against reference environment"""
    reference_dir = Path(reference_dir)

    ref_env_file = reference_dir / "environment.json"
    ref_hash_file = reference_dir / "config_hash.txt"

    if not ref_env_file.exists() or not ref_hash_file.exists():
        raise ReproducibilityError("Reference files not found")

    try:
        with open(ref_env_file, "r") as f:
            ref_env = json.load(f)

        with open(ref_hash_file, "r") as f:
            ref_config_hash = f.read().strip()

        current_env = get_environment_info()
        current_config_hash = calculate_hash(str(config.model_dump()))

        checks = {
            "python_version": ref_env["python_version"]
            == current_env["python_version"],
            "platform": ref_env["platform"] == current_env["platform"],
            "config_hash": ref_config_hash == current_config_hash,
            "cuda_available": ref_env["cuda_available"]
            == current_env["cuda_available"],
        }

        ref_packages = ref_env.get("packages", {})
        current_packages = current_env.get("packages", {})

        key_packages = ["torch", "numpy", "transformers", "datasets"]
        for package in key_packages:
            if package in ref_packages and package in current_packages:
                checks[f"{package}_version"] = (
                    ref_packages[package] == current_packages[package]
                )

        return all(checks.values())

    except Exception as e:
        raise ReproducibilityError(f"Error verifying reproducibility: {e}")


def create_reproducibility_report(
    config, output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """Create comprehensive reproducibility report"""
    output_dir = Path(output_dir)

    ensure_reproducibility(config)

    report = {
        "timestamp": datetime.now().isoformat(),
        "seed": config.seed,
        "environment": get_environment_info(),
        "config_hash": calculate_hash(str(config.model_dump())),
        "reproducibility_status": "configured",
    }

    report_file = output_dir / "reproducibility_report.json"

    try:
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    except Exception as e:
        raise ReproducibilityError(f"Error saving reproducibility report: {e}")

    return report
