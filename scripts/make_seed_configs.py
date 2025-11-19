import argparse
from pathlib import Path
from typing import Dict, Any
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2, allow_unicode=True)


def with_seed_suffix(s: str, seed: int) -> str:
    if not s:
        return s
    if s.endswith(f"_seed{seed}"):
        return s
    import re

    base = re.sub(r"_seed\d+$", "", s)
    return f"{base}_seed{seed}"


def generate_for_seed(base_yaml: Path, seed: int, out_yaml: Path) -> None:
    cfg = load_yaml(base_yaml)
    cfg["seed"] = seed
    tr = cfg.get("training", {})
    if "output_dir" in tr and isinstance(tr["output_dir"], str):
        tr["output_dir"] = with_seed_suffix(tr["output_dir"], seed)
    if "logging_dir" in tr and isinstance(tr["logging_dir"], str):
        tr["logging_dir"] = with_seed_suffix(tr["logging_dir"], seed)
    cfg["training"] = tr
    save_yaml(cfg, out_yaml)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="configs/t5mini-50epoch/seed1",
        help="Directory containing seed1 base YAMLs (FML.yaml, FOL.yaml)",
    )
    parser.add_argument(
        "--out-root",
        default="configs/t5mini-50epoch",
        help="Root directory to create seed2..10 subfolders",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Seeds to generate (default 1..10)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_root = Path(args.out_root)

    templates = ["FML.yaml", "FOL.yaml"]

    for seed in args.seeds:
        out_dir = out_root / f"seed{seed}"
        for name in templates:
            base_yaml = base_dir / name
            if not base_yaml.exists():
                continue
            out_yaml = out_dir / name
            generate_for_seed(base_yaml, seed, out_yaml)
            print(f"Generated: {out_yaml}")


if __name__ == "__main__":
    main() 