from transformers import T5Tokenizer
from typing import Dict, Optional
import json
import os


def setup_t5_tokenizer(
    model_name: str = "google/t5-base", special_tokens: Optional[Dict[str, str]] = None
) -> T5Tokenizer:
    if special_tokens is None:
        special_tokens = {
            "formula_start": "<formula>",
            "formula_end": "</formula>",
            "equation_start": "<eq>",
            "equation_end": "</eq>",
            "variable_start": "<var>",
            "variable_end": "</var>",
        }

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    special_tokens_list = list(special_tokens.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    return tokenizer


def save_tokenizer_config(
    tokenizer: T5Tokenizer,
    save_path: str,
    special_tokens: Optional[Dict[str, str]] = None,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    tokenizer.save_pretrained(save_path)

    if special_tokens:
        config_path = os.path.join(save_path, "special_tokens.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, indent=2, ensure_ascii=False)


def load_tokenizer_config(load_path: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(load_path)

    return tokenizer
