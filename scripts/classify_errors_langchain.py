import os
import json
import time
import argparse
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


CATEGORIES = [
    "QuantifierTypeMismatch",
    "QuantifierCountMismatch",
    "PredicateSymbolMismatch",
    "ArgumentRoleOrderMismatch",
    "SubformulaPresenceMismatch",
    "ConstantEntityMismatch",
    "AritySignatureMismatch",
    "QuantifierScopeMismatch",
    "ConnectiveMismatch",
    "NegationScopeMismatch",
    "ScopeParenthesesPrecedenceMismatch",
    "Other",
]


class SingleLabel(BaseModel):
    category: Literal[tuple(CATEGORIES)]
    subcategory: str
    details: str


def build_prompt() -> ChatPromptTemplate:
    system = (
        "You are an expert in formal logic and semantic parsing. "
        "Both formulas are well-formed but not logically equivalent (bi-implication fails). "
        "Select exactly one label: the single most critical factor that causes a logical difference. "
        "Return one category and a brief explanation."
    )
    human = "Gold formula:\n{gold}\n\nPredicted formula:\n{predicted}"
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])


def _canon(category: str, sub: str) -> tuple[str, str]:
    c = category.strip()
    s = sub.strip()
    s_lower = s.lower()
    if c == "QuantifierTypeMismatch":
        if "existential" in s_lower and "universal" in s_lower:
            s = "ExistentialVsUniversal"
        elif "existential" in s_lower and ("missing" in s_lower or "extra" in s_lower or "flip" in s_lower):
            s = "ExistentialType"
        elif "universal" in s_lower and ("missing" in s_lower or "extra" in s_lower or "flip" in s_lower):
            s = "UniversalType"
        elif "type" in s_lower or "all" in s_lower or "forall" in s_lower or "exists" in s_lower:
            s = "TypeFlip"
        else:
            s = "TypeFlip"
    elif c == "QuantifierCountMismatch":
        if "different" in s_lower and "number" in s_lower:
            s = "DifferentNumberOfQuantifiers"
        elif "missing" in s_lower and "quantifier" in s_lower:
            s = "MissingQuantifier"
        elif "extra" in s_lower and "quantifier" in s_lower:
            s = "ExtraQuantifier"
        elif "existential" in s_lower and ("count" in s_lower or "mismatch" in s_lower):
            s = "ExistentialQuantifierCountMismatch"
        elif "universal" in s_lower and ("count" in s_lower or "mismatch" in s_lower):
            s = "UniversalQuantifierCountMismatch"
        elif "variable count" in s_lower:
            s = "VariableCountMismatch"
        else:
            s = "DifferentNumberOfQuantifiers"
    elif c == "PredicateSymbolMismatch":
        if "name" in s_lower or "different predicate" in s_lower or "symbols" in s_lower or "lexical" in s_lower:
            s = "PredicateName"
        else:
            s = "PredicateName"
    elif c == "ArgumentRoleOrderMismatch":
        if "subject" in s_lower and ("object" in s_lower or "swap" in s_lower or "mismatch" in s_lower):
            s = "SubjectObjectSwap"
        elif "order" in s_lower or "role" in s_lower:
            s = "ArgumentOrder"
        elif "variable" in s_lower and ("mismatch" in s_lower or "reuse" in s_lower or "misalignment" in s_lower):
            s = "VariableMismatch"
        else:
            s = "ArgumentOrder"
    elif c == "SubformulaPresenceMismatch":
        if "missing conjunct" in s_lower or ("missing" in s_lower and "conjunct" in s_lower):
            s = "MissingConjunct"
        elif "extra conjunct" in s_lower or ("extra" in s_lower and "conjunct" in s_lower):
            s = "ExtraConjunct"
        elif "missing subformula" in s_lower or ("missing" in s_lower and "subformula" in s_lower):
            s = "MissingSubformula"
        elif "extra subformula" in s_lower or ("extra" in s_lower and "subformula" in s_lower):
            s = "ExtraSubformula"
        else:
            s = "MissingConjunct"
    elif c == "ConstantEntityMismatch":
        if "role" in s_lower:
            s = "EntityRoleMismatch"
        elif "attribute" in s_lower:
            s = "AttributeAssignment"
        else:
            s = "EntityMismatch"
    elif c == "AritySignatureMismatch":
        if "arity" in s_lower and "predicate" in s_lower:
            s = "PredicateArityMismatch"
        elif "arity" in s_lower and "event" in s_lower:
            s = "EventArityMismatch"
        elif "missing argument" in s_lower:
            s = "MissingArgument"
        elif "extra argument" in s_lower:
            s = "ExtraArgument"
        else:
            s = "PredicateArityMismatch"
    elif c == "QuantifierScopeMismatch":
        if "scope" in s_lower:
            s = "ExistentialScope" if "existential" in s_lower else "ScopeNesting"
        else:
            s = "ScopeNesting"
    elif c == "ConnectiveMismatch":
        if "and" in s_lower or "or" in s_lower or "implication" in s_lower or "equivalence" in s_lower:
            s = "WrongConnective"
        elif "missing" in s_lower:
            s = "MissingConnective"
        elif "extra" in s_lower:
            s = "ExtraConnective"
        else:
            s = "WrongConnective"
    elif c == "NegationScopeMismatch":
        if "scope" in s_lower:
            s = "NegationScope"
        elif "missing" in s_lower:
            s = "MissingNegation"
        elif "extra" in s_lower:
            s = "ExtraNegation"
        else:
            s = "NegationScope"
    elif c == "ScopeParenthesesPrecedenceMismatch":
        if "parentheses" in s_lower:
            s = "Parentheses"
        else:
            s = "Precedence"
    else:
        if "typo" in s_lower:
            s = "TypographicalError"
        else:
            s = s or ""
    return c, s


def _load_env_key(name: str, env_path: str = ".env") -> str:
    current = os.environ.get(name, "").strip()
    if current:
        return current
    file_key = None
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == name:
                        file_key = v.strip().strip('"').strip("'")
                        break
    if file_key:
        os.environ[name] = file_key
        return file_key
    raise SystemExit(f"{name} not found in environment or {env_path}")


def _load_processed_indices(jsonl_path: str) -> set:
    processed = set()
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = obj.get("index")
                    if isinstance(idx, int):
                        processed.add(idx)
                except Exception:
                    pass
    return processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-csv", default="")
    ap.add_argument("--raw-jsonl", default="")
    ap.add_argument("--errors-jsonl", default="")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    _load_env_key("OPENAI_API_KEY")

    df = pd.read_csv(args.input, sep="\t")
    if "formula" not in df.columns or "predicted_formula" not in df.columns:
        raise SystemExit("TSV must contain columns: formula, predicted_formula")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fout = open(args.output, "a", encoding="utf-8")
    raw_f = open(args.raw_jsonl, "a", encoding="utf-8") if args.raw_jsonl else None
    csv_f = open(args.output_csv, "a", encoding="utf-8") if args.output_csv else None
    err_f = open(args.errors_jsonl, "a", encoding="utf-8") if args.errors_jsonl else None
    if args.output_csv and (not os.path.isfile(args.output_csv) or os.stat(args.output_csv).st_size == 0):
        csv_f.write("index,category,subcategory,details\n")
        csv_f.flush()

    processed = _load_processed_indices(args.output) if args.resume else set()

    total = len(df)
    start = max(args.start, 0)
    end = total if args.limit <= 0 else min(total, start + args.limit)
    print(f"Classifying {end - start} of {total} rows (start={start})", flush=True)

    prompt = build_prompt()
    llm = ChatOpenAI(model=args.model, temperature=0)
    parser = llm.with_structured_output(SingleLabel)
    chain = prompt | parser

    for idx in range(start, end):
        if idx in processed:
            continue
        row = df.iloc[idx]
        gold = str(row.get("formula", ""))
        pred = str(row.get("predicted_formula", ""))
        out = None
        last_err = None
        for r in range(args.retries):
            try:
                out = chain.invoke({"gold": gold, "predicted": pred})
                break
            except Exception as e:
                last_err = str(e)
                if r == args.retries - 1:
                    out = None
                else:
                    time.sleep(args.backoff * (2 ** r))
        if out:
            cat, sub = _canon(out.category, out.subcategory)
            fout.write(json.dumps({"index": int(idx), "category": cat, "subcategory": sub, "details": out.details}, ensure_ascii=False) + "\n")
            if csv_f:
                csv_f.write(f"{idx},{cat},{sub},{out.details}\n")
            if raw_f:
                raw_f.write(json.dumps({"index": int(idx), "gold": gold, "predicted": pred, "raw": out.model_dump()}, ensure_ascii=False) + "\n")
        else:
            if err_f:
                err_f.write(json.dumps({"index": int(idx), "error": last_err or "api_error"}, ensure_ascii=False) + "\n")
        fout.flush()
        if raw_f:
            raw_f.flush()
        if csv_f:
            csv_f.flush()
        if err_f:
            err_f.flush()
        n_done = idx - start + 1
        if args.progress_every > 0 and (n_done % args.progress_every == 0 or idx == end - 1):
            print(f"Progress: {n_done}/{end - start}", flush=True)
    fout.close()
    if raw_f:
        raw_f.close()
    if csv_f:
        csv_f.close()
    if err_f:
        err_f.close()


if __name__ == "__main__":
    main() 