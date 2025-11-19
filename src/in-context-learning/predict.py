import argparse
import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import time
import os
import dotenv

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

try:
	from langchain_openai import ChatOpenAI
	from langchain.prompts import ChatPromptTemplate
	from langchain.schema import AIMessage
except Exception:
	print("LangChain and langchain_openai are required")
	raise

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(REPO_ROOT))
from src.training.metrics import ExactMatchMetric, ProverMetric, DrsMatchMetric  # type: ignore


def build_fewshot_block(examples: pd.DataFrame) -> str:
	lines = []
	for _, r in examples.iterrows():
		lines.append(f"text: {str(r['text']).strip()}\nformula: {str(r['formula']).strip()}")
	return "\n\n".join(lines)


def guidelines_for_task(task: str) -> str:
	if task == "fml":
		return (
			"Output only the logical formula, no explanations.\n"
			"Use leading-underscore predicate and role names exactly as in the examples: "
			"_dog(x1), _run(e1), _in(e1,x3), (subj(e1) = x1), (obj(e1) = x2).\n"
			"Quantification: exists x.(...). Conjunction: &. Equality: =.\n"
			"Negation: use the hyphen '-'.\n"
			"Multiword predicates are single tokens with hyphens/underscores: _t-shirt(x), _in_front_of(e,x).\n"
			"Variables: entities x1,x2,...; events e1,e2,.... Keep parentheses balanced and whitespace minimal."
		)
	else:
		return (
			"Output only the logical formula, no explanations.\n"
			"Use plain predicate and role names (no leading underscores) exactly as in the examples: "
			"dog(x1), run(e1), in(e1,x3), (subj(e1) = x1), (obj(e1) = x2).\n"
			"Quantification: exists e x.(...). Conjunction: &. Equality: =.\n"
			"Negation: use the hyphen '-'.\n"
			"Multiword predicates are single tokens joined with underscores: in_front_of(e,x).\n"
			"Variables: entities x1,x2,...; events e1,e2,.... Keep parentheses balanced and whitespace minimal."
		)


def create_llm(model: str, api_key: str, seed: int | None = None):
    base_kwargs = {"model": model, "temperature": 0.0, "api_key": api_key}

    if seed is not None:
        model_kwargs = {"seed": seed}
    else:
        model_kwargs = {}

    if model == "gpt-5":
        extra = {
            "output_version": "responses/v1",
            "reasoning": {"effort": "minimal"},
            "text": {"verbosity": "low"},
        }
        model_kwargs = {**model_kwargs, **extra}
    return ChatOpenAI(**{**base_kwargs, "model_kwargs": model_kwargs})


def load_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	return df


def find_vampire_exec() -> Path:
	return REPO_ROOT / "vampire-4.9casc2024/build/bin/vampire"


def compute_is_wff_flags(preds: list[str]) -> list[bool]:
	flags: list[bool] = []
	sys.path.insert(0, str(REPO_ROOT / "ccg2lambda" / "scripts"))
	try:
		from logic_parser import lexpr  # type: ignore
		from nltk2normal import is_wff  # type: ignore
		exceptions = (Exception,)
		for s in preds:
			ok = False
			try:
				expr = lexpr(str(s))
				ok = bool(is_wff(expr))
			except exceptions:
				ok = False
			flags.append(ok)
	finally:
		sys.path.pop(0)
	return flags


def write_summary_md(exp_dir: Path, detailed_df: pd.DataFrame) -> None:
	rel_pred = str((exp_dir / "predictions" / "detailed_report.tsv").relative_to(REPO_ROOT))
	loss_fig = ""
	def m(series):
		return float(pd.to_numeric(series, errors="coerce").mean(skipna=True)) if series is not None else float("nan")
	n = int(len(detailed_df))
	em = m(detailed_df.get("is_exact_match"))
	pc = m(detailed_df.get("is_prover_correct"))
	dp = m(detailed_df.get("drs_precision"))
	dr = m(detailed_df.get("drs_recall"))
	df1 = m(detailed_df.get("drs_fscore"))
	non_wff_ratio = 1.0 - m(detailed_df.get("is_wff")) if "is_wff" in detailed_df.columns else float("nan")
	name = exp_dir.name
	lines = []
	lines.append("| experiment | n | is_exact_match | is_prover_correct | drs_precision | drs_recall | drs_fscore | non_wellformed | predictions | loss_figure |")
	lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
	lines.append(
		f"| {name} | {n} | {em:.6f} | {pc:.6f} | {dp:.6f} | {dr:.6f} | {df1:.6f} | {non_wff_ratio:.6f} | {rel_pred} | {loss_fig} |"
	)
	(exp_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--model", required=False, default="gpt-4o")
	ap.add_argument("--task", required=True, choices=["fml", "fol"])
	ap.add_argument("--train_csv", required=False)
	ap.add_argument("--test_csv", required=False)
	ap.add_argument("--seed", type=int, required=True)
	ap.add_argument("--retries", type=int, default=3)
	ap.add_argument("--backoff", type=float, default=2.0)
	ap.add_argument("--resume", action="store_true")
	args = ap.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)

	if args.task == "fml":
		train_path = Path(args.train_csv) if args.train_csv else REPO_ROOT / "data" / "fml" / "train.csv"
		test_path = Path(args.test_csv) if args.test_csv else REPO_ROOT / "data" / "fml" / "test.csv"
		exp_dir = REPO_ROOT / "results" / f"{args.model}_{args.task}_{args.seed}_in-context"
	elif args.task == "fol":
		train_path = Path(args.train_csv) if args.train_csv else REPO_ROOT / "data" / "fol" / "train.csv"
		test_path = Path(args.test_csv) if args.test_csv else REPO_ROOT / "data" / "fol" / "test.csv"
		exp_dir = REPO_ROOT / "results" / f"{args.model}_{args.task}_{args.seed}_in-context"
	else:
		raise ValueError("invalid task")

	pred_dir = exp_dir / "predictions"
	pred_dir.mkdir(parents=True, exist_ok=True)
	out_tsv = pred_dir / "detailed_report.tsv"

	train_df = load_csv(train_path)
	test_df = load_csv(test_path)
	fewshot_df = train_df.sample(n=min(5, len(train_df)), random_state=args.seed)
	fewshot_block = build_fewshot_block(fewshot_df)
	notes = guidelines_for_task(args.task)

	llm = create_llm(args.model, api_key=api_key, seed=args.seed)
	prompt = ChatPromptTemplate.from_messages([
		("system", "You are a precise semantic parser that maps natural language to a logical formula. Respond with only the formula, no explanations."),
		("human", "Examples:\n\n{fewshot}\n\nGuidelines for this task:\n{notes}\n\nNow parse the following text to its logical formula.\n\ntext: {text}\nformula:")
	])

	prompt_sent = messages = prompt.format_messages(fewshot=fewshot_block, notes=notes, text="\{source text\}")
	print("===== PROMPT =====")
	print(prompt_sent)
	print("==================")

	N = len(test_df)
	print(f"Total test samples: {N}")

	em_metric = ExactMatchMetric()
	vampire_path = find_vampire_exec()
	if not vampire_path.exists() or not vampire_path.is_file():
		raise RuntimeError(f"Vampire executable not found at hardcoded path: {vampire_path}")
	prover_metric = ProverMetric(str(vampire_path))

	drs_parsing_dir = REPO_ROOT / "DRS_parsing"
	drs_metric = None
	if drs_parsing_dir.exists() and drs_parsing_dir.is_dir():
		try:
			drs_metric = DrsMatchMetric(drs_parsing_path=str(drs_parsing_dir))
		except Exception:
			drs_metric = None

	write_header = True
	processed_ids = set()
	if out_tsv.exists():
		try:
			df_exist = pd.read_csv(out_tsv, sep="\t")
			if "id" in df_exist.columns:
				processed_ids = set(df_exist["id"].astype(str).tolist())
			write_header = False
		except Exception:
			write_header = not out_tsv.exists()

	mode = "w" if write_header else "a"
	out_f = open(out_tsv, mode, encoding="utf-8")
	if write_header:
		cols = list(test_df.columns) + [
			"predicted_formula",
			"is_exact_match",
			"is_prover_correct",
			"drs_precision",
			"drs_recall",
			"drs_fscore",
			"is_wff",
		]
		out_f.write("\t".join(cols) + "\n")
		out_f.flush()

	for idx, r in enumerate(test_df.itertuples(index=False), start=1):
		row_id = str(getattr(r, "id", ""))
		if args.resume and row_id in processed_ids:
			print(f"[SKIP] id={row_id}")
			continue
		text = str(getattr(r, "text", ""))
		messages = prompt.format_messages(fewshot=fewshot_block, notes=notes, text=text)
		resp = None
		for attempt in range(args.retries):
			try:
				resp = llm.invoke(messages)
				break
			except Exception:
				if attempt == args.retries - 1:
					raise
				time.sleep(args.backoff * (2 ** attempt))
		content = resp.content if isinstance(resp, AIMessage) else str(resp)
		pred_text = None
		if isinstance(content, list):
			for part in content:
				if isinstance(part, dict) and ("text" in part):
					pred_text = part.get("text")
					break
		if pred_text is None:
			pred_text = content if isinstance(content, str) else str(content)
		pred = " ".join(str(pred_text).split()).strip()

		gold = str(getattr(r, "formula", ""))
		_, em_list = em_metric.compute([pred], [gold])
		is_em = bool(em_list[0])
		_, pr_list = prover_metric.compute([pred], [gold])
		is_prover = bool(pr_list[0])
		if drs_metric is not None:
			_, drs_list = drs_metric.compute([pred], [gold])
			if drs_list:
				drs_p = float(drs_list[0].get("precision", 0.0))
				drs_r = float(drs_list[0].get("recall", 0.0))
				drs_f = float(drs_list[0].get("fscore", 0.0))
			else:
				drs_p = 0.0
				drs_r = 0.0
				drs_f = 0.0
		else:
			drs_p = np.nan
			drs_r = np.nan
			drs_f = np.nan

		is_wff = compute_is_wff_flags([pred])[0]

		base_vals = [str(getattr(r, c, "")) for c in test_df.columns]
		row_vals = base_vals + [
			pred,
			str(is_em),
			str(is_prover),
			str(drs_p),
			str(drs_r),
			str(drs_f),
			str(is_wff),
		]
		out_f.write("\t".join(row_vals) + "\n")
		out_f.flush()
		print(f"[{idx}/{N}] id={row_id}")

	out_f.close()

	detailed_df = pd.read_csv(out_tsv, sep="\t")
	write_summary_md(exp_dir, detailed_df)


if __name__ == "__main__":
	main() 