# A Theorem-Proving-Based Evaluation of Neural Semantic Parsing
The codebase for "A Theorem-Proving-Based Evaluation of Neural Semantic Parsing," by Hayate Funakura, Hyunsoo Kim, and Koji Mineshima (BlackboxNLP 2025)

## Environment

This codebase was executed under the following conditions:

- Ubuntu 20.04.4 LTS (kernel 5.4.0-54-generic)
- CPU: Intel(R) Xeon(R) W-2235 (6 cores / 12 threads)
- Memory: approximately 62 GiB
- GPU: 2 x NVIDIA GeForce RTX 3090 (24576 MiB each), driver 535.230.02
- Python: 3.11
- Dependencies listed in `pyproject.toml`
- vampire-4.9casc2024
- OpenAI API key available (required for ICL and LLM-based error analysis)

## Setup

### Third-Party Dependencies
Set this repository as your working directory, then run each command:

Counter (DRS parsing and evaluation toolkit):
```sh
git clone https://github.com/RikVN/DRS_parsing.git third-party/DRS_parsing
```

ccg2lambda (semantic composition and logical inference pipeline):
```sh
git clone https://github.com/mynlp/ccg2lambda.git third-party/ccg2lambda
```
Vampire (theorem prover):
```sh
mkdir -p third-party/vampire
curl -L -o third-party/vampire/vampire-4.9casc2024.zip https://github.com/vprover/vampire/releases/download/v4.9casc2024/vampire-4.9casc2024.zip
bash scripts/install-vampire.sh
```
### OpenAI API
Create a `.env` file in the repository root with your OpenAI API key, following the format shown in `.env.example`.

### Python Dependencies
Install the required libraries with `uv sync`. Install uv first if it is not already available.

```sh
uv sync
```

## Supervised Fine-tuning

Run experiments for each configuration file as follows:

```sh
bash scripts/exp-sft.sh configs/t5base-50epoch/seed1/FML.yaml
bash scripts/exp-sft.sh configs/t5base-50epoch/seed1/FOL.yaml
```

Experimental results are saved in the following locations:
- `results/exp_fml_t5base_ep50_seed{1-10}/predictions/detailed_report.tsv`
- `results/exp_fol_t5base_ep50_seed{1-10}/predictions/detailed_report.tsv`

## In-context Learning
Run the ICL experiments with:

```sh
bash scripts/exp-in-context.sh "1 2 3" "fol" "gpt-4o gpt-4.1 gpt-5" src/in-context-learning/predict.py
```

Results are saved to:
- `results/ICL_results/gpt-4o_fol_{1-3}_in-context.tsv`
- `results/ICL_results/gpt-4.1_fol_{1-3}_in-context.tsv`
- `results/ICL_results/gpt-5_fol_{1-3}_in-context.tsv`

## Result Analysis

### SFT
Generate `results/exp_fol_t5base_ep50_seed*/predictions/sick_test_valid_with_predicted_formula.tsv` from `results/exp_fol_t5base_ep50_seed*/predictions/detailed_report.tsv`:

```sh
python scripts/attach_predicted_formula_valid_only_tsv.py \
  --test_csv ccg2lambda/data/sick_test.csv \
  --report_tsv results/exp_fol_t5base_ep50_seed5/predictions/detailed_report.tsv \
  --output_tsv results/exp_fol_t5base_ep50_seed5/predictions/sick_test_valid_with_predicted_formula.tsv
```

Extract misclassified cases from `sick_test_valid_with_predicted_formula.tsv` to create `results/exp_fol_t5base_ep50_seed*/predictions/wff_labels_langchain_single_raw.jsonl`:

```sh
python scripts/classify_errors_langchain.py \
  --input results/exp_fol_t5base_ep50_seed5/predictions/sick_test_valid_with_predicted_formula.tsv \
  --output results/exp_fol_t5base_ep50_seed5/predictions/wff_labels_langchain_single_raw.jsonl \
  --raw-jsonl results/exp_fol_t5base_ep50_seed5/predictions/wff_labels_langchain_single_raw.jsonl
```

Render figures with the generated labels and save them under `results/exp_fol_t5base_ep50_seed*/predictions/analysis/`:

```sh
python scripts/analyze_category_slices.py \
  --input_tsv results/exp_fol_t5base_ep50_seed5/predictions/sick_test_valid_with_predicted_formula.tsv \
  --report_tsv results/exp_fol_t5base_ep50_seed5/predictions/detailed_report.tsv \
  --wff_jsonl results/exp_fol_t5base_ep50_seed5/predictions/wff_labels_langchain_single_raw.jsonl \
  --output_dir results/exp_fol_t5base_ep50_seed5/predictions/analysis \
  --num_bins 8
```

### ICL
Run the ICL experiments and obtain `results/ICL_results/gpt-*_fol_{1,2,3}_in-context.tsv`:

```sh
bash scripts/exp-in-context.sh "1 2 3" "fol" "gpt-4o gpt-4.1 gpt-5" src/in-context-learning/predict.py
```

Aggregate the results with `src/in-context-learning/predict.py` to produce `results/{model}_{task}_{seed}_in-context/predictions/detailed_report.tsv` and the corresponding figures:

```sh
python src/in-context-learning/predict.py --task fol --seed 1
```

To generate figures from the ICL results, run `scripts/analyze_category_slices.py` with `results/ICL_results/{model}_{task}_{seed}_in-context.tsv` as input:

```sh
python scripts/analyze_category_slices.py \
  --input_tsv results/ICL_results/gpt-5_fol_1_in-context.tsv \
  --report_tsv results/ICL_results/gpt-5_fol_1_in-context.tsv \
  --output_dir results/ICL_results/analysis/gpt-5_fol_1-in-context \
  --num_bins 8
```

The `--wff_jsonl` argument is optional. If omitted, error-category analysis is skipped. We do not specify this argument for ICL experiments because we do not analyze error categories for them.

## AI-generated Docstrings

See `ai-generated-docstrings.txt` for a detailed list of AI-generated docstrings.
