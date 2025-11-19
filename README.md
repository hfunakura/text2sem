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

- `src/__init__.py`
  - l1
- `src/data/data_loader.py`
  - l1-5
  - l17
  - l25-31
  - l38-49
  - l94-104
  - l123-130
  - l152-159
- `src/data/dataset_builder.py`
  - l331-341
  - l373-381
- `src/data/preprocessor.py`
  - l17
  - l37
  - l49
  - l89
  - l129
- `src/evaluation/__init__.py`
  - l1
- `src/evaluation/evaluator-master.py`
  - l1-6
  - l28
  - l35
  - l60-62
  - l169
- `src/evaluation/evaluator.py`
  - l1-6
  - l27
  - l33-41
  - l86
  - l114-116
  - l214
- `src/models/__init__.py`
  - l1
- `src/models/model_config.py`
  - l1-4
  - l10
  - l24
- `src/models/semantic_parser.py`
  - l1-4
  - l19
  - l23
  - l28
  - l33
  - l39
  - l44
  - l47-52
  - l80-90
  - l103-114
  - l151-162
  - l197-201
  - l215-223
  - l264-268
  - l273
  - l277
- `src/training/__init__.py`
  - l1
- `src/training/callbacks.py`
  - l1
  - l14
  - l17-21
  - l33
  - l50
  - l64
  - l74
  - l85
  - l94
  - l99-104
  - l119
  - l161
  - l164-168
  - l183
  - l198
  - l207
  - l234
  - l239-245
  - l262
  - l278
  - l303
  - l328
- `src/training/metrics.py`
  - l1-6
  - l28
  - l35
  - l43
  - l55
  - l67
  - l88
  - l95
  - l98-104
  - l113-124
  - l129
  - l134
  - l145
  - l148-153
  - l187
  - l195-200
  - l210
  - l334-343
  - l373-381
  - l412
  - l429
  - l432-437
  - l458
  - l464
  - l564
  - l569-575
  - l582
  - l588
  - l602-611
  - l641
  - l650
  - l653-658
  - l664
  - l670
  - l687-696
  - l722
  - l732
  - l735-740
  - l746
  - l752
  - l779-788
  - l814
  - l824
  - l827-832
  - l838
  - l845
  - l885-894
  - l925
  - l939
  - l942
  - l947
  - l956-962
  - l969-977
  - l984
  - l988-997
  - l1006
  - l1009-1015
  - l1030-1036
  - l1040-1045
  - l1056-1069
  - l1104
  - l1122
  - l1126
  - l1133
  - l1137
  - l1147-1158
  - l1173-1185
- `src/training/trainer.py`
  - l1
  - l30
  - l51
  - l59
  - l67
  - l74
  - l85-94
  - l125
  - l140
  - l192
  - l219
  - l250
  - l260
  - l293
  - l325
  - l343
  - l355
  - l373
  - l386
  - l397-404
- `src/utils/__init__.py`
  - l1
- `src/utils/config.py`
  - l9
  - l13
  - l27
  - l66
  - l107
  - l162
  - l174
  - l206
  - l239
  - l252
  - l256
  - l275
- `src/utils/logging_utils.py`
  - l17
  - l22
  - l42
  - l88
  - l97
  - l118
  - l138
  - l169
  - l224
  - l284
  - l294
  - l311
  - l327
- `src/utils/reproducibility.py`
  - l15
  - l21
  - l45
  - l80
  - l98
  - l106
  - l125
  - l146
  - l193
