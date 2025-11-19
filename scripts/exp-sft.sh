#!/bin/bash

set -e

usage() {
    echo "Usage: $0 <config_file>"
    echo "  <config_file>: Path to the experiment config file (e.g., configs/t5base-50epoch/seed1/FML.yaml)"
    exit 1
}

if [ "$#" -ne 1 ]; then
    usage
fi

CONFIG_FILE=$1
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: .venv directory not found. Assuming environment is already set up."
fi

echo "Reading configuration from $CONFIG_FILE..."
OUTPUT_DIR=$(grep -A 20 "training:" "$CONFIG_FILE" | grep "output_dir:" | awk '{print $2}' | tr -d '"')
DATA_SOURCE=$(grep -A 20 "data:" "$CONFIG_FILE" | grep "source:" | awk '{print $2}' | tr -d '"')
TEST_FILE_NAME=$(grep -A 20 "data:" "$CONFIG_FILE" | grep "test_file:" | awk '{print $2}' | tr -d '"')

DATA_SOURCE_STRIPPED=${DATA_SOURCE%/}
EVAL_DATA="${DATA_SOURCE_STRIPPED}/${TEST_FILE_NAME}"

if [ -z "$OUTPUT_DIR" ] || [ -z "$EVAL_DATA" ]; then
    echo "Error: Could not parse output_dir or test_file from $CONFIG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
PREDICTION_FILE="$OUTPUT_DIR/predictions/detailed_report.tsv"
mkdir -p "$(dirname "$PREDICTION_FILE")"

echo "--- Starting Training ---"
echo "Config: $CONFIG_FILE"
echo "Output directory will be: $OUTPUT_DIR"

MASTER_PORT_CHOSEN=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
 
echo "Using master port: $MASTER_PORT_CHOSEN"
 
torchrun --nproc_per_node=1 --master_port "$MASTER_PORT_CHOSEN" src/training/train.py --config "$CONFIG_FILE"
echo "--- Training Finished ---"
 
LAST_CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
if [ -z "$LAST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi
echo "--- Last checkpoint found: $LAST_CHECKPOINT ---"

echo "--- Starting Evaluation ---"
echo "Evaluating checkpoint: $LAST_CHECKPOINT"
echo "Evaluation data: $EVAL_DATA"
echo "Prediction file will be saved to: $PREDICTION_FILE"

python -m src.evaluation.evaluator \
    --config "$CONFIG_FILE" \
    --checkpoint "$LAST_CHECKPOINT" \
    --csv "$EVAL_DATA" \
    --save-detailed-report "$PREDICTION_FILE" \
    --batch-size 8 \
    --device "cpu"
echo "--- Evaluation Finished ---"
echo "Detailed report saved to: $PREDICTION_FILE"

if [ -f scripts/summarize_results.py ]; then
    echo "--- Summarizing results for $OUTPUT_DIR ---"
    uv run python scripts/summarize_results.py "$OUTPUT_DIR" --output-md "$OUTPUT_DIR/summary.md"
    echo "Summary saved to: $OUTPUT_DIR/summary.md"
fi

echo "--- E2E Experiment Completed Successfully ---" 