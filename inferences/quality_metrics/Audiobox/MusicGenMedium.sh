#!/bin/bash
set -e

TYPE_FOLDERS=("musiccaps" "songdescriber")
BATCH_SIZE=64

SCRIPT_NAME=$(basename "$0")
MODEL_NAME=${SCRIPT_NAME%.sh}
echo "Detected script name: $MODEL_NAME"

if [[ "$MODEL_NAME" =~ ^([A-Za-z0-9._-]+)(Small|Medium|Large)$ ]]; then
    MODEL_BASE="${BASH_REMATCH[1]}"
    SIZE="${BASH_REMATCH[2]}"
    BASE_INPUT_DIR="../audios/genaudios/${MODEL_BASE}/${SIZE}"
    echo "Processing model: ${MODEL_BASE} | Size: ${SIZE}"
else
    echo "Error: Script name must end in Small, Medium, or Large"
    exit 1
fi

echo
echo "PART 1: Scoring audio files"

for TYPE in "${TYPE_FOLDERS[@]}"; do

    if [ -d "${BASE_INPUT_DIR}/${TYPE}" ]; then
        CURRENT_DIR="${BASE_INPUT_DIR}/${TYPE}"
    else
        CURRENT_DIR="${BASE_INPUT_DIR}"
    fi

    INPUT_JSONL="input_${TYPE}.jsonl"
    OUTPUT_JSONL="${MODEL_NAME}_${TYPE}_scores.jsonl"

    if [ ! -d "$CURRENT_DIR" ]; then
        echo "Warning: Directory not found: $CURRENT_DIR"
        continue
    fi

    echo "Generating $INPUT_JSONL from $CURRENT_DIR"
    rm -f "$INPUT_JSONL"

    # helper for JSONL generation
    python3 - "$CURRENT_DIR" > "$INPUT_JSONL" <<'PYCODE'
import os, sys, json
root = sys.argv[1]
for dirpath, dirnames, filenames in os.walk(root):
    filenames.sort()
    for fn in filenames:
        if fn.lower().endswith('.wav'):
            path = os.path.join(dirpath, fn)
            print(json.dumps({"path": path}))
PYCODE

    if [ ! -s "$INPUT_JSONL" ]; then
        echo "No audio files found. Skipping."
        rm -f "$INPUT_JSONL"
        continue
    fi

    audio-aes "$INPUT_JSONL" --batch-size "$BATCH_SIZE" > "$OUTPUT_JSONL"
    rm -f "$INPUT_JSONL"
done

echo "PART 2: Aggregating scores into CSV"

SUMMARY_CSV="${MODEL_NAME}_audiobox.csv"
echo "Model,Baseline,Steps,Avg_CE,Avg_CU,Avg_PC,Avg_PQ" > "$SUMMARY_CSV"

for TYPE in "${TYPE_FOLDERS[@]}"; do
    RESULT_FILE="${MODEL_NAME}_${TYPE}_scores.jsonl"
    if [ -s "$RESULT_FILE" ]; then
        AVG_CE=$(jq -s 'map(.CE) | add / length' "$RESULT_FILE")
        AVG_CU=$(jq -s 'map(.CU) | add / length' "$RESULT_FILE")
        AVG_PC=$(jq -s 'map(.PC) | add / length' "$RESULT_FILE")
        AVG_PQ=$(jq -s 'map(.PQ) | add / length' "$RESULT_FILE")
        echo "$MODEL_NAME,$TYPE,$SIZE,$AVG_CE,$AVG_CU,$AVG_PC,$AVG_PQ" >> "$SUMMARY_CSV"
    fi
done

echo "Done. Summary: $SUMMARY_CSV"