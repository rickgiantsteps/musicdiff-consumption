#!/bin/bash
set -e

STEP_FOLDERS=("10" "25" "50" "100" "150" "200")
TYPE_FOLDERS=("musiccaps" "songdescriber")
BATCH_SIZE=64

# get model name from filename
SCRIPT_NAME=$(basename "$0")      # e.g., AudioLDM.sh
MODEL_NAME=${SCRIPT_NAME%.sh}     # e.g., AudioLDM

BASE_INPUT_DIR="../audios/genaudios/$MODEL_NAME"


echo "Scoring audio files..."

for STEP in "${STEP_FOLDERS[@]}"; do
    echo "Processing Step: $STEP"

    for TYPE in "${TYPE_FOLDERS[@]}"; do
        echo "  -- Baseline: $TYPE --"

        CURRENT_DIR="$BASE_INPUT_DIR/$TYPE/$STEP"
        if [ ! -d "$CURRENT_DIR" ]; then
            echo "Directory not found, skipping: $CURRENT_DIR"
            continue
        fi

        INPUT_JSONL="input_${TYPE}_${STEP}.jsonl"
        OUTPUT_JSONL="${MODEL_NAME}_${TYPE}_${STEP}_scores.jsonl"

        # generate input.jsonl for this audio directory
        echo "Generating $INPUT_JSONL from $CURRENT_DIR"
        find "$CURRENT_DIR" -type f -iname '*.wav' | \
            awk '{gsub(/\\/,"/"); print "{\"path\":\""$0"\"}"}' > "$INPUT_JSONL"

        NUM_FILES=$(wc -l < "$INPUT_JSONL")
        if [ "$NUM_FILES" -eq 0 ]; then
            echo "No audio files found. Skipping..."
            rm "$INPUT_JSONL"
            continue
        fi
        echo "Found $NUM_FILES audio files."

        echo "Running audio-aes..."
        audio-aes "$INPUT_JSONL" --batch-size "$BATCH_SIZE" > "$OUTPUT_JSONL"

        rm "$INPUT_JSONL" # temporary file

        echo "Scores saved: $(pwd)/$OUTPUT_JSONL"
    done
    echo
done

echo "Scores assigned successfully!"

echo "Aggregating Scores into CSV..."

# check for jq dependency
if ! command -v jq &> /dev/null; then
    echo >&2 "Error: This script requires 'jq' to aggregate results. Please install and try again."
    exit 1
fi

SUMMARY_CSV="${MODEL_NAME}_audiobox.csv"
echo "Model,Baseline,Steps,Avg_CE,Avg_CU,Avg_PC,Avg_PQ" > "$SUMMARY_CSV"

# loop through the generated score files to average
for TYPE in "${TYPE_FOLDERS[@]}"; do
    for STEP in "${STEP_FOLDERS[@]}"; do
        RESULT_FILE="${MODEL_NAME}_${TYPE}_${STEP}_scores.jsonl"

        if [ -s "$RESULT_FILE" ]; then
            echo "  -> Processing $RESULT_FILE..."
            # calculate the average of each metric
            AVG_CE=$(jq -s '(map(.CE) | add) / length' "$RESULT_FILE")
            AVG_CU=$(jq -s '(map(.CU) | add) / length' "$RESULT_FILE")
            AVG_PC=$(jq -s '(map(.PC) | add) / length' "$RESULT_FILE")
            AVG_PQ=$(jq -s '(map(.PQ) | add) / length' "$RESULT_FILE")
            echo "$MODEL_NAME,$TYPE,$STEP,$AVG_CE,$AVG_CU,$AVG_PC,$AVG_PQ" >> "$SUMMARY_CSV"
        else
            echo "  -> Skipping missing or empty file: $RESULT_FILE"
        fi

    done
done

echo
echo "Final scores saved at: $(pwd)/$SUMMARY_CSV"
