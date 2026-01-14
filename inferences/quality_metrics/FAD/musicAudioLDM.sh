#!/bin/bash

SCRIPT_NAME=$(basename "$0" .sh)
OUTPUT_DIR="./output"
GLOBAL_OUTPUT_FILE="${OUTPUT_DIR}/${SCRIPT_NAME}_fad_scores.csv"
BASELINE_DIR="../audios/baseline"
GENAUDIO_DIR="../audios/genaudios/${SCRIPT_NAME}"
BASELINES=("musiccaps_chosen" "songdescriber_chosen")

declare -A BASELINE_MAP
BASELINE_MAP["musiccaps_chosen"]="musiccaps"
BASELINE_MAP["songdescriber_chosen"]="songdescriber"

GENAUDIO_SUBFOLDERS=("10" "25" "50" "100" "150" "200")

mkdir -p "$OUTPUT_DIR"

# Header for final CSV
echo "model,steps,baseline,fad_score_laion,fad_score_encodec" > "$GLOBAL_OUTPUT_FILE"

for baseline in "${BASELINES[@]}"; do
    BASELINE_PATH="${BASELINE_DIR}/${baseline}"
    BASELINE_NAME="${BASELINE_MAP[$baseline]}"

    for subfolder in "${GENAUDIO_SUBFOLDERS[@]}"; do
        GENAUDIO_PATH="${GENAUDIO_DIR}/${BASELINE_NAME}/${subfolder}"
        echo "Processing: Baseline ${BASELINE_PATH} vs Generated ${GENAUDIO_PATH}"

        # run FAD with LAION and encodec (adjust encodec layer as you prefer)
        fadtk clap-laion-music "$BASELINE_PATH" "$GENAUDIO_PATH" laion_scores.csv --inf
        fadtk encodec-emb "$BASELINE_PATH" "$GENAUDIO_PATH" encodec_scores.csv --inf

        if [ ! -f laion_scores.csv ]; then
            echo "No LAION output file generated for baseline ${baseline} and subfolder ${subfolder}! Skipping."
            rm -f laion_scores.csv encodec_scores.csv
            continue
        fi
        if [ ! -f encodec_scores.csv ]; then
            echo "No encodec output file generated for baseline ${baseline} and subfolder ${subfolder}! Skipping."
            rm -f laion_scores.csv encodec_scores.csv
            continue
        fi

        # Helper to get the numeric 'score' column robustly:
        get_score() {
            local csvfile="$1"
            # try to find header index for column named "score"
            local idx
            idx=$(head -n1 "$csvfile" | awk -F, '{
                for(i=1;i<=NF;i++){
                  g=$i;
                  gsub(/^[ \t\r\n]+|[ \t\r\n]+$/,"",g);
                  if(g=="score"){ print i; exit }
                }
            }')
            # fallback to column 4 if not found
            if [ -z "$idx" ]; then
                idx=4
            fi
            awk -F, -v IDX="$idx" 'NR==2{print $IDX}' "$csvfile"
        }

        fad_laion=$(get_score laion_scores.csv)
        fad_encodec=$(get_score encodec_scores.csv)

        # Sanity: ensure we extracted numbers (allow scientific notation and decimals)
        num_re='^-?[0-9]+([eE][-+]?[0-9]+)?(\.[0-9]+)?$'
        if ! printf "%s" "$fad_laion" | grep -Eq "$num_re"; then
            echo "Warning: laion FAD is not numeric for ${GENAUDIO_PATH}: '$fad_laion' -- skipping row."
            rm -f laion_scores.csv encodec_scores.csv
            continue
        fi
        if ! printf "%s" "$fad_encodec" | grep -Eq "$num_re"; then
            echo "Warning: encodec FAD is not numeric for ${GENAUDIO_PATH}: '$fad_encodec' -- skipping row."
            rm -f laion_scores.csv encodec_scores.csv
            continue
        fi

        # Use consistent metadata for output columns
        model="${SCRIPT_NAME}"
        steps="${subfolder}"
        baseline_name="${BASELINE_NAME}"

        printf "%s,%s,%s,%s,%s\n" "$model" "$steps" "$baseline_name" "$fad_laion" "$fad_encodec" >> "$GLOBAL_OUTPUT_FILE"

        rm -f laion_scores.csv encodec_scores.csv
    done
done
