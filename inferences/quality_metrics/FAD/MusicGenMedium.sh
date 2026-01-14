#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME_RAW=$(basename "$0")
SCRIPT_NAME="${SCRIPT_NAME_RAW%.sh}"

if [ $# -ge 2 ]; then
  MODEL="$1"
  SIZE="$2"
else
  if [[ "$SCRIPT_NAME" =~ ^(MusicGen|Magnet)(Small|Medium|Large)$ ]]; then
    MODEL="${BASH_REMATCH[1]}"
    SIZE="${BASH_REMATCH[2]}"
  else
    echo "Unable to parse model+size from script name '$SCRIPT_NAME'."
    echo "Rename script to e.g. MusicGenSmall.sh or call with two args: $0 <Model> <Size>"
    exit 1
  fi
fi

case "$MODEL" in
  MusicGen|Magnet) ;;
  *)
    echo "Unsupported model: $MODEL. Supported: MusicGen, Magnet"
    exit 2
    ;;
esac

case "$SIZE" in
  Small|Medium|Large) ;;
  *)
    echo "Unsupported size: $SIZE. Supported: Small, Medium, Large"
    exit 3
    ;;
esac

OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
GLOBAL_OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}${SIZE}_fad_scores.csv"

BASELINE_DIR="../audios/baseline"
GENAUDIO_BASE="../audios/genaudios"

BASELINES=("musiccaps_chosen" "songdescriber_chosen")
declare -A BASELINE_MAP
BASELINE_MAP["musiccaps_chosen"]="musiccaps"
BASELINE_MAP["songdescriber_chosen"]="songdescriber"

echo "model,size,baseline,fad_score_laion,fad_score_encodec" > "${GLOBAL_OUTPUT_FILE}"

cleanup() {
  rm -f laion_scores.csv encodec_scores.csv
}
trap cleanup EXIT

get_score() {
  local csvfile="$1"
  [ -f "$csvfile" ] || return 1

  local idx
  idx=$(head -n1 "$csvfile" | awk -F, '{
    for(i=1;i<=NF;i++){
      g=$i; gsub(/^[ \t\r\n]+|[ \t\r\n]+$/,"",g);
      if(g=="score"){ print i; exit }
    }
  }')

  if [ -z "$idx" ]; then
    idx=4
  fi

  awk -F, -v IDX="$idx" 'NR==2{print $IDX}' "$csvfile"
}

num_re='^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'

for baseline in "${BASELINES[@]}"; do
  BASELINE_PATH="${BASELINE_DIR}/${baseline}"
  baseline_name="${BASELINE_MAP[$baseline]}"
  GENAUDIO_PATH="${GENAUDIO_BASE}/${MODEL}/${SIZE}/${baseline_name}"

  echo "Processing: model=${MODEL}, size=${SIZE}, baseline=${baseline} -> ${GENAUDIO_PATH}"

  if [ ! -d "$BASELINE_PATH" ]; then
    echo "  Warning: baseline path missing: ${BASELINE_PATH} -- skipping."
    continue
  fi
  if [ ! -d "$GENAUDIO_PATH" ]; then
    echo "  Warning: generated-audio path missing: ${GENAUDIO_PATH} -- skipping."
    continue
  fi

  fadtk clap-laion-music "$BASELINE_PATH" "$GENAUDIO_PATH" laion_scores.csv --inf
  fadtk encodec-emb "$BASELINE_PATH" "$GENAUDIO_PATH" encodec_scores.csv --inf

  if [ ! -f laion_scores.csv ]; then
    echo "  No laion output for ${GENAUDIO_PATH} -- skipping."
    rm -f laion_scores.csv encodec_scores.csv
    continue
  fi
  if [ ! -f encodec_scores.csv ]; then
    echo "  No encodec output for ${GENAUDIO_PATH} -- skipping."
    rm -f laion_scores.csv encodec_scores.csv
    continue
  fi

  fad_laion=$(get_score laion_scores.csv) || fad_laion=""
  fad_encodec=$(get_score encodec_scores.csv) || fad_encodec=""

  if ! printf "%s" "$fad_laion" | grep -Eq "$num_re"; then
    echo "  Warning: laion FAD not numeric for ${GENAUDIO_PATH}: '$fad_laion' -- skipping."
    rm -f laion_scores.csv encodec_scores.csv
    continue
  fi
  if ! printf "%s" "$fad_encodec" | grep -Eq "$num_re"; then
    echo "  Warning: encodec FAD not numeric for ${GENAUDIO_PATH}: '$fad_encodec' -- skipping."
    rm -f laion_scores.csv encodec_scores.csv
    continue
  fi

  printf "%s,%s,%s,%s,%s\n" "$MODEL" "$SIZE" "$baseline_name" "$fad_laion" "$fad_encodec" >> "${GLOBAL_OUTPUT_FILE}"

  rm -f laion_scores.csv encodec_scores.csv
done

echo "Done. Results written to: ${GLOBAL_OUTPUT_FILE}"
