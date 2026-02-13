#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
DATASETS_DIR="$OUTPUT_DIR/sim_data/runs"
MODELS_DIR="$OUTPUT_DIR/models"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "outputs directory not found at $OUTPUT_DIR"
  exit 1
fi

echo "This will delete all datasets in $DATASETS_DIR and all model artifacts in $MODELS_DIR."
echo "Please create your own backup if needed."

read -rp "Type DELETE (uppercase) to confirm you want to wipe the datasets/models: " CONFIRM_FIRST
if [[ "$CONFIRM_FIRST" != "DELETE" ]]; then
  echo "First confirmation failed. Aborting without touching any data."
  exit 1
fi

read -rp "Type CONFIRM to proceed with the deletion: " CONFIRM_SECOND
if [[ "$CONFIRM_SECOND" != "CONFIRM" ]]; then
  echo "Second confirmation failed. Aborting."
  exit 1
fi

echo "Removing datasets/models..."
rm -rf "$DATASETS_DIR" "$MODELS_DIR"

echo "Recreating empty dataset/model directories..."
mkdir -p "$DATASETS_DIR" "$MODELS_DIR" "$MODELS_DIR/versions"

echo "Wipe complete. Build new datasets/models as needed."
