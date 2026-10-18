#!/bin/bash

# ==================== USER CONFIGURATIONS ====================
TARGET_FONT_PATH="fonts/target_font.ttf"
TRAIN_SPLIT_RATIO=0.8
VAL_SPLIT_RATIO=0.2
SPLIT_RANDOM_SEED=2025

# ==================== DO NOT MODIFY BELOW ====================
# run.py picks the compute device; export HANZIGEN_BACKEND to force one.
uv run python run.py split_dataset.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --split_random_seed "$SPLIT_RANDOM_SEED"

exit $?