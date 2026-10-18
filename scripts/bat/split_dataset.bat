@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/target_font.ttf"
set "TRAIN_SPLIT_RATIO=0.8"
set "VAL_SPLIT_RATIO=0.2"
set "SPLIT_RANDOM_SEED=2025"

REM ==================== DO NOT MODIFY BELOW ====================
REM run.py picks the compute device; set HANZIGEN_BACKEND to force one.
uv run python run.py split_dataset.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --split_ratios "%TRAIN_SPLIT_RATIO%" "%VAL_SPLIT_RATIO%" ^
    --split_random_seed "%SPLIT_RANDOM_SEED%"

exit /b %errorlevel%