@echo off
setlocal enabledelayedexpansion

REM ==================== MIOpen / ROCm kernel log ====================
REM Suppress the a_grid_desc_* / b_grid_desc_* / e_grid_desc_* debug spam that
REM MIOpen and hipBLASLt print to stderr while selecting / compiling convolution
REM kernels. Delete these lines if you need to debug kernel selection.
set "MIOPEN_LOG_LEVEL=1"
set "MIOPEN_ENABLE_LOGGING=0"
set "HIPBLASLT_LOG_LEVEL=0"
set "ROCBLAS_LOG_LEVEL=0"


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
