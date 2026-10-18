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
set "TIMESTAMP=auto"
set "EVAL_BATCH_SIZE=2"
REM Set HANZIGEN_BACKEND to force a compute backend; otherwise run.py auto-detects.

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "SAMPLE_ROOT=samples_!FILENAME!/"

if "!TIMESTAMP!" == "auto" (
    set "LATEST_LDM_DIR="
    set "LATEST_TIME=0"
    
    for /d %%d in ("!SAMPLE_ROOT!ldm_training_*") do (
        set "DIR_NAME=%%~nd"
        set "CURRENT_TIMESTAMP=!DIR_NAME:ldm_training_=!"
        if "!CURRENT_TIMESTAMP!" gtr "!LATEST_TIME!" (
            set "LATEST_TIME=!CURRENT_TIMESTAMP!"
            set "LATEST_LDM_DIR=%%d"
        )
    )
    
    set "TIMESTAMP=!LATEST_TIME!"
    echo Auto-detected timestamp: !TIMESTAMP!
) else (
    echo Using manual timestamp: !TIMESTAMP!
)

set "GENERATED_IMG_DIR=%SAMPLE_ROOT%ldm_training_!TIMESTAMP!/eval/gen/"
set "GROUND_TRUTH_IMG_DIR=%SAMPLE_ROOT%ldm_training_!TIMESTAMP!/eval/gt/"

REM run.py picks the compute device; set HANZIGEN_BACKEND to force one.
uv run python run.py compute_metrics.py ^
    --generated_img_dir "%GENERATED_IMG_DIR%" ^
    --ground_truth_img_dir "%GROUND_TRUTH_IMG_DIR%" ^
    --eval_batch_size "%EVAL_BATCH_SIZE%"

exit /b %errorlevel%
