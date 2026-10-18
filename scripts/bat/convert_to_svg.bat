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
set "BLACKLEVEL=0.5"
set "TURDSIZE=2"
set "ALPHAMAX=1.0"
set "OPTTOLERANCE=0.2"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "SAMPLE_ROOT=samples_!FILENAME!/"

if "!TIMESTAMP!" == "auto" (
    set "LATEST_INFERENCE_DIR="
    set "LATEST_TIME=0"
    
    for /d %%d in ("!SAMPLE_ROOT!ldm_inference_*") do (
        set "DIR_NAME=%%~nd"
        set "CURRENT_TIMESTAMP=!DIR_NAME:ldm_inference_=!"
        if "!CURRENT_TIMESTAMP!" gtr "!LATEST_TIME!" (
            set "LATEST_TIME=!CURRENT_TIMESTAMP!"
            set "LATEST_INFERENCE_DIR=%%d"
        )
    )
    
    set "TIMESTAMP=!LATEST_TIME!"
    echo Auto-detected timestamp: !TIMESTAMP!
) else (
    echo Using manual timestamp: !TIMESTAMP!
)

set "INPUT_DIR=%SAMPLE_ROOT%ldm_inference_!TIMESTAMP!/infer/gen/"
set "OUTPUT_DIR=%SAMPLE_ROOT%ldm_inference_!TIMESTAMP!/svg/"

uv run python convert_to_svg.py ^
    --input_dir "%INPUT_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --blacklevel "%BLACKLEVEL%" ^
    --turdsize "%TURDSIZE%" ^
    --alphamax "%ALPHAMAX%" ^
    --opttolerance "%OPTTOLERANCE%"

exit /b %errorlevel%
