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
set "REFERENCE_FONTS_DIR=fonts/jigmo/"
set "CHARSET_PATH=charsets/gbk_coverage/target_font/missing.txt"
set "BATCH_SIZE=16"
set "SAMPLE_STEPS=50"
set "IMG_WIDTH=512"
set "IMG_HEIGHT=512"
REM Set HANZIGEN_BACKEND to force a compute backend; otherwise run.py auto-detects.

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "PRETRAINED_LDM_PATH=checkpoints/ldm_!FILENAME!.pth"
set "SAMPLE_ROOT=samples_!FILENAME!/"

if "%CHARSET_PATH%"=="auto" (
    set "CHARSET_PATH=charsets/jf7000_coverage/!FILENAME!/missing.txt"
    echo Auto-detected charset path: !CHARSET_PATH!
) else (
    echo Using manual charset path: %CHARSET_PATH%
)

REM run.py picks the compute device; set HANZIGEN_BACKEND to force one.
uv run python run.py inference.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --reference_fonts_dir "%REFERENCE_FONTS_DIR%" ^
    --charset_path "%CHARSET_PATH%" ^
    --pretrained_ldm_path "%PRETRAINED_LDM_PATH%" ^
    --batch_size "%BATCH_SIZE%" ^
    --sample_root "%SAMPLE_ROOT%" ^
    --sample_steps "%SAMPLE_STEPS%" ^
    --img_size "%IMG_WIDTH%" "%IMG_HEIGHT%"

exit /b %errorlevel%
