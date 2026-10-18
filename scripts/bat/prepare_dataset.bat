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
set "IMG_WIDTH=512"
set "IMG_HEIGHT=512"
set "SAMPLE_RATIO=1.0"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "SOURCE_CHARSET_PATH=charsets/unihan_coverage/!FILENAME!/covered.txt"

if exist "data" (
    echo Removing existing data/ directory...
    rmdir /s /q "data"
)

uv run python prepare_dataset.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --reference_fonts_dir "%REFERENCE_FONTS_DIR%" ^
    --source_charset_path "%SOURCE_CHARSET_PATH%" ^
    --img_size "%IMG_WIDTH%" "%IMG_HEIGHT%" ^
    --sample_ratio "%SAMPLE_RATIO%"

exit /b %errorlevel%
