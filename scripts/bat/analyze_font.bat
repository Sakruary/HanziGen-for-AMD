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

REM ==================== DO NOT MODIFY BELOW ====================
uv run python analyze_font.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --reference_fonts_dir "%REFERENCE_FONTS_DIR%" ^
    --analyze_target_font ^
    --analyze_reference_fonts

exit /b %errorlevel%
