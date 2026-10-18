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
set "SPLIT_RANDOM_SEED=2026"
set "BATCH_SIZE=24"
set "LEARNING_RATE=5e-4"
set "NUM_EPOCHS=800"
set "SAMPLE_STEPS=50"
set "IMG_SAVE_INTERVAL=10"
set "LPIPS_EVAL_INTERVAL=10"
set "EVAL_BATCH_SIZE=2"
set "RESUME=false"
set "USE_AMP=true"

REM ==================== OPTIONAL: FORCE A BACKEND ====================
REM Set a backend to skip auto-detection (run.py reads this):
REM   set "HANZIGEN_BACKEND=cuda"      NVIDIA
REM   set "HANZIGEN_BACKEND=rocm"      AMD, ROCm / HIP
REM   set "HANZIGEN_BACKEND=zluda"     AMD, ZLUDA on Windows
REM   set "HANZIGEN_BACKEND=directml"  AMD / Intel on Windows
REM   set "HANZIGEN_BACKEND=cpu"       no GPU

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "PRETRAINED_VQVAE_PATH=checkpoints/vqvae_!FILENAME!.pth"
set "MODEL_SAVE_PATH=checkpoints/ldm_!FILENAME!.pth"
set "TENSORBOARD_LOG_DIR=runs/LDM_!FILENAME!/"
set "SAMPLE_ROOT=samples_!FILENAME!/"

REM run.py resolves the backend and maps it to the right --device for you.
REM To force a backend, set HANZIGEN_BACKEND instead of editing this file.
uv run python run.py train_ldm.py ^
    --split_ratios "%TRAIN_SPLIT_RATIO%" "%VAL_SPLIT_RATIO%" ^
    --split_random_seed "%SPLIT_RANDOM_SEED%" ^
    --batch_size "%BATCH_SIZE%" ^
    --learning_rate "%LEARNING_RATE%" ^
    --num_epochs "%NUM_EPOCHS%" ^
    --pretrained_vqvae_path "%PRETRAINED_VQVAE_PATH%" ^
    --model_save_path "%MODEL_SAVE_PATH%" ^
    --tensorboard_log_dir "%TENSORBOARD_LOG_DIR%" ^
    --sample_root "%SAMPLE_ROOT%" ^
    --sample_steps "%SAMPLE_STEPS%" ^
    --img_save_interval "%IMG_SAVE_INTERVAL%" ^
    --lpips_eval_interval "%LPIPS_EVAL_INTERVAL%" ^
    --eval_batch_size "%EVAL_BATCH_SIZE%" ^
    --resume "%RESUME%" ^
    --use_amp "%USE_AMP%"

exit /b %errorlevel%