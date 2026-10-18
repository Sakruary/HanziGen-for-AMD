@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/target_font.ttf"
set "TRAIN_SPLIT_RATIO=0.8"
set "VAL_SPLIT_RATIO=0.2"
set "SPLIT_RANDOM_SEED=2026"
set "BATCH_SIZE=24"
set "LEARNING_RATE=1e-3"
set "NUM_EPOCHS=500"
set "IMG_SAVE_INTERVAL=10"
set "RESUME=false"
set "USE_AMP=true"

REM ==================== OPTIONAL: FORCE A BACKEND ====================
REM Set a backend to skip auto-detection (run.py reads this):
REM   set "HANZIGEN_BACKEND=cuda"      NVIDIA
REM   set "HANZIGEN_BACKEND=rocm"      AMD, ROCm / HIP
REM   set "HANZIGEN_BACKEND=zluda"     AMD, ZLUDA on Windows
REM   set "HANZIGEN_BACKEND=directml"  AMD / Intel on Windows
REM   set "HANZIGEN_BACKEND=cpu"       no GPU
REM An index is allowed, e.g. set "HANZIGEN_BACKEND=cuda:1" for the second GPU

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "MODEL_SAVE_PATH=checkpoints/vqvae_!FILENAME!.pth"
set "TENSORBOARD_LOG_DIR=runs/VQVAE_!FILENAME!"
set "SAMPLE_ROOT=samples_!FILENAME!/"

REM run.py resolves the backend and maps it to the right --device for you.
REM To force a backend, set HANZIGEN_BACKEND instead of editing this file.
uv run python run.py train_vqvae.py ^
    --split_ratios "%TRAIN_SPLIT_RATIO%" "%VAL_SPLIT_RATIO%" ^
    --split_random_seed "%SPLIT_RANDOM_SEED%" ^
    --batch_size "%BATCH_SIZE%" ^
    --learning_rate "%LEARNING_RATE%" ^
    --num_epochs "%NUM_EPOCHS%" ^
    --model_save_path "%MODEL_SAVE_PATH%" ^
    --tensorboard_log_dir "%TENSORBOARD_LOG_DIR%" ^
    --sample_root "%SAMPLE_ROOT%" ^
    --img_save_interval "%IMG_SAVE_INTERVAL%" ^
    --resume "%RESUME%" ^
    --use_amp "%USE_AMP%"

exit /b %errorlevel%