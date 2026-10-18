# AMD 顯示卡安裝指南

本文件說明如何在 **AMD 顯示卡**（以 Radeon RX 9070 XT / RDNA4 / `gfx1201` 為例）上執行 HanziGen。
本專案原本只針對 NVIDIA CUDA 設計，自 v1.4.0 起改為 **後端無關（backend-agnostic）**：
程式會自動偵測可用的運算後端，並針對各後端調整混合精度與資料搬移策略。

> [!IMPORTANT]
>
> AMD 顯示卡有 **三條可用路線**，請依你的作業系統與需求選擇：
>
> | 路線 | 平台 | 訓練速度 | 安裝難度 | 建議情境 |
> | --- | --- | --- | --- | --- |
> | **ROCm / HIP** | Linux 原生、WSL2、Windows（ROCm 預覽版 wheel） | ★★★★★ | 中 | **首選**，RX 9070 XT 可完整發揮 |
> | **ZLUDA** | Windows 原生 | ★★★★ | 中高 | 想在 Windows 原生訓練，且願意手動放 DLL |
> | **DirectML** | Windows 原生 | ★★ | 低 | 想快速跑起來、可接受較慢 |
>
> 三條路線都可以用同一份程式碼，**不需要修改任何一行訓練程式**。
>
> **已實機驗證**：本專案在 Windows 11 + Radeon RX 9070 XT 上，**ROCm 與 DirectML
> 兩條路線都完整跑通**了 VQ-VAE 訓練、LDM 訓練（含 PSNR / SSIM / LPIPS 指標）與
> DDIM 推論，測試套件全數通過。實測效能（512×512、batch 8 的 conv + groupnorm 訓練步）：
>
> | 後端 | 每步耗時 | 吞吐 | 混合精度 |
> | --- | --- | --- | --- |
> | **ROCm / HIP** | **52 ms** | **153 張/秒** | 支援 fp16 autocast |
> | DirectML | 171 ms | 47 張/秒 | 不支援，退回 fp32 |
> | 純 CPU（Ryzen 7000/9000） | 699 ms | 11 張/秒 | 不支援 |
>
> **結論：在 Windows 上 ROCm 是 9070 XT 的最佳路線，比 DirectML 快 3.3 倍。**
> ZLUDA 因開發環境限制無法實測，文件內容來自公開資料。
>
> [!NOTE]
>
> 本文件的版本、索引網址與已知問題，整理自 2026 年 10 月的公開資料，
> 完整來源清單與逐條考證記錄見 [AMD 後端技術调研報告](research-amd-pytorch-2026.md)。
> ROCm 生態變動很快，**安裝前請以 AMD 官方文件為最終依據**。

---

## 目錄

1. [HanziGen 如何選擇後端](#1-hanzigen-如何選擇後端)
2. [路線 A：ROCm / HIP（建議）](#2-路線-arocm--hip建議)
3. [路線 B：ZLUDA（Windows 原生）](#3-路線-bzludawindows-原生)
4. [路線 C：DirectML（Windows 原生，最簡單）](#4-路線-cdirectmlwindows-原生最簡單)
5. [驗證安裝](#5-驗證安裝)
6. [針對 RX 9070 XT 的效能調校](#6-針對-rx-9070-xt-的效能調校)
7. [疑難排解](#7-疑難排解)
8. [本次為了 AMD 做了哪些修改](#8-本次為了-amd-做了哪些修改)

---

## 1. HanziGen 如何選擇後端

所有硬體判斷都集中在 [`utils/hardware/backends.py`](../utils/hardware/backends.py)，
其餘程式碼只透過 [`utils/hardware/hardware_utils.py`](../utils/hardware/hardware_utils.py) 取得裝置與混合精度物件。

偵測順序（`--device` 未指定或設為 `auto` 時）：

1. `torch.cuda.is_available()` 為真且 `torch.version.hip` 有值 → **ROCm / HIP**（AMD）
2. `torch.cuda.is_available()` 為真且裝置指紋不像 NVIDIA → **ZLUDA**（AMD）
3. `torch.cuda.is_available()` 為真 → **CUDA**（NVIDIA）
4. `torch.backends.mps.is_available()` → **MPS**（Apple Silicon）
5. 可以匯入 `torch_directml` → **DirectML**（AMD / Intel，Windows）
6. 以上皆非 → **CPU**

### `--device` 可接受的值

| 值 | 意義 | 實際 torch device |
| --- | --- | --- |
| `auto`（預設／省略） | 自動偵測 | 依上表 |
| `cuda`、`cuda:0`、`cuda:1` | 標準 PyTorch 字串 | `cuda:n` |
| `rocm`、`hip`、`amd` | 強制 HIP 路徑 | `cuda:n`（ROCm 借用 cuda 命名空間） |
| `zluda` | 強制 ZLUDA 路徑 | `cuda:n` |
| `directml`、`dml` | 強制 DirectML 路徑 | `privateuseone:n` |
| `mps`、`metal` | Apple Silicon | `mps` |
| `cpu` | 純 CPU | `cpu` |

也可以用環境變數強制指定，適合寫在批次檔或 CI 中：

```bash
# Windows (PowerShell)
$env:HANZIGEN_BACKEND = "rocm"

# Linux / WSL2
export HANZIGEN_BACKEND=rocm
```

### 混合精度（`USE_AMP`）在各後端的行為

| 後端 | `torch.amp.autocast` | `torch.amp.GradScaler` | HanziGen 的行為 |
| --- | --- | --- | --- |
| CUDA | 支援（`cuda` + fp16） | 支援 | 正常啟用 |
| ROCm / HIP | 支援（`cuda` + fp16） | 支援 | 正常啟用 |
| ZLUDA | 依 ZLUDA 版本而定 | 支援 | 啟用，但若出現 NaN 請改為 `false` |
| DirectML | **不支援** | **不支援** | **自動退回 float32** 並印出警告 |
| CPU | 支援（但無意義） | 不支援 | 自動退回 float32 |

也就是說：**在 DirectML 上把 `USE_AMP` 設為 `true` 不會出錯，但也不會加速**，
程式會自動停用並告訴你原因。

---

## 2. 路線 A：ROCm / HIP（建議）

### 2-1 前置條件

- **Linux（原生 ROCm）**：安裝 AMD 官方的 `amdgpu` 核心模組與 ROCm runtime，
  並把使用者加入 `render`、`video` 群組。
- **WSL2**：在 Windows 安裝 AMD 驅動，再於 WSL2 內安裝 ROCm（WSL 的 ROCm 由驅動提供，不需重裝核心模組）。
- **Windows 原生**：需要 AMD 官方的 ROCm on Windows 預覽版 wheel（見 2-3）。

確認驅動看得到顯示卡：

```bash
# Linux / WSL2
rocminfo | grep -i gfx      # RX 9070 XT 應該顯示 gfx1201
rocm-smi                    # 應該列出你的顯示卡
```

### 2-2 建立環境

以下步驟已於 Windows 11 + RX 9070 XT 實測成功。

```bash
# 1) 建立專用環境（不要動到主要的 .venv）
uv venv --python 3.12 .venv-amd-rocm
```

Windows 上請使用 **AMD 的 TheRock 索引**：

| 來源 | 索引網址 | 說明 |
| --- | --- | --- |
| **TheRock（AMD 官方）** | `https://rocm.nightlies.amd.com/v2/gfx120X-all/` | Windows 唯一可用來源，輪子名稱含 `gfx120X-all` |
| `download.pytorch.org` 的 `rocm<版本>` | `https://download.pytorch.org/whl/rocm7.2` 等 | **僅有 Linux（manylinux）輪子**，Windows 無法使用 |

> [!IMPORTANT]
>
> 網路上大量教學把 `download.pytorch.org/whl/rocmX.Y` 推薦給 Windows 使用者，
> 那是**錯的** —— 該索引不含 `win_amd64` 輪子。

安裝時有**兩個陷阱**會讓你裝到 CPU 版而不是 ROCm 版（我實際踩到了）：

1. 加上 `--extra-index-url https://pypi.org/simple` 會讓 PyPI 上**版本號更高**的
   CPU 輪子勝出（結果裝成 `2.14.1+cpu`）。
2. ROCm 輪子帶 local version tag（`+rocm...`），所以單純寫 `torch==2.10.0`
   可能被 CPU 輪子滿足。

因此**必須鎖到完整的 local version，並加上 `--no-deps`**：

```powershell
$PY = ".venv-amd-rocm\Scripts\python.exe"
$IDX = "https://rocm.nightlies.amd.com/v2/gfx120X-all/"

# 1) 非 torch 的執行期依賴先從 PyPI 裝
uv pip install --python $PY filelock fsspec jinja2 networkx sympy numpy pillow `
  typing-extensions setuptools --index-url https://pypi.org/simple

# 2) 精確鎖定的 ROCm 三件套（--no-deps 避免被 PyPI 搶走）
uv pip install --python $PY --no-deps --index-url $IDX `
  "torch==2.10.0+rocm7.13.0a20260511" `
  "torchvision==0.25.0+rocm7.13.0a20260511" `
  "torchaudio==2.10.0+rocm7.13.0a20260511"

# 3) ROCm SDK：torch 在 import 時會載入，缺了會直接 ModuleNotFoundError
#    （rocm_sdk 模組來自 rocm 這個 meta 包，而它是 sdist，需要本地建置）
uv pip install --python $PY --no-deps --index-url $IDX `
  "rocm-sdk-core==7.13.0a20260511" `
  "rocm-sdk-libraries-gfx120x-all==7.13.0a20260511" `
  "rocm==7.13.0a20260511"
```

> [!NOTE]
>
> 第 3 步的 `rocm==7.13.0a20260511` 是 sdist，安裝時會**在本地建置**。
> 若 uv 的建置臨時目錄被系統權限或防毒軟體阻擋而失敗，把快取指到可寫位置後重試：
>
> ```powershell
> $env:UV_CACHE_DIR = "$PWD\.scratch\uv-cache"
> $env:TEMP = "$PWD\.scratch\tmp"
> ```

### 2-3 《重要》給 MIOpen 一個可寫的快取

MIOpen 會在**第一次用到某個卷積形狀時編譯 kernel** 並寫入快取。若快取不可寫，
它會不斷重試：表面上看起來是**第一次卷積整個卡住**，最後以
`RuntimeError: miopenStatusUnknownError` 或 `miopenStatusInternalError` 收場。

因此請指定可寫位置：

```powershell
New-Item -ItemType Directory -Force -Path "$PWD\.scratch\miopen\db","$PWD\.scratch\miopen\cache"
$env:MIOPEN_USER_DB_PATH     = "$PWD\.scratch\miopen\db"
$env:MIOPEN_CUSTOM_CACHE_DIR = "$PWD\.scratch\miopen\cache"
```

> [!TIP]
>
> 這個快取**跨執行有效**。第一次訓練會比較慢（每個新卷積形狀要編譯一次），
> 之後就快了。實測 512×512、batch 24 的第一個卷積約 **46 秒**，
> 之後同一個卷積只要 **0.002 秒** —— 差了兩萬倍以上，所以第一個 epoch
> 特別慢是正常的，**不要在第一個 epoch 就跑完前中斷**。
>
> 若希望每次開新終端機都自動生效，可寫進使用者環境變數：
>
> ```powershell
> [Environment]::SetEnvironmentVariable("MIOPEN_CUSTOM_CACHE_DIR", "D:\cache\miopen\cache", "User")
> [Environment]::SetEnvironmentVariable("MIOPEN_USER_DB_PATH",     "D:\cache\miopen\db",    "User")
> ```

> [!WARNING]
>
> **看到這些訊息不代表失敗**：
>
> | 訊息 | 意義 |
> | --- | --- |
> | `warning: xnack 'Off' was requested for a processor that does not support it!` | LLVM 對 gfx1201 的正常噪音。**只在真正編譯 kernel 時出現**，可以據此判斷進度 |
> | `MIOpen Error: ... Internal error while accessing SQLite database: unable to open database file` | **這個要處理** —— find-db 打不開，等於每次都重新搜索編譯。請照上面設好 `MIOPEN_USER_DB_PATH` |
> | `RuntimeError: miopenStatusInternalError` | 上一條的後果，通常伴隨出現 |

### 2-4 驗證與訓練

```powershell
# 確認真的走 HIP 而不是 CPU
& $PY -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 預期：7.13.26176 True AMD Radeon RX 9070 XT

& $PY check_hardware.py --device rocm --benchmark
& $PY train_vqvae.py --device rocm
& $PY train_ldm.py --device rocm
```

ROCm 上 `torch.cuda.is_available()` 為 `True`（HIP 借用 CUDA 命名空間），
所以 `--device rocm` 與 `--device cuda` 都會走同一條路徑。

> [!WARNING]
>
> **驅動陷阱**：Windows 上部分 Radeon 驅動版本（例如 26.7.1）會在 GPU 閒置約
> 10 秒後把仍在使用的 HIP 顯存換出，導致訓練中途出現無預期的錯誤或崩潰。
> 若遇到這種情況，請換用 AMD 為 ROCm 發行的專用驅動分支
> （"AMD Software: Adrenalin Edition for ROCm"），或退回先前的驅動版本。

訓練指令：

```powershell
& $PY check_hardware.py --device rocm --benchmark
& $PY train_vqvae.py --device rocm --batch_size 8 --use_amp true
& $PY train_ldm.py   --device rocm --use_amp true
```

`USE_AMP` 在 ROCm 上**確實有效**：`check_hardware.py` 的
「float16 autocast」檢查會顯示 `out dtype=torch.float16`。

---

## 3. 路線 B：ZLUDA（Windows 原生）

ZLUDA 會用 AMD 的 HIP 實作取代 NVIDIA 的 CUDA runtime DLL，因此 **PyTorch 會以為自己在 NVIDIA 顯示卡上**，
`torch.cuda.is_available()` 會回傳 `True`，`device.type` 也是 `"cuda"`。
好處是幾乎所有 CUDA 路徑都能直接跑；代價是需要手動放置 DLL，且版本相容性要自己顧。

> [!CAUTION]
>
> **ZLUDA 自 v6 起已回到「個人愛好專案」狀態**（AMD 停止資助），官方重心已轉向其他方向。
> 目前活躍的是社群 fork 與打包專案。**請把 ZLUDA 當成「讓 CUDA 程式跑得起來」的相容層，
> 而不是長期生產方案**；能走 ROCm 就走 ROCm。

### 3-1 安裝 PyTorch（CUDA 版）

```bash
uv venv --python 3.12
uv sync --no-install-package torch --no-install-package torchvision --no-install-package torchaudio
uv pip install -r constraints/amd-zluda.txt
```

注意這裡刻意安裝 **CUDA 版** PyTorch —— ZLUDA 的工作就是冒充 CUDA。

### 3-2 放入 ZLUDA DLL

1. 安裝 **AMD HIP SDK for Windows**（ZLUDA 需要其中的 HIP runtime；缺少時會出現
   類似 `Could not find module ...\ROCm\<版本>\bin\hiprtc0XXX.dll` 的錯誤）。
   ZLUDA 也會讀取 `HIP_PATH` 環境變數，請確認它指向 HIP SDK 安裝目錄。
2. 從 ZLUDA（或你選用的社群 fork）取得 Windows 壓縮檔。多數 fork 會附帶
   針對特定 HIP SDK 版本的修補腳本（例如 `patch-zluda-hip64.bat`），
   **請使用與你安裝的 HIP SDK 版本相符的腳本**。
3. 將 ZLUDA 提供、**與 CUDA 函式庫同名**的 DLL 覆蓋到 **PyTorch 的 `lib` 目錄**：

   ```
   .venv/Lib/site-packages/torch/lib/
   ```

   典型檔案包含 `nvcuda.dll`（驅動層）、`cublas64_*.dll` / `cublasLt64_*.dll`、
   `cusparse64_*.dll`、`cusolver64_*.dll`、`cufft64_*.dll`、`nvml.dll`，
   以及 **cuDNN**（`cudnn64_*.dll`）。**Conv 網路缺 cuDNN 會直接失敗**，
   本專案的 VQ-VAE / U-Net 全部是 conv 網路，請務必確認 cuDNN 有到位。
4. 先備份原本的同名檔案（若存在），以便還原。

> [!TIP]
>
> ZLUDA 附帶一個 `cuda_check` 工具，是驗證安裝是否成功最快的方法。
> 另外，由於各家 fork 的檔案清單與腳本名稱不同，**請以你所選 fork 的 README 為準**。

### 3-3 設定環境變數

```powershell
$env:HIP_VISIBLE_DEVICES = "0"
$env:AMD_SERIALIZE_KERNEL  = "0"
$env:HANZIGEN_BACKEND      = "zluda"
# 若不穩定，可關閉部分最佳化
# $env:AMD_LOG_LEVEL = "0"
```

### 3-4 驗證與訓練

```bash
uv run python check_hardware.py --device zluda --benchmark
```

`check_hardware.py` 會顯示「ZLUDA detected」的提示，並印出
`torch.cuda.get_device_name()` 的結果 —— 若是 `AMD Radeon RX 9070 XT`，代表設定成功。

ZLUDA 對 float16 autocast 的支援依版本而異。第一次訓練建議先關閉：

```bash
uv run python train_vqvae.py --device zluda --use_amp false
```

確認穩定後再試 `--use_amp true`；若出現 `NaN` 或 device-side assert，就維持 `false`。

---

## 4. 路線 C：DirectML（Windows 原生，最簡單）

`torch-directml` 由微軟維護，直接以 DirectML 驅動 AMD / Intel / NVIDIA 顯示卡，
**不需要 ROCm，也不需要放 DLL**，是 Windows 上最快能跑起來的路線。

> [!WARNING]
>
> 這個套件**自 2024 年 9 月起就沒有新版本**（最新版為 `0.2.5.dev240914`），
> 且 DirectML 並非 AMD 在 2026 年的首選路徑。請把它當成「臨時能跑」的方案。

### 4-1 安裝

DirectML 的 PyTorch 與本專案預設的 CUDA 版互斥，因此**建議另開一個專用環境**，
不要動到主要的 `.venv`：

```bash
uv venv --python 3.12 .venv-amd
uv pip install --python .venv-amd/Scripts/python.exe -r constraints/amd-directml.txt
```

> [!TIP]
>
> 之後所有指令都改用這個環境的直譯器即可，例如
> `.venv-amd/Scripts/python.exe check_hardware.py`。
> 若 `uv` 的預設快取目錄不可寫（例如受限的 Windows 帳號），先指向工作區：
>
> ```powershell
> $env:UV_CACHE_DIR = "$PWD\.scratch\uv-cache"
> $env:TEMP = "$PWD\.scratch\tmp"     # uv/pip 也需要可寫的暫存目錄
> ```

> [!IMPORTANT]
>
> **`uv run` 會使用 `.venv`，不是 `.venv-amd`。**
>
> `uv run` 綁定專案的主要環境 `.venv`（由 `uv sync` 建立，內含 CUDA 版 torch）。
> 在 AMD 機器上直接執行 `uv run python check_hardware.py` 會得到 `CPU` 後端，
> 因為那個環境裡的 torch 看不到你的顯示卡。
>
> 使用 DirectML 時請直接呼叫該環境的直譯器，或把主要環境也換成 AMD 版本：
>
> ```bash
> # 方法一（推薦）：直接指定 DirectML 環境
> .venv-amd/Scripts/python.exe check_hardware.py --device directml
> .venv-amd/Scripts/python.exe run.py train_vqvae.py
>
> # 方法二：讓 .venv 本身也走 AMD 後端（會失去 NVIDIA 支援）
> uv venv --python 3.12
> uv pip install -r constraints/amd-directml.txt
> ```

> [!CAUTION]
>
> `torch-directml` 自帶一份特定版本的 PyTorch（實測為 torch 2.4.1 + torchvision 0.19.1），
> **不能**與另外安裝的 `torch` 並存。請不要同時執行 `uv sync` 與這個 constraints 檔。
>
> 也因為它綁定 `torchvision==0.19.1`，**無法**把它宣告成 `pyproject.toml` 的
> optional extra —— 那會讓整個依賴圖無法求解，連 `uv lock` 都會失敗。這也是本專案
> 改用 `constraints/` 檔案的原因。
>
> 另外，`torch-directml` 的輪子只發佈到 **Python 3.12** 為止（其最後一次建置在
> Python 3.13 釋出之前），這也是本專案 `requires-python` 訂為 `>=3.12`、
> 並在文件中建議直接用 3.12 的原因。

裝完之後請確認真的看到顯示卡：

```bash
uv run python -c "import torch_directml as d; print(d.device_count(), d.device_name(0))"
```

在 RX 9070 XT 上應該印出 `1 AMD Radeon RX 9070 XT`。

### 4-2 驗證與訓練

```bash
uv run python check_hardware.py --device directml --benchmark
uv run python train_vqvae.py --device directml --batch_size 4
```

`check_hardware.py` 會顯示「DirectML does not support torch.amp ...」的提示，
代表程式正確偵測到限制。**不要加 `--use_amp true`**，加了也只會被自動忽略。

在 RX 9070 XT 上的實測結果（僅供對照，非效能承諾）：

| 項目 | 結果 |
| --- | --- |
| 後端偵測 | `DirectML (directml)`，裝置名 `AMD Radeon RX 9070 XT` |
| 9 項算子檢查 | 全部 PASS |
| VQ-VAE 訓練（小資料集，2 epoch） | 約 1.2 秒 |
| 測試套件 | 69 passed, 2 skipped |

> [!NOTE]
>
> 訓練時可能看到這類訊息，屬於正常現象：
>
> ```
> UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported on
> the DML backend and will fall back to run on the CPU.
> ```
>
> 這是 DirectML 算子覆蓋不全，PyTorch 自動把該算子丟回 CPU 執行。功能正確，
> 但會拖慢速度 —— 這也是 DirectML 只建議當「臨時能跑」方案的原因。

### 4-3 何時該放棄 DirectML

DirectML 的 PyTorch 後端**算子覆蓋不完整，而且回退機制不可靠** ——
遇到不支援的算子時，有時會直接報錯而不是退回 CPU（例如 `torch.eye`）。
它也沒有 autocast、沒有 `torch.compile`、沒有 Triton，並且走的是 D3D12 通用路徑，
**完全用不到 RDNA4 的 FP8 / WMMA 新硬體**，效能自然遠低於 ROCm。

若 `check_hardware.py` 出現大量 `FAIL`，或訓練速度慢到無法接受，
請改用路線 A 或 B。資料準備階段（`analyze_font.py`、`prepare_dataset.py`、
`split_dataset.py`、`convert_to_svg.py`）本身不太吃 GPU，用 DirectML 沒問題。

---

## 5. 驗證安裝

不論走哪條路線，都先執行硬體診斷腳本：

```bash
uv run python check_hardware.py
```

它會依序輸出：

1. **環境資訊** —— 作業系統、Python、PyTorch、`torch.version.cuda` / `torch.version.hip`。
   這是分辨 CUDA 版與 ROCm 版最直接的方法。
2. **選中的後端** —— 後端名稱、廠商、裝置名稱、架構、VRAM，以及 `torch.cuda` 的詳細指紋。
3. **功能檢查** —— 逐一實際執行本專案會用到的算子：

   | 檢查項 | 對應的 HanziGen 元件 |
   | --- | --- |
   | `conv2d (stride 2 down)` | `VQVAEDownBlock` |
   | `conv_transpose2d (stride 2 up)` | `VQVAEUpBlock` |
   | `group_norm + silu + tanh` | 所有區塊的歸一化與激活 |
   | `codebook distance + argmin` | `VQVAEQuantizer` 的最近鄰查表 |
   | `embedding lookup` | `VQVAEQuantizer.emb` |
   | `autograd step` | 訓練迴圈的 backward + Adam |
   | `float16 autocast` | `--use_amp true` |
   | `grad scaler step` | `--use_amp true` |
   | `device -> host transfer` | 取樣圖輸出 |

4. **效能基準**（加上 `--benchmark`）—— 一段 conv + groupnorm 的
   forward + backward，並附上各後端的參考區間。

若某項顯示 `FAIL`，請加上 `--strict` 讓腳本以非零狀態結束，方便寫進 CI。

---

## 6. 針對 RX 9070 XT 的效能調校

RX 9070 XT（RDNA4、`gfx1201`、16 GB VRAM）的建議設定：

| 參數 | 建議值 | 說明 |
| --- | --- | --- |
| `--device` | `rocm` 或 `auto` | 不要用 `cuda`，避免在其他機器上誤判 |
| `--batch_size` | 16（VQ-VAE）、32（LDM） | 16 GB VRAM 足夠；先用 `check_hardware.py` 確認 |
| `--use_amp` | ROCm / ZLUDA 為 `true`，DirectML 為 `false` | RDNA4 原生支援 fp16 |
| `--num_workers` | 4～8 | 資料載入是 AMD 路徑最常見的瓶頸 |

其他建議：

- **資料載入**：`num_workers` 調高到 CPU 實體核心數，可明顯改善 GPU 閒置。
- **首次評估會下載 528 MB 權重**：LDM 訓練的 LPIPS 指標會下載 VGG 權重到
  `~/.cache/torch`。若你的家目錄不可寫、或系統碟空間不足，請先指定快取位置：

  ```bash
  # Linux / WSL2
  export TORCH_HOME=/path/to/writable/cache

  # Windows (PowerShell)
  $env:TORCH_HOME = "D:\cache\torch"
  ```

- **`channels_last`**：本專案的張量都是 `NCHW` 且以 `Conv2d` / `ConvTranspose2d` 為主，
  ROCm 對 `NCHW` 已有良好最佳化，**不需要**改成 `channels_last`。
- **`torch.compile`**：ROCm 的 Windows 支援仍不穩定（`gfx1201` 上 JIT 編譯有已知問題），
  本專案預設不使用。若你在 Linux + ROCm 上想嘗試，可自行在模型外層包一層 `torch.compile`。
- **VRAM**：若出現 OOM，先降 `--batch_size`，再考慮 `--eval_batch_size`。
- **監控**：Linux 用 `rocm-smi`，Windows 用 AMD Software 的效能頁面觀察 GPU 使用率。
  若使用率長期低於 60%，瓶頸通常在資料載入而非顯示卡。

---

## 7. 疑難排解

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `check_hardware.py` 顯示 `CPU` | 沒有偵測到任何 GPU 後端 | 檢查驅動；確認裝的是 ROCm/ZLUDA/DirectML 版 torch，而非 CPU 版 |
| `torch.version.hip` 是 `None` | 裝到 CPU 或 CUDA 版 torch | 重新執行 `uv pip install -r constraints/amd-rocm.txt` |
| `RuntimeError: Device 'rocm' was requested but torch.cuda.is_available() is False` | 驅動或 runtime 沒裝好 | `rocminfo` 確認 `gfx1201` 有出現；只有 Linux/WSL2 或 Windows ROCm 預覽版才支援 |
| `RuntimeError: Device 'directml' was requested but torch_directml could not be imported` | 沒裝 `torch-directml` | `uv pip install -r constraints/amd-directml.txt` |
| 混合精度警告，訓練仍繼續 | DirectML 不支援 AMP | 正常行為，不需要處理 |
| 訓練中出現 `NaN` | ZLUDA 的 fp16 不穩定 | 加上 `--use_amp false` |
| `RuntimeError: expected scalar type Half but found Float` | 舊版程式碼在 autocast 下做最近鄰查表 | 本專案已修正（見第 8 節）；請確認 `models/vqvae/vqvae_encoder_decoder.py` 是最新版 |
| 速度遠低於參考區間 | 驅動過舊、其他程式佔用 GPU、或資料載入瓶頸 | 更新驅動；提高 `--num_workers`；關閉其他 GPU 程式 |
| `import torch` 在 Windows 失敗，找不到 DLL | ROCm 輪子需要 Visual C++ 執行期提供的 `libomp140.x86_64.dll` | 安裝微軟 **Visual C++ Redistributable**（x64） |
| 訓練跑一段時間後莫名崩潰 | 部分 Radeon 驅動會在 GPU 閒置約 10 秒後換出仍在使用的顯存 | 換用 AMD 的 ROCm 專用驅動分支，或退回較舊的驅動版本 |
| 出現 `hipErrorInvalidValue`，且發生在注意力運算之後 | 某些 torch + ROCm 修補版本組合在 `gfx1201` 上有已知回歸 | 同一 ROCm 系列換一個修補版本（例如改用 `+rocm7.14.0` 而非 `7.14.1`） |
| `pip install` 找不到 `torch-directml` | 使用了國內鏡像站 | 改用官方 PyPI，或在 constraints 檔指定 `--index-url https://pypi.org/simple` |
| Windows 上 `download.pytorch.org/whl/rocmX.Y` 裝不到 | 該索引只有 Linux 輪子 | 改用 AMD 的 TheRock 索引（見第 2-2 節） |
| ROCm 安裝後 `torch.__version__` 是 `+cpu` | `--extra-index-url` 讓 PyPI 的高版本號勝出 | 鎖定完整 local version 並加 `--no-deps`（見第 2-2 節） |
| `import torch` 報 `No module named 'rocm_sdk'` | 缺少 ROCm SDK 套件 | 安裝 `rocm-sdk-core`、`rocm-sdk-libraries-gfx120x-all`、`rocm`（見第 2-2 節第 3 步） |
| 第一次卷積卡住很久，最後 `miopenStatusUnknownError` | MIOpen 快取不可寫，一直重試 | 設定 `MIOPEN_USER_DB_PATH` 與 `MIOPEN_CUSTOM_CACHE_DIR`（見第 2-3 節） |
| `MIOpen Error: unable to open database file` + `miopenStatusInternalError` | MIOpen 的 SQLite find-db 打不開，每個卷積都要重新搜索編譯 | 同上。這是**第一個 epoch 特別慢**最常見的原因 |
| 一直跳出 `xnack 'Off' was requested ...` | LLVM 對 gfx1201 的正常噪音，只在編譯 kernel 時出現 | 不用處理。可用它判斷「還在編譯」而非「卡死」 |
| 第一個 epoch 很慢，之後變快 | MIOpen 首次為每個卷積形狀編譯 kernel（實測 46 秒 → 0.002 秒） | 正常。**不要在跑完第一個 epoch 前中斷**，否則下次還要重編 |
| 出現 `Weights only load failed ... _rebuild_device_tensor_from_numpy` | 該 checkpoint 是用舊版程式碼（或在 DirectML 上以 `torch.save(model.state_dict())`）存的，內含 `privateuseone` 張量 | 本專案已改用 `save_state_dict()` 存檔（先搬到 CPU）。請以現行版本重新訓練該階段 |
| 測試或訓練中途出現 `device_ready_queues_` 的 INTERNAL ASSERT | 第一個 backward 是 CPU 運算時，PyTorch autograd 引擎沒有為 `privateuseone` 建立佇列（DirectML 的初始化順序問題） | 已於 `tests/conftest.py` 以 session 層級的暖機 backward 修正；若你自己的指令碼遇到，請先對 GPU 做一次 backward 再進行 CPU 運算 |

若問題仍在，請把 `uv run python check_hardware.py --benchmark` 的完整輸出附在 issue 中。

---

## 8. 本次為了 AMD 做了哪些修改

原本的程式碼在四處硬性假設了 NVIDIA/CUDA 環境，以下是對應的修正：

| 檔案 | 原本 | 現在 |
| --- | --- | --- |
| `utils/hardware/backends.py` | 不存在 | **新增**。後端列舉、偵測、裝置字串正規化、AMP 能力查詢、DirectML 安全匯入（含 watchdog，避免無 GPU 時卡死） |
| `utils/hardware/hardware_utils.py` | `select_device` 只會回傳 `mps` / `cuda` / `cpu` | 支援 `rocm` / `zluda` / `directml` / `auto` / `HANZIGEN_BACKEND`，並新增 `print_backend_banner`、`resolve_amp_settings`、`enable_fast_math` |
| `train_vqvae.py`、`train_ldm.py` | `GradScaler(device=device)`，在 DirectML 上直接崩潰 | 改用 `make_grad_scaler`；不支援時自動退回 float32 並警告 |
| `models/vqvae/vqvae.py`、`models/ldm/ldm.py` | `autocast(device_type=self.device.type)`；`scaler is None` 時走另一條分支 | 改用 `autocast_context`；兩條分支合併為一條，DirectML 上也不會出錯 |
| `datasets/loader.py` | `pin_memory=True if device.type == "cuda"` | 改用 `is_accelerator`，涵蓋 `privateuseone` 等其他加速器 |
| `utils/image/image_utils.py` | `T.ToPILImage()(image)` 假設張量在 CPU | 非 CPU 張量先 `.cpu()`，並轉為 `float32` |
| `models/vqvae/vqvae_encoder_decoder.py` | 最近鄰查表直接在 autocast 下做，fp16 會精度不足甚至型別不符 | 以 `no_autocast` 包住，強制 fp32 計算距離後再轉回原 dtype |
| `models/vqvae/vqvae.py`、`models/ldm/ldm.py` | `torch.save(self.state_dict(), path)` 直接存下裝置張量 | 改用 `save_state_dict()`，先搬到 CPU 再存（跨後端可攜、可用 `weights_only=True` 載入） |
| `train_vqvae.py`、`train_ldm.py`、`inference.py`、`models/ldm/ldm.py` | `torch.load(map_location=device, ...)`，其中 `device` 是 `torch.device`，在 DirectML 上直接崩潰 | 改用 `load_state_dict()`，一律載到 CPU 後再由模型 `.to(device)` |
| `utils/hardware/backends.py` | 不存在 | 裝置名稱去除 DirectML 回報的 `\x00` 填補位元組；新增 `reset_device_cache()` |
| `pyproject.toml`、`constraints/` | 只指向 `pytorch-cu118` | 保留 CUDA 為預設，另提供 CPU / ROCm / ZLUDA / DirectML 四份 constraints 檔；**移除 directml extra**（見第 4-1 節說明） |
| `README.md`、`docs/` | 硬體需求寫「NVIDIA 顯示卡」 | 改為列出各後端需求，並連到本文件 |
| `check_hardware.py` | 不存在 | **新增**。硬體診斷與效能基準腳本 |
| `tests/` | 不存在 | **新增**。不需 GPU 即可驗證後端偵測、啟動器與真實訓練批次；`conftest.py` 處理裝置快取隔離與 DirectML autograd 暖機 |

---

## 延伸閱讀

- [NVIDIA 顯示卡安裝指南](NVIDIA_SETUP.md)
- [專案 README](../README.md)
