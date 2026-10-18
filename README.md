<div align="center">

![logo](./assets/logo.svg)

# 【字生字 · HanziGen】 for AMD

## 汉字字体生成与补全系统 · AMD 显卡适配版

</div>

> 本项目是原版 [**wangwenho/HanziGen**](https://github.com/wangwenho/HanziGen) 的 **AMD 显卡适配版**。
> 在保留原版完整功能的基础上，针对 **AMD ROCm / HIP**（并兼容 DirectML / ZLUDA）做了后端抽象与效率强化：
> ROCm 后端自动检测、MIOpen 缓存修复、bf16 / fp16 混合精度、VQ-VAE 训练坍缩检测。
> 此版本也支持 NVIDIA 显卡，但建议直接使用原版项目。

## 📌 项目简介

本项目论文〈[HanziGen: Supervised High-Fidelity Chinese Font Generation via Latent Diffusion Models](https://doi.org/10.1109/ICCE-Taiwan71481.2026.11652402)〉已发表于 2026 IEEE ICCE-Taiwan。

本系统整合了字体分析、模型训练与缺字生成等自动化流程，旨在协助设计师加速字体的开发与补全工作。

![pipeline](./assets/pipeline.svg)

### 方法概述

本项目采用潜在扩散模型（Latent Diffusion Model, LDM）作为图像生成核心。通过两阶段训练策略，模型能有效降低运算成本，并学习目标字体的风格特征，随后借助参考字体的结构引导，生成风格一致的缺失字形图像。

![concept](./assets/concept.svg)

### 主要特色

- **模块化流程**：整合字体分析至字形向量化一系列流程，并具备高度可定制性。
- **低硬件需求**：仅需 4 GB VRAM 的显卡，即可完成训练与推理。
- **多后端支持**：兼容 NVIDIA CUDA、AMD ROCm / ZLUDA / DirectML、Apple MPS 与纯 CPU，程序会自动检测并调整混合精度策略。
- **智慧混合精度**：支持 fp16 与 bf16（设 `HANZIGEN_AMP_DTYPE=bf16` 即可切换）；在 AMD 上会自动配置 MIOpen 缓存、避免触发 MIOpen 冷启动的基准测试，并开启 TF32 加速。
- **训练健康监测**：VQ-VAE 训练会实时检测码本坍缩／重建失效，提前停机避免浪费 GPU 时间，且只在模型健康时存储最佳检查点。
- **高品质输出**：支持生成分辨率达 512 × 512 像素以上的清晰字形图像。

> [!IMPORTANT]
>
> - 本项目非通用生成模型，每种目标字体均需独立完成训练
> - 本项目不提供字体文件，请自行准备目标字体与参考字体
> - 本项目不提供权重文件，请依照使用说明自行训练该模型

---

## 🏗️ 模型架构

本项目采用的潜在扩散模型由向量量化自编码器（VQ-VAE）与 U-Net 两大核心组件构成。VQ-VAE 负责将高分辨率的字形图像压缩至离散的潜在空间，以完整保留字体的风格与结构特征。随后，U-Net 于此空间中执行加噪与去噪流程，进而生成高品质且风格一致的字形图像。

### 第一阶段：训练 VQ-VAE

训练 VQ-VAE 将字形图像编码为离散潜在向量，并重建原始图像。训练目标如下：

$$
\mathcal{L}_{\text{VQ}}=\|\mathbf{x}-\hat{\mathbf{x}}\|_2^2+\|\text{sg}[\mathbf{z}_e]-\mathbf{z}_q\|_2^2+\beta\|\text{sg}[\mathbf{z}_q]-\mathbf{z}_e\|_2^2.
$$

### 第二阶段：训练 LDM

利用预训练的 VQ-VAE 将参考字形图像与目标字形图像分别编码为离散潜在向量，接著将参考向量与加噪后的目标向量拼接，输入 U-Net 预测杂讯。训练目标为最小化预测误差：

$$
\mathcal{L}_{\text{LDM}}=\mathbb{E}_{\mathbf{z}_0,\mathbf{c},\epsilon,t}\left[\|\epsilon-\epsilon_\theta([\mathbf{z}_t \| \mathbf{c}],t)\|_2^2\right].
$$

### 推理阶段：生成字形

推理过程从高斯杂讯出发，以参考字形为条件，通过 DDIM 演算法在潜在空间中迭代去噪，最终由解码器还原出对应的目标字形图像。

![architecture](./assets/architecture.svg)

---

## ✨ 生成样本

下图展示了字生字在三种不同风格之目标字体上的生成效果。每组样本由上至下依序为：参考字形图像、模型生成图像以及目标字形图像（若目标字体未收录该字，则以虚线叉号标示）。

![samples](./assets/samples-1.svg)

![samples](./assets/samples-2.svg)

---

## 🚀 使用说明

### 1. 环境部署 ⚙️

此步骤将建立本项目所需的 Python 环境并安装相关软件包。

> [!IMPORTANT]
>
> - **作业系统**：Linux、Windows（原生环境或 WSL2）
> - **硬件需求**：至少 4 GB VRAM 的显卡。支持 **NVIDIA**、**AMD**、**Apple Silicon** 与 **纯 CPU**
> - **驱动程序**：NVIDIA 需 CUDA 11.8 以上；AMD 依所选后端而定（见下方说明）

#### 1-1 下载本项目

```bash
git clone https://github.com/Sakruary/HanziGen-for-AMD.git
cd HanziGen-for-AMD
```

#### 1-2 建立 uv 环境

> [!NOTE]
>
> - 若尚未安装 uv，请参考 [uv 官方网站](https://docs.astral.sh/uv/#installation) 安装对应的平台版本

```bash
uv sync
```

> [!NOTE]
>
> - 本项目默认安装 CUDA 11.8 版本的 PyTorch。若需调整为 CPU 或其他 CUDA 版本，请依 [uv 官方文档](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) 修改 [`pyproject.toml`](./pyproject.toml) 的索引设置，并执行 `uv sync` 同步环境。

#### 1-3 选择运算后端

本项目支持多种运算后端，程序会在执行时自动检测。

| 你的显卡 | 建议后端 | 安装说明 |
| --- | --- | --- |
| NVIDIA | CUDA（`uv sync` 默认） | [NVIDIA 显卡安装指南](./docs/NVIDIA_SETUP.md) |
| AMD（Radeon RX 9070 XT 等） | ROCm / HIP（首选）、ZLUDA 或 DirectML | [AMD 显卡安装指南](./docs/AMD_SETUP.md) |
| Apple Silicon | MPS | 使用 `uv sync` 默认环境即可 |
| 无显卡 | CPU | 见 [`constraints/cpu.txt`](./constraints/cpu.txt) |

> [!TIP]
>
> **AMD 用户请注意**：本项目已在 Windows 11 + Radeon RX 9070 XT 上，
> **ROCm 与 DirectML 两条路线都实测跑通**完整训练与推理流程。
> 实测性能（512×512、batch 8 的 conv + groupnorm 训练步）：
>
> | 后端 | 每步耗时 | 吞吐 | 混合精度 |
> | --- | --- | --- | --- |
> | **ROCm / HIP** | **52 ms** | **153 张/秒** | 支持 fp16 |
> | DirectML | 171 ms | 47 张/秒 | 不支持 |
> | 纯 CPU | 699 ms | 11 张/秒 | 不支持 |
>
> **建议走 ROCm**（快 3.3 倍且支持混合精度），完整步骤见
> [AMD 显卡安装指南](./docs/AMD_SETUP.md)。
>
> 建议配置：直接把 `.venv` 建成 ROCm 环境（`uv run` 与 `scripts\bat\*.bat` 都绑定它）：
>
> | 目录 | 后端 | 用途 |
> | --- | --- | --- |
> | `.venv` | **ROCm / HIP**（torch 2.10.0+rocm7.13.0a20260511） | `uv run` 与 `.bat` 的默认环境 |
> | `.venv-amd`（可选） | DirectML（torch 2.4.1 + torch-directml） | 备援，不需 ROCm |

 ```powershell
 # ROCm（建议）
 uv venv --python 3.12 .venv
 # 接著依 docs/AMD_SETUP.md 第 2-2 节或本页“AMD 快速开始”安装
uv venv --python 3.12 .venv

$PY  = ".venv\Scripts\python.exe"
$IDX = "https://rocm.nightlies.amd.com/v2/gfx120X-all/"

# 1) torch 的运行时依赖
uv pip install --python $PY filelock fsspec jinja2 networkx sympy numpy pillow `
  typing-extensions setuptools --index-url https://pypi.org/simple

# 2) ROCm 三件套（精确锁版本 + --no-deps）
uv pip install --python $PY --no-deps --index-url $IDX `
  "torch==2.10.0+rocm7.13.0a20260511" `
  "torchvision==0.25.0+rocm7.13.0a20260511" `
  "torchaudio==2.10.0+rocm7.13.0a20260511"

# 3) ROCm SDK 运行时（缺了 import torch 会 ModuleNotFoundError: rocm_sdk）
uv pip install --python $PY --no-deps --index-url $IDX `
  "rocm-sdk-core==7.13.0a20260511" `
  "rocm-sdk-libraries-gfx120x-all==7.13.0a20260511" `
  "rocm==7.13.0a20260511"

# 4) 项目应用依赖
uv pip install --python $PY `
  "scikit-image==0.25.2" "lpips==0.1.4" "clean-fid==0.1.25" `
  "rich==14.0.0" "tqdm==4.67.1" "tensorboard==2.20.0" "setuptools>=70" `
  "einops==0.8.1" "scipy==1.16.2" "matplotlib==3.10.1" `
  "fonttools==4.57.0" "potracer==0.0.4"

# 5) 防止 uv run 把 ROCm torch 重装成 CUDA 版
setx UV_NO_SYNC "1"
```

验证：

```powershell
& $PY -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：7.13.26176 True AMD Radeon RX 9070 XT
```

之后即可用 `scripts\bat\*.bat` 直接跑训练与推理。

> [!TIP]
>
> - 首次训练会花约 1 分钟编译 MIOpen 卷积 kernel，属正常现象。
> - 数值不稳时设 `$env:HANZIGEN_AMP_DTYPE = "bf16"` 改用 bfloat16。
> - 训练后可用 `python scripts/inspect_checkpoint.py` 检查权重是否健康。
> - **不要手动跑 `uv sync`**：`uv.lock` 锁的是 CUDA 版 torch，会覆盖掉 ROCm 版。

```powershell
> # DirectML（可选，较简单但较慢）
> uv venv --python 3.12 .venv-amd
> uv pip install --python .venv-amd\Scripts\python.exe -r constraints/amd-directml.txt
 ```

⚠️ `uv run` 只绑定 `.venv`。想切换默认后端，就是把 `.venv` 建成对应后端。

#### 1-4 项目结构

```
HanziGen-for-AMD/
├── run.py                        # 后端感知启动器（解析 --backend，必要时自动安装对应 PyTorch）
├── check_hardware.py             # 硬件诊断与性能基准脚本
├── constraints/                  # 各后端的 torch 安装设置档（cuda / rocm / zluda / directml / cpu）
├── docs/
│   ├── AMD_SETUP.md              # AMD 显卡安装指南
│   ├── NVIDIA_SETUP.md           # NVIDIA 显卡安装指南
│   └── research-amd-pytorch-2026.md
├── utils/hardware/               # 后端检测与装置抽象层
│   ├── backends.py
│   └── hardware_utils.py
├── tests/                        # 不需 GPU 即可执行的测试
├── configs/                      # 模型、数据集与训练设置
├── datasets/                     # 数据集与 DataLoader
├── models/                       # VQ-VAE 与 LDM 模型
├── scripts/bat/                  # Windows 批次脚本
├── scripts/sh/                   # Linux / macOS 壳层脚本
├── scripts/inspect_checkpoint.py # VQ-VAE 检查点健康检查（权重／码本／重建三重验证）
├── assets/                       # README 示意图
├── charsets/                     # 字符集与覆盖率分析结果
└── fonts/                        # 目标字体与参考字体
```

> [!TIP]
>
> 所有训练与推理脚本都支持 `--device`。若改用环境变量，可设
> `HANZIGEN_BACKEND`（例如 `rocm`、`directml`），或直接使用启动器：
>
> ```bash
> uv run python run.py --backend rocm train_vqvae.py --num_epochs 100
> ```
>
> `scripts/bat/` 与 `scripts/sh/` 底下的脚本都会经过 `run.py`，因此
> **不需要**修改脚本内容就能换后端 —— 只要在执行前设置 `HANZIGEN_BACKEND` 即可：
>
> ```bash
> # Windows (PowerShell)
> $env:HANZIGEN_BACKEND = "rocm"; .\scripts\bat\train_vqvae.bat
>
> # Linux / macOS
> HANZIGEN_BACKEND=rocm bash scripts/sh/train_vqvae.sh
> ```
>
> 未设置时会自动检测，AMD 用户不需要手动把 `DEVICE` 改成 `cuda`。

---

### 2. 准备字体文件 🖋️

此步骤将准备项目所需的目标字体与参考字体文件。

#### 2-1 建立文件夹

- 在项目根目录建立 `fonts/` 文件夹

#### 2-2 放置目标字体

- 选择一款欲补全的目标字体（`.ttf` 或 `.otf` 格式）
- 将目标字体文件放入 `fonts/` 文件夹

#### 2-3 放置参考字体

- 前往 [Jigmo 官方网站](https://kamichikoichi.github.io/jigmo/) 下载 Jigmo 字体作为参考字体
- 在 `fonts/` 文件夹内建立 `jigmo/` 子文件夹
- 将下载的 `jigmo.ttf`、`jigmo2.ttf`、`jigmo3.ttf` 文件放入 `fonts/jigmo/` 文件夹

> [!CAUTION]
>
> - 请确认目标字体的授权条款允许修改与再发布
> - 请确保目标字体的档名不含空格以及特殊字符

> [!TIP]
>
> - 建议选用至少包含 2000 个汉字的目标字体，以获得更好的补全效果

<details>
<summary>📁 查看文件结构</summary>

```
fonts/
├── [target_font].[ttf_or_otf]
└── jigmo/
    ├── jigmo.ttf
    ├── jigmo2.ttf
    └── jigmo3.ttf
```

- `[target_font].[ttf_or_otf]`：欲补全的目标字体文件。
- `jigmo/`：存放参考字体的文件夹。
  - `jigmo.ttf`、`jigmo2.ttf`、`jigmo3.ttf`：Jigmo 系列参考字体文件。

</details>

---

### 3. 分析字体汉字覆盖率 📊

此步骤将分析目标字体与参考字体在 [jf7000](https://justfont.com/jf7000) 与 [Unihan](https://www.unicode.org/charts/unihan.html) 字符集中的汉字覆盖情况，并产生对应的覆盖与缺失字符集。

#### 3-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/analyze_font.sh`](./scripts/sh/analyze_font.sh)
- **Windows 原生环境**：开启 [`scripts/bat/analyze_font.bat`](./scripts/bat/analyze_font.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `REFERENCE_FONTS_DIR`：参考字体文件夹路径（字串）

</details>

#### 3-2 执行脚本

> [!IMPORTANT]
>
> - 请根据您的作业系统选择对应的指令，并于终端机执行

- **Linux/WSL2 环境**

```bash
bash scripts/sh/analyze_font.sh
```

- **Windows 原生环境**

```bat
scripts\bat\analyze_font.bat
```

#### 3-3 查看输出

- 覆盖率统计：目标字体与参考字体对 `jf7000` 和 `Unihan` 各子字符集的覆盖率将显示于终端机。
- 文件输出：覆盖（`covered.txt`）与缺失（`missing.txt`）字符集文件将存储在 `charsets/` 文件夹中。

<details>
<summary>📊 查看覆盖率统计</summary>

```
                        Jigmo jf7000 Coverage Statistics
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Charset           ┃ Total ┃ Covered ┃ Missing ┃ Covered Ratio ┃ Missing Ratio ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ basic.txt         │  6373 │    6373 │       0 │       100.00% │         0.00% │
│ ext_cantonese.txt │   127 │     117 │      10 │        92.13% │         7.87% │
│ ext_japan.txt     │   534 │     533 │       1 │        99.81% │         0.19% │
│ ext_naming.txt    │   618 │     611 │       7 │        98.87% │         1.13% │
│ ext_taiwan.txt    │   810 │     700 │     110 │        86.42% │        13.58% │
│ jf7000_all.txt    │  8349 │    8221 │     128 │        98.47% │         1.53% │
└───────────────────┴───────┴─────────┴─────────┴───────────────┴───────────────┘
```

- `basic.txt`：jf7000 汉字基本包。
- `ext_cantonese.txt`：jf7000 港澳扩充包。
- `ext_japan.txt`：jf7000 日文扩充包。
- `ext_naming.txt`：jf7000 台湾命名扩充包。
- `ext_taiwan.txt`：jf7000 本土语言扩充包。
- `jf7000_all.txt`：上述子字符集的集合字符集。

---

```
                        Jigmo unihan Coverage Statistics
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Charset         ┃  Total ┃ Covered ┃ Missing ┃ Covered Ratio ┃ Missing Ratio ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ basic.txt       │  20992 │   20992 │       0 │       100.00% │         0.00% │
│ compat.txt      │    472 │     472 │       0 │       100.00% │         0.00% │
│ compat_supp.txt │    542 │       0 │     542 │         0.00% │       100.00% │
│ ext_a.txt       │   6592 │    6592 │       0 │       100.00% │         0.00% │
│ ext_b.txt       │  42720 │       1 │   42719 │         0.00% │       100.00% │
│ ext_c.txt       │   4160 │       0 │    4160 │         0.00% │       100.00% │
│ ext_d.txt       │    222 │       0 │     222 │         0.00% │       100.00% │
│ ext_e.txt       │   5774 │       0 │    5774 │         0.00% │       100.00% │
│ ext_f.txt       │   7473 │       0 │    7473 │         0.00% │       100.00% │
│ ext_g.txt       │   4939 │       0 │    4939 │         0.00% │       100.00% │
│ ext_h.txt       │   4192 │       0 │    4192 │         0.00% │       100.00% │
│ ext_i.txt       │    622 │       0 │     622 │         0.00% │       100.00% │
│ ext_j.txt       │   4298 │       0 │    4298 │         0.00% │       100.00% │
│ unihan_all.txt  │ 102998 │   28057 │   74941 │        27.24% │        72.76% │
└─────────────────┴────────┴─────────┴─────────┴───────────────┴───────────────┘
```

- `basic.txt`：中日韩统一表意文字字符集。
- `compat.txt`：中日韩兼容表意文字字符集。
- `compat_supp.txt`：中日韩兼容表意文字补充字符集。
- `ext_a.txt`～`ext_j.txt`：中日韩统一表意文字扩充字符集 A～J 区。
- `unihan_all.txt`：上述子字符集的集合字符集。

</details>

<details>
<summary>📁 查看文件结构</summary>

```
charsets/
├── jf7000_coverage/
│   ├── [target_font]/
│   │   ├── covered.txt
│   │   └── missing.txt
│   └── ...
├── unihan_coverage/
│   ├── [target_font]/
│   │   ├── covered.txt
│   │   └── missing.txt
│   └── ...
└── ...
```

- `jf7000_coverage/`：存储字体对于 `jf7000` 字符集覆盖结果的文件夹。
  - `[target_font]/`：以目标字体命名的文件夹。
    - `covered.txt`：目标字体覆盖字符集文件。
    - `missing.txt`：目标字体缺失字符集文件。
- `unihan_coverage/`：存储字体对于 `Unihan` 字符集覆盖结果的文件夹。
  - `[target_font]/`：以目标字体命名的文件夹。
    - `covered.txt`：目标字体覆盖字符集文件。
    - `missing.txt`：目标字体缺失字符集文件。

</details>

---

### 4. 准备字形图像数据集 🖨️

此步骤将产生目标字体与参考字体的字形图像，作为后续模型训练所需的数据集。

#### 4-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/prepare_dataset.sh`](./scripts/sh/prepare_dataset.sh)
- **Windows 原生环境**：开启 [`scripts/bat/prepare_dataset.bat`](./scripts/bat/prepare_dataset.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `REFERENCE_FONTS_DIR`：参考字体文件夹路径（字串）
- `IMG_WIDTH`：字形图像宽度（整数）
- `IMG_HEIGHT`：字形图像高度（整数）
- `SAMPLE_RATIO`：抽样比例（浮点数，范围 0.0～1.0）

</details>

> [!WARNING]
>
> - 请确保 `IMG_WIDTH` 与 `IMG_HEIGHT` 参数设置值相同

> [!TIP]
>
> - 可调整 `SAMPLE_RATIO` 参数，以实验不同的字形图像数据集大小

#### 4-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/prepare_dataset.sh
```

- **Windows 原生环境**

```bat
scripts\bat\prepare_dataset.bat
```

#### 4-3 查看输出

- 文件输出：目标字形（`target`）与参考字形（`reference`）图像将存储在 `data/` 文件夹中。

<details>
<summary>📁 查看文件结构</summary>

```
data/
├── reference/
│   ├── 04E00.png
│   ├── 04E8C.png
│   ├── 04E09.png
│   └── ...
└── target/
    ├── 04E00.png
    ├── 04E8C.png
    ├── 04E09.png
    └── ...
```

- `reference/`：存放参考字形图像的文件夹。
  - `04E00.png`：参考字形图像，档名为 Unicode 码位。
- `target/`：存放目标字形图像的文件夹。
  - `04E00.png`：目标字形图像，档名为 Unicode 码位。

</details>

---

### 5. 划分训练／验证字符集 🎯

此步骤将划分训练字符集与验证字符集。

#### 5-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/split_dataset.sh`](./scripts/sh/split_dataset.sh)
- **Windows 原生环境**：开启 [`scripts/bat/split_dataset.bat`](./scripts/bat/split_dataset.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `TRAIN_SPLIT_RATIO`：训练字符集比例（浮点数，范围 0.0～1.0）
- `VAL_SPLIT_RATIO`：验证字符集比例（浮点数，范围 0.0～1.0）
- `SPLIT_RANDOM_SEED`：划分随机种子（整数）
- `DEVICE`：指定运算设备（字串，可选 `"auto"`、`"cuda"`、`"rocm"`、`"zluda"`、`"directml"`、`"mps"`、`"cpu"`）

</details>

> [!WARNING]
>
> - 请确保 `TRAIN_SPLIT_RATIO` 与 `VAL_SPLIT_RATIO` 之和等于 1.0

> [!TIP]
>
> - 可调整 `TRAIN_SPLIT_RATIO`、`VAL_SPLIT_RATIO` 与 `SPLIT_RANDOM_SEED` 参数，以实验不同的字符集划分
> - 可调整 `DEVICE` 参数，以指定运算设备（例如：`"cuda"`、`"rocm"`、`"directml"`、`"cpu"`；省略时会自动检测）。AMD 显卡请参考 [AMD 显卡安装指南](./docs/AMD_SETUP.md)

#### 5-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/split_dataset.sh
```

- **Windows 原生环境**

```bat
scripts\bat\split_dataset.bat
```

#### 5-3 查看输出

- 文件输出：划分好的训练字符集（`train.txt`）与验证字符集（`val.txt`）文件将存储在 `charsets/` 文件夹中。

<details>
<summary>📁 查看文件结构</summary>

```
charsets/
├── splits/
│   ├── [target_font]/
│   │   ├── train.txt
│   │   └── val.txt
│   └── ...
└── ...
```

- `splits/`：存储字符集划分结果的文件夹。
  - `[target_font]/`：以目标字体命名的文件夹。
    - `train.txt`：训练字符集文件。
    - `val.txt`：验证字符集文件。

</details>

---

### 6. 训练 VQ-VAE 🏋️

此步骤将训练 VQ-VAE 模型，作为字形图像的编码与解码模块。

#### 6-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/train_vqvae.sh`](./scripts/sh/train_vqvae.sh)
- **Windows 原生环境**：开启 [`scripts/bat/train_vqvae.bat`](./scripts/bat/train_vqvae.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `TRAIN_SPLIT_RATIO`：训练字符集比例（浮点数，范围 0.0～1.0）
- `VAL_SPLIT_RATIO`：验证字符集比例（浮点数，范围 0.0～1.0）
- `SPLIT_RANDOM_SEED`：划分随机种子（整数）
- `BATCH_SIZE`：训练批次大小（整数）
- `LEARNING_RATE`：学习率（浮点数）
- `NUM_EPOCHS`：训练周期数（整数）
- `IMG_SAVE_INTERVAL`：生成样本存储周期间隔（整数）
- `DEVICE`：指定运算设备（字串，可选 `"auto"`、`"cuda"`、`"rocm"`、`"zluda"`、`"directml"`、`"mps"`、`"cpu"`）
- `RESUME`：是否从检查点接续训练（布林值：`true`/`false`）
- `USE_AMP`：是否使用混合精度训练（布林值：`true`/`false`）

</details>

> [!WARNING]
>
> - 请确保 `TRAIN_SPLIT_RATIO` 、 `VAL_SPLIT_RATIO` 与 `SPLIT_RANDOM_SEED` 参数与步骤 5 一致

> [!TIP]
>
> - 可调整 `BATCH_SIZE`、`LEARNING_RATE` 与 `NUM_EPOCHS` 参数，以实验不同的训练设置
> - 可调整 `DEVICE` 参数，以指定运算设备（例如：`"cuda"`、`"rocm"`、`"directml"`、`"cpu"`；省略时会自动检测）。AMD 显卡请参考 [AMD 显卡安装指南](./docs/AMD_SETUP.md)
> - `USE_AMP` 在 CUDA 与 ROCm / ZLUDA 后端可设为 `true`；DirectML 不支持混合精度，程序会自动退回 float32 并显示提示
> - 混合精度默认使用 fp16；若在 AMD 上遇到数值不稳，可设 `HANZIGEN_AMP_DTYPE=bf16` 改用 bfloat16（指数范围与 float32 相同、无需梯度缩放，于 RDNA3/RDNA4 更稳定）
> - 若要接续先前中断的训练，可将 `RESUME` 参数设为 `true`

> [!NOTE]
>
> - 如需进一步自定义 VQ-VAE 架构，请参考 [`configs/vqvae_config.py`](./configs/vqvae_config.py) 中的 `VQVAEModelConfig` 类别
> - 训练完成后，可用 `uv run --no-sync python scripts/inspect_checkpoint.py` 检查检查点是否已坍缩／损坏，避免误用跑崩的权重

#### 6-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/train_vqvae.sh
```

- **Windows 原生环境**

```bat
scripts\bat\train_vqvae.bat
```

> [!TIP]
>
> - 训练过程中，可通过 `tensorboard --logdir=runs/VQVAE_[target_font]` 查看训练状态

#### 6-3 查看输出

- 训练状态：VQ-VAE 各周期的训练状态与损失将显示于终端机。
- 训练记录：TensorBoard 训练纪录将存储在 `runs/` 文件夹中。
- 模型权重：VQ-VAE 模型权重将存储在 `checkpoints/` 文件夹中。
- 生成样本：训练与验证样本图像将存储在 `samples_[target_font]/` 文件夹中。

<details>
<summary>📈 查看训练状态</summary>

```
✅ Best model saved (val loss: 0.002913)
          Epoch Status [20/100]
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric        ┃ Train Loss ┃ Val Loss ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total         │   0.003001 │ 0.002913 │
│ Recon         │   0.001008 │ 0.000985 │
│ Vq            │   0.001993 │ 0.001928 │
│ Learning Rate │   0.000914 │        - │
└───────────────┴────────────┴──────────┘
```

- `✅ Best model saved`：当验证集总损失（Val Total Loss）优于历史最佳时，系统将自动存储模型权重。
- `Epoch Status`：显示目前训练进度（当前周期 / 总周期）。
- `Train Loss` / `Val Loss`：训练集与验证集的损失值。
- `Total`：总损失（Total Loss）。
- `Recon`：重建损失（Reconstruction Loss）。
- `Vq`：向量量化损失（Vector Quantization Loss）。
- `Learning Rate`：当前学习率。

</details>

<details>
<summary>📁 查看文件结构</summary>

```
runs/
├── VQVAE_[target_font]/
│   └── [YYYYMMDD_HHMMSS]/
│       └── events.out.tfevents.xxxxx
└── ...
```

- `VQVAE_[target_font]/`：以目标字体命名的文件夹。
  - `YYYYMMDD_HHMMSS/`：以训练启动的时间戳记命名的文件夹。
    - `events.out.tfevents.xxxxx`：TensorBoard 训练纪录文件。

---

```
checkpoints/
├── vqvae_[target_font].pth
└── ...
```

- `vqvae_[target_font].pth`：VQ-VAE 模型权重文件。

---

```
samples_[target_font]/
├── vqvae_training_[YYYYMMDD_HHMMSS]/
│   ├── train/
│   │   ├── epoch_0000_ref_04E00.png
│   │   ├── epoch_0000_tgt_04E8C.png
│   │   └── ...
│   └── val/
│       ├── epoch_0000_ref_04E09.png
│       ├── epoch_0000_tgt_056DB.png
│       └── ...
└── ...
```

- `vqvae_training_[YYYYMMDD_HHMMSS]/`：以训练启动的时间戳记命名的文件夹。
  - `train/`：训练样本图像文件夹。
    - `epoch_0000_ref_04E00.png`：以当前周期与 Unicode 码位命名的字形图像文件。图像分为左右两个部分，分别为参考字形与重建字形。
    - `epoch_0000_tgt_04E8C.png`：以当前周期与 Unicode 码位命名的字形图像文件。图像分为左右两个部分，分别为目标字形与重建字形。
  - `val/`：验证样本图像文件夹。
    - `epoch_0000_ref_04E09.png`：以当前周期与 Unicode 码位命名的字形图像文件。图像分为左右两个部分，分别为参考字形与重建字形。
    - `epoch_0000_tgt_056DB.png`：以当前周期与 Unicode 码位命名的字形图像文件。图像分为左右两个部分，分别为目标字形与重建字形。

</details>

---

### 7. 训练 LDM 🏋️

此步骤将训练 LDM，作为最终的字形生成模型。

#### 7-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/train_ldm.sh`](./scripts/sh/train_ldm.sh)
- **Windows 原生环境**：开启 [`scripts/bat/train_ldm.bat`](./scripts/bat/train_ldm.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `TRAIN_SPLIT_RATIO`：训练字符集比例（浮点数，范围 0.0～1.0）
- `VAL_SPLIT_RATIO`：验证字符集比例（浮点数，范围 0.0～1.0）
- `SPLIT_RANDOM_SEED`：划分随机种子（整数）
- `BATCH_SIZE`：训练批次大小（整数）
- `LEARNING_RATE`：学习率（浮点数）
- `NUM_EPOCHS`：训练周期数（整数）
- `SAMPLE_STEPS`：生成样本去噪步骤数（整数）
- `IMG_SAVE_INTERVAL`：生成样本存储周期间隔（整数）
- `LPIPS_EVAL_INTERVAL`：LPIPS 评估周期间隔（整数）
- `EVAL_BATCH_SIZE`：评估批次大小（整数）
- `DEVICE`：指定运算设备（字串，可选 `"auto"`、`"cuda"`、`"rocm"`、`"zluda"`、`"directml"`、`"mps"`、`"cpu"`）
- `RESUME`：是否从检查点接续训练（布林值：`true`/`false`）
- `USE_AMP`：是否使用混合精度训练（布林值：`true`/`false`）

</details>

> [!WARNING]
>
> - 请确保 `TRAIN_SPLIT_RATIO`、`VAL_SPLIT_RATIO` 与 `SPLIT_RANDOM_SEED` 参数与步骤 5 一致

> [!TIP]
>
> - 可调整 `BATCH_SIZE`、`LEARNING_RATE` 与 `NUM_EPOCHS` 参数，以实验不同的训练设置
> - 可调整 `DEVICE` 参数，以指定运算设备（例如：`"cuda"`、`"rocm"`、`"directml"`、`"cpu"`；省略时会自动检测）。AMD 显卡请参考 [AMD 显卡安装指南](./docs/AMD_SETUP.md)
> - `USE_AMP` 在 CUDA 与 ROCm / ZLUDA 后端可设为 `true`；DirectML 不支持混合精度，程序会自动退回 float32 并显示提示
> - 若要接续先前中断的训练，可将 `RESUME` 参数设为 `true`

> [!NOTE]
>
> - 如需进一步自定义 LDM 架构，请参考 [`configs/ldm_config.py`](./configs/ldm_config.py) 中的 `LDMModelConfig` 类别

#### 7-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/train_ldm.sh
```

- **Windows 原生环境**

```bat
scripts\bat\train_ldm.bat
```

> [!TIP]
>
> - 训练过程中，可通过 `tensorboard --logdir=runs/LDM_[target_font]` 查看训练状态

#### 7-3 查看输出

- 训练状态：LDM 各周期的训练状态与损失将显示于终端机。
- 训练记录：TensorBoard 训练纪录将存储在 `runs/` 文件夹中。
- 模型权重：LDM 模型权重将存储在 `checkpoints/` 文件夹中。
- 生成样本：训练与验证样本图像将存储在 `samples_[target_font]/` 文件夹中。

<details>
<summary>📈 查看训练状态</summary>

```
✅ Best model saved. (LPIPS score: 0.181055)
          Epoch Status [20/250]
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric        ┃ Train Loss ┃ Val Loss ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total         │   0.016379 │ 0.015627 │
│ Learning Rate │   0.000491 │        - │
└───────────────┴────────────┴──────────┘
```

- `✅ Best model saved`：验证集 LPIPS 分数（LPIPS Score）优于历史最佳时，系统将自动存储模型权重。
- `Epoch Status`：显示目前训练进度（当前周期 / 总周期）。
- `Train Loss` / `Val Loss`：训练集与验证集的损失值。
- `Total`：总损失（Total Loss）。
- `Learning Rate`：当前学习率。

</details>

<details>
<summary>📁 查看文件结构</summary>

```
runs/
├── LDM_[target_font]/
│   └── [YYYYMMDD_HHMMSS]/
│       └── events.out.tfevents.xxxxx
└── ...
```

- `LDM_[target_font]/`：以目标字体命名的文件夹。
  - `YYYYMMDD_HHMMSS/`：以训练启动的时间戳记命名的文件夹。
    - `events.out.tfevents.xxxxx`：TensorBoard 训练纪录文件。

---

```
checkpoints/
├── ldm_[target_font].pth
└── ...
```

- `ldm_[target_font].pth`：LDM 模型权重文件。

---

```
samples_[target_font]/
├── ldm_training_[YYYYMMDD_HHMMSS]/
│   ├── train/
│   │   ├── epoch_0000_04E00.png
│   │   └── ...
│   └── val/
│       ├── epoch_0000_04E8C.png
│       └── ...
└── ...
```

- `ldm_training_[YYYYMMDD_HHMMSS]/`：以训练启动的时间戳记命名的文件夹。
  - `train/`：训练样本图像文件夹。
    - `epoch_0000_04E00.png`：以当前周期与 Unicode 码位命名的字形图像文件。图像分为左中右三个部分，分别为参考字形、目标字形与生成字形。
  - `val/`：验证样本图像文件夹。
    - `epoch_0000_04E8C.png`：以当前周期与 Unicode 码位命名的字形图像文件。图像分为左中右三个部分，分别为参考字形、目标字形与生成字形。

</details>

---

### 8. 计算评估指标 📈

此步骤将计算 PSNR、SSIM、LPIPS 与 FID 等指标，以评估 LDM 生成之验证集字形图像的品质。

#### 8-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/compute_metrics.sh`](./scripts/sh/compute_metrics.sh)
- **Windows 原生环境**：开启 [`scripts/bat/compute_metrics.bat`](./scripts/bat/compute_metrics.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `TIMESTAMP`：指定时间戳或使用自动检测（字串，可设为 `"auto"` 自动检测最新训练结果）
- `EVAL_BATCH_SIZE`：评估批次大小（整数）
- `DEVICE`：指定运算设备（字串，可选 `"auto"`、`"cuda"`、`"rocm"`、`"zluda"`、`"directml"`、`"mps"`、`"cpu"`）

</details>

> [!TIP]
>
> - 可调整 `DEVICE` 参数，以指定运算设备（例如：`"cuda"`、`"rocm"`、`"directml"`、`"cpu"`；省略时会自动检测）。AMD 显卡请参考 [AMD 显卡安装指南](./docs/AMD_SETUP.md)

#### 8-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/compute_metrics.sh
```

- **Windows 原生环境**

```bat
scripts\bat\compute_metrics.bat
```

#### 8-3 查看输出

- 评估指标：PSNR、SSIM、LPIPS 与 FID 之计算结果将显示于终端机。

<details>
<summary>📈 查看评估指标</summary>

```
  Metrics Results
┏━━━━━━━━┳━━━━━━━━━┓
┃ Metric ┃  Score  ┃
┡━━━━━━━━╇━━━━━━━━━┩
│ PSNR   │ 14.4034 │
│ SSIM   │  0.8998 │
│ LPIPS  │  0.0741 │
│ FID    │  7.1209 │
└────────┴─────────┘
```

- `PSNR`：峰值信噪比（Peak Signal-to-Noise Ratio），越高越好。
- `SSIM`：结构相似度指标（Structural Similarity Index Measure），越高越好。
- `LPIPS`：感知相似度（Learned Perceptual Image Patch Similarity），越低越好。
- `FID`：Fréchet Inception 距离（Fréchet Inception Distance），越低越好。

</details>

---

### 9. 生成字形图像 🖼️

此步骤将利用训练完成的 LDM 模型生成目标字形图像。

#### 9-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/inference.sh`](./scripts/sh/inference.sh)
- **Windows 原生环境**：开启 [`scripts/bat/inference.bat`](./scripts/bat/inference.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `REFERENCE_FONTS_DIR`：参考字体文件夹路径（字串）
- `CHARSET_PATH`：生成字符集文件路径（字串，可设为 `"auto"` 自动使用缺失字符集）
- `BATCH_SIZE`：生成批次大小（整数）
- `SAMPLE_STEPS`：生成样本去噪步骤数（整数）
- `IMG_WIDTH`：生成图像宽度（整数）
- `IMG_HEIGHT`：生成图像高度（整数）
- `DEVICE`：指定运算设备（字串，可选 `"auto"`、`"cuda"`、`"rocm"`、`"zluda"`、`"directml"`、`"mps"`、`"cpu"`）

</details>

> [!WARNING]
>
> - 请确保 `IMG_WIDTH` 与 `IMG_HEIGHT` 参数与步骤 4 一致

> [!TIP]
>
> - 可调整 `CHARSET_PATH` 参数，以指定要生成的字符集。默认为目标字体对 `jf7000` 的缺失字形
> - 若要生成对 `Unihan` 的缺失字形，请将 `CHARSET_PATH` 设为 `charsets/unihan_coverage/[target_font]/missing.txt`
> - 可调整 `DEVICE` 参数，以指定运算设备（例如：`"cuda"`、`"rocm"`、`"directml"`、`"cpu"`；省略时会自动检测）。AMD 显卡请参考 [AMD 显卡安装指南](./docs/AMD_SETUP.md)

> [!NOTE]
>
> - 可使用 `charsets/test/` 目录中的范例字符集
> - 如需自定义字符集，请使用每行一个汉字的纯文字文件
> - 由于生成具有随机性，建议可重复执行以取得更理想的结果

<details>
<summary>📁 查看范例字符集</summary>

```
charsets/
├── test/
│   ├── ordinals.txt
│   ├── periodic_table.txt
│   ├── qianziwen.txt
│   └── repeats.txt
└── ...
```

- `test/`：范例字符集文件夹。
  - `ordinals.txt`：序数字符集文件（中文数字、天干地支、十二生肖...）。
  - `periodic_table.txt`：元素周期表字符集文件（氢、氦、锂、铍...）。
  - `qianziwen.txt`：千字文字符集文件（天、地、玄、黄...）。
  - `repeats.txt`：叠字字符集文件（一、二、三、亖...）。

</details>

#### 9-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/inference.sh
```

- **Windows 原生环境**

```bat
scripts\bat\inference.bat
```

#### 9-3 查看输出

- 缺字字形：生成的字形图像将存储在 `samples_[target_font]/ldm_inference_[YYYYMMDD_HHMMSS]/infer/gen/` 文件夹中。

<details>
<summary>📁 查看文件结构</summary>

```
samples_[target_font]/
├── ldm_inference_[YYYYMMDD_HHMMSS]/
│   └── infer/
│       ├── gen/
│       │   ├── 04E00.png
│       │   └── ...
│       ├── gt/
│       │   ├── 04E00.png
│       │   └── ...
│       └── ref/
│           ├── 04E00.png
│           └── ...
└── ...
```

- `ldm_inference_[YYYYMMDD_HHMMSS]/`：以推理启动的时间戳记命名的文件夹。
  - `infer/`：推理结果文件夹。
    - `gen/`：生成字形图像文件夹。
      - `04E00.png`：生成字形图像，档名为 Unicode 码位。
    - `gt/`：真实字形图像文件夹。
      - `04E00.png`：真实字形图像，档名为 Unicode 码位，若目标字体无收录该字形则为空白图像。
    - `ref/`：参考字形图像文件夹。
      - `04E00.png`：参考字形图像，档名为 Unicode 码位。

</details>

---

### 10. 向量化字形图像 🔄

此步骤会将生成的位图字形图像转换为对应的 SVG 向量图形。

#### 10-1 设置参数

- **Linux/WSL2 环境**：开启 [`scripts/sh/convert_to_svg.sh`](./scripts/sh/convert_to_svg.sh)
- **Windows 原生环境**：开启 [`scripts/bat/convert_to_svg.bat`](./scripts/bat/convert_to_svg.bat)
- 设置 `TARGET_FONT_PATH` 为你的目标字体路径（例如：`"fonts/target_font.ttf"`）
- 调整其他参数（可选）

<details>
<summary>📋 查看脚本参数</summary>

- `TARGET_FONT_PATH`：目标字体文件路径（字串）
- `TIMESTAMP`：指定时间戳或使用自动检测（字串，可设为 `"auto"` 自动检测最新推理结果）
- `BLACKLEVEL`：黑色阈值（浮点数，范围 0.0～1.0）
- `TURDSIZE`：杂讯大小（整数）
- `ALPHAMAX`：最大 alpha 值（浮点数，范围 0.0～1.0）
- `OPTTOLERANCE`：优化容差（浮点数，范围 0.0～1.0）

</details>

> [!TIP]
>
> - 可调整 `BLACKLEVEL`、`TURDSIZE`、`ALPHAMAX` 与 `OPTTOLERANCE` 参数，以实验不同的向量化效果

#### 10-2 执行脚本

- **Linux/WSL2 环境**

```bash
bash scripts/sh/convert_to_svg.sh
```

- **Windows 原生环境**

```bat
scripts\bat\convert_to_svg.bat
```

#### 10-3 查看输出

- 向量字形：向量字形图像将存储在 `samples_[target_font]/ldm_inference_[YYYYMMDD_HHMMSS]/svg` 文件夹中。

<details>
<summary>📁 查看文件结构</summary>

```
samples_[target_font]/
├── ldm_inference_[YYYYMMDD_HHMMSS]/
│   └── svg/
│       ├── 04E00.svg
│       └── ...
└── ...
```

- `ldm_inference_[YYYYMMDD_HHMMSS]/`：以推理启动的时间戳记命名的文件夹。
  - `svg/`：向量字形图像文件夹。
    - `04E00.svg`：以 Unicode 码位命名的向量字形文件。

</details>

> [!NOTE]
>
> - 至此，您已完成从字体分析、模型训练、字形生成到向量化的完整项目流程。产出的向量字形文件（SVG）可导入如 FontForge 等字体编辑软件，进行后续的微调与字体封装作业
> - 若已熟悉上述各步骤，亦可使用一键式管线脚本（`bash scripts/sh/pipeline.sh` 或 `scripts\bat\pipeline.bat`）执行完整流程

---

## ⚠️ 模型限制

在处理笔画繁复、结构特殊的汉字时，模型生成的图像可能出现笔画断裂、变形或结构错误等情形（如下图所示），这反映了模型在极端样本上的泛化能力仍有提升空间。

![fail_cases](./assets/limitations.svg)

---

## 📚 参考资料

本项目开发期间，参考了多篇学术论文、网络文章及相关书目，特此汇整以供参考。

### 学术论文

- Ronneberger, O. et al. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), _arXiv:1505.04597_.
- van den Oord, A. et al. (2017). [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937), _arXiv:1711.00937_.
- Ho, J. et al. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), _arXiv:2006.11239_.
- Song, J. et al. (2020). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), _arXiv:2010.02502_.
- Rombach, R. et al. (2021). [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), _arXiv:2112.10752_.

### 网络文章

- 苏炜翔（2014）。[关于“南去经三国，东来过五湖”](https://blog.justfont.com/2014/12/jfbook-example/)。justfont Blog。
- 苏炜翔（2015）。[字体下载之前，先看看字体是怎么制作的](https://blog.justfont.com/2015/07/check-this-out-b4-downloading-fonts/)。justfont Blog。
- Ruby（2018）。[字体版权通识课：保护方式与产业现况](https://blog.justfont.com/2018/10/copyright-course-1/)。justfont Blog。
- 苏炜翔（2019）。[有种志业，叫做字体设计](https://blog.justfont.com/2019/01/typedesigners-talk/)。justfont Blog。
- 姜呈颖（2023）。[下载字体前先看｜为什么会缺字？](https://blog.justfont.com/2023/06/charactersets)。justfont Blog。

### 相关书目

- Graphic 社编辑部、卵形｜叶忠宜（统筹．设计）（2016–2020）。Typography 字志 系列（Issue 01–06）。脸谱出版。
- 柯志杰、苏炜翔（2019）。字体散步 Next：从台湾日常出发，无所不在的中文字体学。脸谱出版。

### 线上资源

- [justfont](https://justfont.com/)：台湾的字体设计与教育推广品牌。
- [Google Fonts](https://fonts.google.com/)：Google 提供的免费字体资源库。
- [BabelMap Online](https://www.babelstone.co.uk/Unicode/babelmap.html)：线上 Unicode 字符编码查询工具。
- [Unicode 区段](https://zh.wikipedia.org/wiki/Unicode区段)：维基百科上关于 Unicode 区段的介绍。

---

## 🙏 特别感谢

本项目得以完成，仰赖众多开源社群、创作者及社群伙伴的无私贡献，特此致谢。

### 字体资源

- [Jigmo](https://kamichikoichi.github.io/jigmo/)：参考字体。
- [851ゴチカクット](https://pm85122.onamae.jp/851Gkktt.html)：目标字体范例一。
- [851テガキカクット](https://pm85122.onamae.jp/851H_kktt.html)：目标字体范例二。
- [851チカラヅヨク](https://pm85122.onamae.jp/851ch-dz.html)：目标字体范例三。

### 字符集资源

- [jf7000 当务字符集](https://justfont.com/jf7000)：jf7000 字符集。
- [Unihan 字符集](https://www.unicode.org/charts/unihan.html)：Unihan 字符集。

### 开发工具

- [FontForge](https://fontforge.org/)：开源字体编辑软件。

---

## 📖 项目引用

若本项目对您的研究有所帮助，欢迎引用此论文：

```bibtex
@inproceedings{wang2026hanzigen,
  author    = {Wang, Wen-Ho and Su, Po-Chyi},
  booktitle = {2026 IEEE International Conference on Consumer Electronics - Taiwan (ICCE-Taiwan)},
  title     = {HanziGen: Supervised High-Fidelity Chinese Font Generation via Latent Diffusion Models},
  year      = {2026},
  pages     = {915--916},
  month     = jul,
  publisher = {IEEE},
  doi       = {10.1109/ICCE-Taiwan71481.2026.11652402}
}
```

## 📜 授权条款

本项目采用 Apache License 2.0 授权条款，详细内容请参阅 [授权条款](./LICENSE)。

> [!NOTE]
>
> - 本项目所引用的部分资源（如 jf7000 字符集）可能受其原始授权条款（如 CC BY-SA 4.0）约束，使用时请遵循其原始规范
