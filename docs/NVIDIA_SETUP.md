# NVIDIA 顯示卡安裝指南

本文件說明如何在 **NVIDIA 顯示卡** 上執行 HanziGen。這是本專案原本的目標環境，
安裝流程與 v1.3.0 相同，僅在敘述上對齊 v1.4.0 的多後端架構。

> [!TIP]
>
> 使用 AMD 顯示卡請改看 [AMD 顯示卡安裝指南](AMD_SETUP.md)。

---

## 1. 硬體與驅動需求

| 項目 | 需求 |
| --- | --- |
| 顯示卡 | 至少 **4 GB VRAM** 的 NVIDIA 顯示卡 |
| 驅動程式 | 相容於 CUDA 11.8 或以上版本 |
| 作業系統 | Linux、Windows（原生或 WSL2） |
| Python | 3.12 以上 |

PyTorch 的 CUDA 支援由驅動向下相容：只要驅動夠新，就能執行以較舊 CUDA
版本編譯的 PyTorch。若你的驅動較舊，請改成對應的 CUDA 索引（見第 4 節）。

---

## 2. 建立環境

```bash
git clone https://github.com/wangwenho/HanziGen.git
cd HanziGen
uv sync
```

`uv sync` 會依 [`pyproject.toml`](../pyproject.toml) 的設定，安裝來自
`https://download.pytorch.org/whl/cu118` 的 PyTorch（CUDA 11.8 版本），
這與 v1.3.0 的行為一致。

---

## 3. 驗證安裝

```bash
uv run python check_hardware.py --benchmark
```

預期輸出重點：

- `Backend : CUDA (cuda)`、`Vendor : NVIDIA`
- `Build type : CUDA build (NVIDIA, or ZLUDA if an AMD GPU is present)`
- 所有功能檢查皆為 `PASS`
- 效能基準落在 `roughly 20-60 ms` 區間（RTX 30/40 系列）

若顯示為 `CPU`，代表裝到的是 CPU 版 PyTorch，請執行第 4 節的指令重新安裝。

---

## 4. 改用其他 CUDA 版本

編輯 [`pyproject.toml`](../pyproject.toml) 的 `[tool.uv.sources]` 與
`[[tool.uv.index]]`，把 `cu118` 換成你要的版本（例如 `cu121`、`cu124`、`cu126`），
然後重新同步：

```bash
uv sync
```

或者不動 `pyproject.toml`，改用 constraints 檔安裝：

```bash
uv venv --python 3.12
uv sync --no-install-package torch --no-install-package torchvision --no-install-package torchaudio
uv pip install -r constraints/nvidia-cuda.txt
```

---

## 5. 開始訓練

```bash
uv run python train_vqvae.py --device cuda --use_amp true
uv run python train_ldm.py   --device cuda --use_amp true
```

`--device` 省略時會自動偵測，因此在 NVIDIA 機器上不指定也可以。

### 混合精度

`USE_AMP` 對應的 `torch.amp.autocast` 與 `torch.amp.GradScaler` 在 CUDA 上完整支援。
RTX 20 系列（含）以上的顯示卡建議設為 `true`。

v1.4.0 起，CUDA 後端還會自動啟用 TF32：

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

這對 Ampere 以上的顯示卡能帶來免費的加速。若你需要嚴格的數值可重現性，
可在自己的訓練腳本中於 `enable_fast_math()` 之後把它們改回 `False`。

### 多張顯示卡

`--device cuda:1` 可以指定第二張顯示卡。本專案目前**不支援**多 GPU 資料平行
（`DataParallel` / `DistributedDataParallel`），請以單張顯示卡執行。

---

## 6. 疑難排解

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| 顯示 `Backend : CPU` | 裝到 CPU 版 PyTorch | 重新執行 `uv sync`，或改用第 4 節的 constraints 檔 |
| `CUDA out of memory` | 批次過大 | 降低 `--batch_size`，LDM 另可降低 `--eval_batch_size` |
| `torch.cuda.is_available()` 為 `False` | 驅動過舊或未安裝 | 更新 NVIDIA 驅動 |
| 訓練速度遠低於預期 | 資料載入瓶頸 | 提高 `num_workers`；確認資料集放在 SSD |
| 出現 `ZLUDA detected` 提示 | 系統裝了 ZLUDA DLL | 這是 AMD 路徑的正常行為，請參考 [AMD 指南](AMD_SETUP.md) |

---

## 延伸閱讀

- [AMD 顯示卡安裝指南](AMD_SETUP.md)
- [專案 README](../README.md)
