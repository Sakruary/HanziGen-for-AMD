# 更新日志

## v1.5.0

- 2026 年 10 月 15 日

### ✨ 合并版本（HanziGen-for-AMD）

- 以 v1.4.0（AMD 后端抽象层）为基底，合并另一条 AMD 适配线路的两项优点，成为单一、干净、开箱即用的版本。

### ✨ 新增功能

- **bf16 混合精度**：新增 `resolve_amp_dtype()`／`supports_bfloat16()`／`amp_requires_grad_scaler()`，以 `HANZIGEN_AMP_DTYPE=bf16` 切换至 bfloat16。bf16 指数范围与 float32 相同、无需梯度缩放，于 AMD RDNA3/RDNA4（gfx1100/gfx1200）更稳定。
- **训练健康监测**：VQ-VAE 新增 `_check_training_health()`，以“重建 MSE vs 灰底基线”与“码本使用占比”两信号检测码本坍缩，连续两次未通过即抛出 `TrainingDivergedError` 停机，且只在健康时存储最佳检查点。
- 新增检查点健康检查工具 [`scripts/inspect_checkpoint.py`](./scripts/inspect_checkpoint.py)：对已保存权重做“权重健全性／码本使用／重建质量”三重验证。

### 🔧 改进项目

- 分离 autocast 启用与梯度缩放两个标志（`use_amp` = AMP 已请求、`use_scaler` = 实际缩放），使 bf16 路径能“开 autocast、关 scaler”；修正原先以 `scaler_is_enabled` 推导 autocast 会在 bf16 下误关 autocast 的问题。
- 训练启动的后端摘要现在会显示实际使用的混合精度 dtype（`fp16` / `bf16`）。
- 新增 `setuptools>=70` 依赖，修复 uv 虚拟环境下 `tensorboard` 命令因缺少 `pkg_resources` 而失败的问题。
- `.gitignore` 补上 `.pytest_cache/` 与 `pytest-cache-files-*/`。
- 更新 [`pyproject.toml`](./pyproject.toml) 版本号至 1.5.0。

### ✅ 验证

Windows 11 + Radeon RX 9070 XT（gfx1201）实机：后端检测、MIOpen 缓存、fp16 AMP +
GradScaler、bf16 AMP（no-op scaler）、LDM 训练步皆产出有限损失；测试套件 **73 passed / 3 skipped**。

## v1.4.0

- 2026 年 10 月 15 日

### ✨ 新增功能

- **支持 AMD 显卡**：新增硬件后端抽象层（[`utils/hardware/backends.py`](./utils/hardware/backends.py)），可自动检测 NVIDIA CUDA、AMD ROCm / HIP、AMD ZLUDA、Windows DirectML、Apple MPS 与纯 CPU，并依后端调整混合精度与数据移动策略。以 Radeon RX 9070 XT（RDNA4 / `gfx1201`）为主要验证目标。
- 新增硬件诊断与性能基准脚本 [`check_hardware.py`](./check_hardware.py)，会列出检测到的后端并实际执行本项目所需的各项算子（`conv2d`、`conv_transpose2d`、`group_norm`、codebook 最近邻查表、autograd 等）。
- 新增后端感知启动器 [`run.py`](./run.py)，统一解析 `--backend`／`HANZIGEN_BACKEND`／constraints 标记；指定 ROCm 而环境不符时会尝试自动安装对应的 PyTorch，并可用 `HANZIGEN_ROCM_WHEEL_INDEX`、`HANZIGEN_NO_PIP_INSTALL` 覆写行为。
- 新增四份安装配置文件：[`constraints/amd-rocm.txt`](./constraints/amd-rocm.txt)、[`constraints/amd-zluda.txt`](./constraints/amd-zluda.txt)、[`constraints/amd-directml.txt`](./constraints/amd-directml.txt)、[`constraints/cpu.txt`](./constraints/cpu.txt)。
- 新增安装文件：[AMD 显卡安装指南](./docs/AMD_SETUP.md) 与 [NVIDIA 显卡安装指南](./docs/NVIDIA_SETUP.md)。
- 新增测试：后端检测（[`tests/test_backend_selection.py`](./tests/test_backend_selection.py)）、启动器（[`tests/test_run_launcher.py`](./tests/test_run_launcher.py)）与端到端训练批次（[`tests/test_training_step.py`](./tests/test_training_step.py)），皆不需 GPU 即可执行。

### 🐛 问题修复（AMD / DirectML / ROCm 实机验证时发现）

- **混合精度其实从未生效**：模型以 `getattr(scaler, "enabled", False)` 推导是否启用 autocast，但 `torch.amp.GradScaler` **没有 `enabled` 属性**（只有 `is_enabled()`），因此即使用户要求 AMP、scaler 也正确建立，autocast 仍永远不会进入。新增 `scaler_is_enabled()` 并修正 VQ-VAE 与 LDM。
- **checkpoint 无法跨后端加载**：原本以 `torch.save(model.state_dict(), path)` 保存会写入设备张量，在 DirectML 上会 pickle 成 `_rebuild_device_tensor_from_numpy`，导致 `weights_only=True` 加载直接失败，使“先训 VQ-VAE 再训 LDM”的标准流程中断。新增 `save_state_dict()`／`load_state_dict()`，保存前先搬到 CPU、加载时一律载到 CPU 再由模型 `.to(device)`。
- **DirectML 下 `torch.load` 直接崩溃**：`map_location` 传入 `torch.device` 对象时，torch-directml 的 storage restore 只接受整数索引，会抛出 `TypeError`。上述 `load_state_dict()` 一并解决此问题。
- **设备名称含 NUL 字节**：DirectML 返回固定长度宽字符缓冲区，设备名尾端带 `\x00`，会污染日志输出与字符串比较。`backends.py` 现在会清理设备名称。
- **混合精度下 codebook 查表精度不足**：最近邻查表在 autocast 下会精度流失，甚至出现 `expected scalar type Half but found Float`。改以 `no_autocast` 强制 fp32 计算距离后再转回原 dtype。
- **ROCm 上 `cudnn.benchmark` 不该开启**：ROCm 的 `torch.backends.cudnn` 存在且 `is_available()` 回报 `True`（它是 MIOpen 的兼容层），因此单纯以 `is_available()` 判断无法排除 AMD。MIOpen 以 JIT 编译 kernel，首次卷积在 `benchmark` 模式下需数秒至数十秒；实测 512×512 首次卷积约 4 秒，小型卷积冷缓存时达 42 秒，而稳态吞吐完全相同（134 vs 135 张/秒）。现改为仅在 CUDA 后端设置。
- **Windows 旧式代码页中断输出**：GBK 环境下输出符号会抛 `UnicodeEncodeError` 并中止执行。标准输出改为 UTF-8 并容许替代字符。
- **`uv lock` 无法求解**：`torch-directml` 绑定 `torchvision==0.19.1`，与项目所需的 0.22.1 冲突。移除 `directml` optional extra，改以 constraints 文件安装于独立环境。
- **测试互相污染**：设备自动检测结果会被缓存，导致测试顺序影响结果；新增 `reset_device_cache()` 并在 `tests/conftest.py` 逐测试重置。
- **DirectML autograd 初始化顺序**：若行程中第一个 backward 是 CPU 运算，PyTorch 不会为 `privateuseone` 建立 autograd ready queue，之后第一个 GPU backward 会以 `device_ready_queues_` INTERNAL ASSERT 失败。`tests/conftest.py` 以 session 层级暖机 backward 修正。

### 🔧 改进项目

- `--device` 除既有的 `cuda` / `mps` / `cpu` 外，新增 `auto`、`rocm`、`hip`、`amd`、`zluda`、`directml`、`dml`，并支持 `HANZIGEN_BACKEND` 环境变量覆写。
- `USE_AMP` 改为后端感知：CUDA / ROCm / ZLUDA 正常启用；DirectML 因不支持 `torch.amp`，会自动退回 float32 并显示警告，而非直接崩溃。
- 合并 VQ-VAE 与 LDM 中 `scaler is None` 的重复分支，改用后端无关的 `autocast_context` 与 `make_grad_scaler`。
- `pin_memory` 改以 `is_accelerator` 判断，涵盖 DirectML（`privateuseone`）等其他加速器。
- 设备张量转为 PIL 图像前先移回主机内存并转为 `float32`。
- VQ-VAE 与 LDM 训练开始前会印出完整的后端摘要（后端、厂商、设备名称、架构、VRAM、混合精度状态）。
- 所有 `scripts/bat/` 与 `scripts/sh/` 包装脚本改为调用 `run.py`，不再硬编码 `DEVICE=cuda`；改后端只需设置 `HANZIGEN_BACKEND`，不必修改脚本。
- `requires-python` 由 `>=3.13` 放宽为 `>=3.12`（`torch-directml` 仅发布至 Python 3.12），并更新 [`.python-version`](./.python-version) 与 [`uv.lock`](./uv.lock)。
- `check_hardware.py` 的性能基准改为符合本项目实际负载（512 × 512、batch 8），并以实测值取代先前的粗略区间。
- 更新 [`pyproject.toml`](./pyproject.toml) 版本号至 1.4.0，新增 ROCm / CPU 索引设置。

### ✅ 验证

Windows 11 + Radeon RX 9070 XT（gfx1201），**ROCm 与 DirectML 两条路线都完整跑通**
VQ-VAE 训练、LDM 训练（含 PSNR / SSIM / LPIPS 指标）、DDIM 推理与 checkpoint 存取。

`check_hardware.py` 的 9 项算子检查全部 PASS。实测性能（512×512、batch 8 的
conv + groupnorm 训练步）：

| 后端 | 每步耗时 | 吞吐 | float16 autocast |
| --- | --- | --- | --- |
| ROCm / HIP（torch 2.10.0+rocm7.13.0a20260511） | 52 ms | 153 张/秒 | 支持 |
| DirectML（torch 2.4.1 + torch-directml） | 171 ms | 47 张/秒 | 不支持 |
| 纯 CPU（Ryzen 7000/9000） | 699 ms | 11 张/秒 | 不支持 |

测试套件：**ROCm 与 DirectML 环境皆为 73 passed / 3 skipped**（ROCm 首次执行需编译
MIOpen 卷积 kernel，冷缓存时约 76 秒，缓存建立后约 11 秒；DirectML 约 4 秒）。

本项目工作区的默认环境配置：`.venv` = ROCm（`uv run` 的默认后端），
`.venv-amd` = DirectML（备用）。

### 📝 文档

- README 的硬件需求与后端说明改为多后端版本，补上 `check_hardware.py` 验证步骤与项目结构。
- [AMD_SETUP.md](./docs/AMD_SETUP.md) 的 ROCm 章节改写为**实测可行的逐步指令**，包含两个安装陷阱（PyPI 版本号抢赢、缺少 `rocm_sdk`）与 MIOpen 缓存权限问题。
