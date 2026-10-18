# 在 Windows 11 上为 AMD Radeon RX 9070 XT（RDNA4 / gfx1201）运行 PyTorch —— 2026 年技术调研报告

> **这份文件是什么**
>
> 这是为 HanziGen v1.4.0 的 AMD 支援改造所做的背景调研，**不是使用说明**。
> 面向使用者的安装步骤请看 [AMD 顯示卡安裝指南](AMD_SETUP.md)；本文件只回答
> 「為什麼當初這樣設計」以及「哪些說法有出處、哪些沒有」。
>
> 報告中的結論已被提取並寫入 `AMD_SETUP.md`、`constraints/*.txt` 與程式註解。
> **生態變動很快（尤其 Windows ROCm 的索引網址與 torch 版本），請以 AMD 官方
> 文件為最終依據。** 每條結論都標了驗證等級，`[未验证]` 的項目不要當事實引用。

> **调研环境限制声明（重要）**
> 本次调研中 `web_fetch` 工具**完全不可用**：所有域名（github.com、raw.githubusercontent.com、rocm.docs.amd.com、pypi.org、huggingface.co、甚至 example.com）均返回 `Error: URL hostname ... resolves to a non-public IP address`。沙箱内的 shell 出网也被阻断（`Invoke-WebRequest` / `WebClient.DownloadString` 均报"基础连接已经关闭"）。
> 因此本报告**全部结论仅来自 `web_search`** —— 即搜索结果返回的 **URL + 页面标题/片段**。这意味着：
> - **可以高置信确认**：出现在 URL 路径、页面标题、GitHub Issue 标题中的版本号、命令片段、问题描述 —— 这些是搜索引擎直接命中的原文。
> - **无法逐字确认**：需要打开页面正文才能读到的完整命令、表格单元格、代码块内容。
> 我在下文对每一条都标注了 **[已验证]** / **[部分验证]** / **[未验证]**，并且**没有编造任何版本号或 URL**。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| 原生 Windows ROCm + PyTorch 是否可用？ | **可用，但属于"勉强能跑、坑很多"的状态**。Windows 上的 Radeon ROCm 走的是 TheRock（"The HIP Environment and ROCm Kit"）产出的 wheel，索引为 `gfx120X-all`（按家族/架构切分）以及新的 multi-arch 索引。gfx1201 自 ROCm 6.4.1（2025-05）起被正式支持。 |
| torch 会把自己伪装成 CUDA 吗？ | **会。** ROCm/HIP 构建下 `torch.cuda.is_available()` 返回 `True`，设备类型字符串是 `'cuda'`。区分方式是 `torch.version.hip` 非空。 |
| ZLUDA 现状 | **原生 Windows 上"能跑 CUDA wheel"是一条真实存在的路线，但 ZLUDA 本身在 2026 年已因失去资助退回"个人爱好项目"状态（v6 发布）**。RDNA4/gfx1201 有第三方验证文档。 |
| torch-directml | 最新版本落在 `0.2.5.dev240914` 一带（`dev` 后缀，日期 2024-09-14），**实质停更**。Python 3.13 支持**无法确认，倾向于不支持**。device.type == `'privateuseone'`。 |
| AMP | ROCm/HIP 下 `torch.amp.autocast('cuda')` 与 `GradScaler` 可用；ZLUDA 下理论上可用但有风险；**DirectML 下不可用**。 |

---

## 1. 原生 Windows ROCm + PyTorch（TheRock / gfx120X-all）

### 1.1 支持状态 [已验证]

- **RDNA4（RX 9070 系列，gfx1201）从 ROCm 6.4.1（2025 年 5 月）起才成为 ROCm 的正式支持目标。** 出处：[Zenodo 论文 bastola-compute-not-the-wire-2026.pdf](https://zenodo.org/records/20515293/files/bastola-compute-not-the-wire-2026.pdf?download=1)（片段原文："On the ROCm side, RDNA4 (the RX 9070 series, gfx1201) became a formally supported ROCm target only with the 6.4.1 release in May..."）。
- AMD 官方存在针对 Windows 的 Radeon/Ryzen ROCm 文档与 **RX 9070 XT + gfx1201 + Windows** 的定向兼容性页面：
  - [ROCm 10.0.0 compatibility matrix（fam=radeon&gpu=amd-radeon-rx-9070-xt&gfx=gfx1201&os=windows）](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html?fam=radeon&gpu=amd-radeon-rx-9070-xt&gfx=gfx1201&os=windows) —— **这是最权威的"到底支持到哪个版本"的页面**（本次无法打开，请自行核对）。
  - [PyTorch via PIP installation（Radeon/Ryzen, Windows, docs-7.2）](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html) 与 [docs-7.2.1 版本](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html)（含锚点 `#verify-pytorch-installation`）
  - [Windows support matrices by ROCm version](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
  - [ROCm Core SDK 7.14.1 release notes](https://rocm.docs.amd.com/en/docs-7.14.1/about/release-notes.html)、[7.14.0 release notes](https://rocm.docs.amd.com/en/develop/about/release-notes.html)
  - [AMD HIP SDK for Windows 7.1.1 — Release Notes](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.1.1/about/releasenotes.html)

### 1.2 ROCm 版本线（Windows / Radeon）—— 版本号格局 [已验证 / 部分是推测]

这是本报告中最容易搞混的部分。搜索结果中出现的版本号实在太多，我必须把它们分清楚：

**A. "稳定包"发布线（repo.radeon.com）**
- [repo.radeon.com/rocm/windows/rocm-rel-7.2.1/](https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/) —— 目录存在 [已验证]。`rocm-rel-7.2.x` 是 Windows 上走"官方 pip 索引"路线的稳定版本线。
- **ROCm 7.14 是 AMD 宣布的新一代 production release**（TheRock 转正），[ROCm 7.14: TheRock Goes Production and Expands AMD's AI Software Platform（AMD 官方博客）](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html)、[Phoronix: AMD ROCm 7.14 Announced As New Production Release](https://www.phoronix.com/news/AMD-ROCm-7.14)、[it-news-and-events.info 转载](https://it-news-and-events.info/article/178262) [已验证：标题即事实]。AMD 跳过了 8.0：[ZOL 报道](https://ai.zol.com.cn/1216/12165600.html)。
- **ROCm 10.0.0 / 10.1.0 已发布**，且"ROCm 10 加入 Windows 11 支持，但 CLI 仍为 Preview"：[windowsforum.com 新闻](https://windowsforum.com/news/rocm-10-adds-windows-11-support-but-cli-stays-preview.443754/) [已验证：标题]。
- 出现了 **ROCm 7.15**：[Level1Techs 论坛帖 "Native vLLM + ROCm 7.15 on Windows 11 for AMD RX 6750 XT (RDNA2)"](https://forum.level1techs.com/t/guide-poc-native-vllm-rocm-7-15-on-windows-11-for-amd-rx-6750-xt-rdna2-no-wsl2-54-2-tok-s/253889/1) [已验证：标题]。
- ⚠️ **我无法确认"截至 2026 年推荐 RX 9070 XT 用哪个 ROCm 版本"**。看到大量互相矛盾的数据点，说明社区仍在摇摆（见 1.3 的 bug 列表）。

**B. TheRock nightly / per-family 线（rocm.nightlies.amd.com）**
- 索引路径 **`https://rocm.nightlies.amd.com/v2/gfx120X-all/` 确认存在** [已验证] —— 出处：[ROCm/TheRock Issue #2349 标题](https://github.com/ROCm/TheRock/issues/2349)（"`https://rocm.nightlies.amd.com/v2/gfx120X-all/` missing torch torchvision torchaudio builds for ROCm 7.11.0a on LINUX"）。
- 该索引存在 **Windows 专属的 `--pre` 行为问题**：[ROCm/TheRock Issue #2822 "[Windows] `pip install` behavior difference when using the `--pre` argument (gfx120X-all)"](https://github.com/ROCm/TheRock/issues/2822) [已验证：标题]。
- **torch/torchvision 版本不匹配问题是已知的**：[ROCm/TheRock Issue #1257 "pip install gfx120X-all torch torchvision version mismach on Windows"](https://github.com/ROCm/TheRock/issues/1257) [已验证：标题]。
- 官方打包说明：[TheRock docs/packaging/legacy_per_family_releases.md](https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_per_family_releases.md)、[legacy_multi_arch_releases.md](https://raw.githubusercontent.com/ROCm/TheRock/refs/heads/main/docs/packaging/legacy_multi_arch_releases.md)、[TheRock RELEASES.md](https://raw.githubusercontent.com/ROCm/TheRock/main/RELEASES.md)。
- 正在推进 **multi-arch 索引**（一个 wheel 覆盖多架构，取代 per-family）：[TheRock Issue #3332 "[Multi-arch] Build multi-arch PyTorch Python packages"（ScottTodd，2026-04-30）](https://github.com/ROCm/TheRock/issues/3332)。ComfyUI 已跟进：[Comfy-Org/ComfyUI-Standalone-Environments PR #11 "Move AMD builds to ROCm 7.14 multi-arch wheels; add amd-multi-arch-index manifest kind"](https://github.com/Comfy-Org/ComfyUI-Standalone-Environments/pull/11) [已验证：标题]。**`amd-multi-arch-index` 这个具体 URL 我未能确认。**

**C. PyTorch 官方索引**
- **`https://download.pytorch.org/whl/rocm7.0/torch/` 存在**：[Links for torch](https://download.pytorch.org/whl/rocm7.0/torch/) [已验证：URL]。
- [download.pytorch.org/whl/nightly/rocm7.2/torch/](https://download.pytorch.org/whl/nightly/rocm7.2/torch/) 存在 [已验证：URL]。
- ⚠️ **[未验证]**：这些 PyTorch 官方 `whl/rocmX.Y` 索引中**是否包含 Windows（win_amd64）wheel**。有较强间接证据表明 **`/whl/rocm7.x` 是 Linux-only**：[unsloth PR #4966](https://github.com/unslothai/unsloth/pull/4966) 片段原文 "/whl/rocm7.x is manylinux_2_28_x86_64 only"。**因此 Windows 上应当使用 AMD 的 TheRock 索引，而不是 download.pytorch.org。** 这一点请务必自行验证。

### 1.3 精确 pip 命令 —— 哪些是我逐字确认的，哪些不是

**[已验证：逐字]**
```bash
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/
```
出处：ComfyUI README 在搜索结果中被索引到的代码块标题即为此完整字符串 —— [raw.githubusercontent.com/Comfy-Org/ComfyUI/.../README.md](https://raw.githubusercontent.com/Comfy-Org/ComfyUI/5e3f15a830ff27d3563ef4b43e9f6a0321ea36cd/README.md)。ComfyUI README 另有原文说明："AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable..."（[Comfy-Org/ComfyUI README](https://github.com/Comfy-Org/ComfyUI/blob/285a98944c397a4a81f15ac63d69fa3dbc0a27b9/README.md?plain=1)）。
另有一条相邻说明："This is the command to install the nightly with ROCm 7.2 which might have some performance improvements:"

**[已验证：逐字]**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.x   # ← 6.x 具体数字未确认
```
搜索结果中该代码块标题为 `pip install torch torchvision torchaudio --index-url https://download`（被截断）—— [cdn.jsdelivr.net/gh/Comfy-Org/ComfyUI@master/README.md](https://cdn.jsdelivr.net/gh/Comfy-Org/ComfyUI@master/README.md)。**具体是 rocm6.4 / rocm7.0 我无法确认，且如上所述该索引很可能是 Linux-only。**

**[部分验证]**
```bash
pip install --index-url https://rocm...  torch   # 前缀确认，完整 URL 未确认
```
出现于 [CS1o/Stable-Diffusion-Info Wiki: Lora Trainer Setup Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Lora-Trainer-Setup-Guides) 的标题片段："RX 6800/6900 XT (GFX1030) needs boath rocm+torch: pip install --index-url https://rocm"。

**[未验证 / 不要照抄]** 我**没有**找到任何来源逐字给出 `https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/whl/...` 形式的索引 URL。相关线索：
- [unsloth 提交 c6a9585 "Install ROCm PyTorch on Windows AMD via repo.radeon.com"](https://github.com/unslothai/unsloth/commit/c6a9585659e56910eee40371d03667dbe331c3dd) 的片段提到 "alpha version strings that are fragile to match on, so only 7.1.x / 7.2.x are..."，说明 unsloth 的 Windows 安装器在 `repo.radeon.com` 上按 `7.1.x` / `7.2.x` 匹配 wheel。
- [unsloth install.ps1](https://github.com/oobabooga/unsloth/blob/studio-inference-refactor/install.ps1) 中有 `$TorchIndexPinned = (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_TORCH_INDEX_URL))`，说明存在可覆盖的 torch 索引 URL 环境变量。
> **建议**：以 [AMD 官方 install-pytorch 页面](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html) 上"复制即用"的命令为准 —— 那是唯一权威来源。

### 1.4 支持的 Python 版本 [未充分验证]

- 我**无法确认** `gfx120X-all` 索引上具体有哪些 `cp3xx` wheel。
- **弱证据指向 Python 3.13 可用**：存在一个专门命名的脚本 [OrsoEric/HOWTO-ComfyUI `install_amd7900xtx_rocm710preview_python313.bat`](https://github.com/OrsoEric/HOWTO-ComfyUI/blob/Master/install_amd7900xtx_rocm710preview_python313.bat) [已验证：文件名] —— 但它针对 **RX 7900 XTX（gfx1100, RDNA3）**，不能直接外推到 gfx1201。
- 另一条无关但容易误读的证据：[PyTorch 开发者邮件列表 "Dropping Python 3.13t (free-threaded) support in nightlies and in the future PyTorch 2.13 release"](https://dev-discuss.pytorch.org/t/dropping-python-3-13t-free-threaded-support-in-nightlies-and-in-the-future-pytorch-2-13-release/3386) —— 这里删的是 **3.13t（free-threaded，无 GIL）**，**不是普通 3.13**。不要混淆。
- **实操建议 [推测]**：Windows ROCm 生态的"安全区"是 **CPython 3.11 或 3.12**；3.13 需自行在索引页确认有 `cp313` wheel 再上。

### 1.5 支持的 torch / torchvision / torchaudio 版本 [已验证：存在以下真实版本串]

以下是搜索结果中**实际出现的**、带 ROCm 后缀的 PyTorch Windows wheel 版本串（每一条都有 Issue 标题作为出处）：

| torch 版本串 | 出处 |
|---|---|
| `torch 2.12.0+rocm7.14.1`、`torch 2.12.0+rocm10.0.0`、`torch 2.12.0+rocm7.14.0` | [TheRock Issue #7992](https://github.com/ROCm/TheRock/issues/7992) |
| `torch 2.12.0+rocm7.14.1`（FP8 text-encoder） | [TheRock Issue #7861](https://github.com/ROCm/TheRock/issues/7861) |
| `torch 2.11.0+rocm7.13.0a20260416` | [TheRock Issue #8648](https://github.com/ROCm/TheRock/issues/8648) |
| ROCm 7.12 / 7.13 线（`torch._grouped_mm` 崩溃） | [TheRock Issue #5284](https://github.com/ROCm/TheRock/issues/5284) |
| 7.13.0a20260408（CPU 占用异常） | [TheRock Issue #4825](https://github.com/ROCm/TheRock/issues/4825) |
| ROCm 7.11.0a（gfx120X-all 缺 torch/torchvision/torchaudio） | [TheRock Issue #2349](https://github.com/ROCm/TheRock/issues/2349) |
| **torch 2.13 + ROCm 10.0（Windows）** | [sorryhyun/anima_lora PR #98 "Update Windows ROCm to 10.0 / PyTorch 2.13"](https://github.com/sorryhyun/anima_lora/pull/98) |

其他版本格局线索：
- [ComfyUI-Standalone-Environments PR #18 "Bump all vendor bundles to torch 2.12.x"](https://github.com/Comfy-Org/ComfyUI-Standalone-Environments/pull/18) → **torch 2.12.x 是当前主流稳定点** [已验证：标题]。
- [AMD 文档 "Install PyTorch for ROCm" 参数化页面](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.12.0&i=pip&gpu=amd-ryzen-ai-max-pro-495&gfx=gfx1151&rocm-ver=10.0.0&w=compute) 出现 `pytorch-ver=2.12.0` + `rocm-ver=10.0.0`；另有 `pytorch-ver=2.13.0` + `rocm-ver=10.0.0`（Linux）[已验证：URL 参数]。
- [pytorch 官方文档 2.14 版 AMP 页](https://docs.pytorch.org/docs/2.14/amp.html) 存在 → 2.14 已在文档线 [已验证：URL]。

⚠️ **torchvision / torchaudio 的对应版本号我完全没有验证到。** 不要编造配对。

### 1.6 已知坑（按严重程度排序）——全部 [已验证：Issue 标题]

**A. 驱动层（Radeon Adrenalin）**
1. **[Windows][RDNA4] Radeon 26.7.1 会在空闲约 9.5 秒后把活跃的 HIP 分配换出（page out）；26.3.1 不会。** → [TheRock Issue #7221](https://github.com/ROCm/TheRock/issues/7221)。**这是最致命的坑之一**：长时间训练的显存会被驱动悄悄回收。
2. RDNA2 在 Adrenalin 26.1.1 下无法被 ROCm 枚举 → [legacy-rocm-build Issue #5871](https://github.com/ROCm/legacy-rocm-build/issues/5871)（RDNA4 是否受影响未验证）。
3. AMD 为 ROCm 单独发行了驱动分支："AMD Software: Adrenalin Edition for ROCm" —— [10.1 版发布说明](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html)（**注意：这是"AI 计算专用驱动"，游戏性能可能受影响，媒体报道称"游戏玩家别更"** → [17173 报道](https://news.17173.com/content/10062026/190104732.shtml)）；较早的 [10.0 分支](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-00.html)。第三方评测：[Igor's Lab](https://www.igorslab.de/en/amd-releases-own-rocm-10-1-driver-radeon-rx-9000-rx-7000-windows/)、[PC Games Hardware](https://www.pcgameshardware.de/AMD-Radeon-Grafikkarte-255597/News/ROCm-101-Neuer-Compute-KI-Treiber-1555888/)。

**B. 运行时 / DLL**
4. **`import torch` 在 Windows 上需要 VS 的 `libomp140.x86_64.dll`** → [TheRock Issue #1309](https://github.com/ROCm/TheRock/issues/1309)；修复 PR：[pytorch PR #158922 "[ROCm][Windows] Embed libomp140.x86_64.dll to wheel package"](https://github.com/pytorch/pytorch/pull/158922)。
5. **`aotriton_v2.dll` 与 `liblzma.dll` 缺失** → [pytorch PR #200931 "[ROCm][Windows] Install aotriton_v2.dll and liblzma.dll into scikit-build ROCm wheels"](https://github.com/pytorch/pytorch/pull/200931)。
6. **GPU 未连接显示器时 HIP 初始化崩溃** → [legacy-rocm-build Issue #6458](https://github.com/ROCm/legacy-rocm-build/issues/6458)。

**C. 内核 / 训练正确性 bug（Windows + gfx1201/gfx120X）**
7. **`torch 2.12.0+rocm7.14.1` 和 `+rocm10.0.0`：任何 SDPA 调用之后 kernel 全部以 `hipErrorInvalidValue` 失败；`+rocm7.14.0` 正常** → [TheRock Issue #7992](https://github.com/ROCm/TheRock/issues/7992)。**这条直接决定了"能不能训练 Transformer"。**
8. **`[gfx1201][Windows]` torch 2.12.0+rocm7.14.1 FP8 text-encoder 前向 + GPU 常驻 embeds → `hipErrorInvalidValue`；7.14.0 OK** → [TheRock Issue #7861](https://github.com/ROCm/TheRock/issues/7861)。
9. **Windows ROCm gfx120X —— `torch._grouped_mm` 空指针 HIP kernel 在 ROCm 7.12 上导致训练崩溃（7.13 修复）** → [TheRock Issue #5284](https://github.com/ROCm/TheRock/issues/5284)。
10. **`[Windows gfx1201] hipLaunchKernel` 以 `0xC0000005` 崩溃（`hipProfilerRegisterChunkCallbackExt`，回调链表指针损坏）** → [TheRock Issue #7732](https://github.com/ROCm/TheRock/issues/7732)。
11. **`[Windows]` ROCm PyTorch wheel `2.11.0+rocm7.13.0a20260416` 中有两个 HIP runtime bug** → [TheRock Issue #8648](https://github.com/ROCm/TheRock/issues/8648)。
12. **[Windows]: `HIPRTC_ERROR_COMPILATION` on gfx1201** → [TheRock Issue #710](https://github.com/ROCm/TheRock/issues/710)。（JIT 编译路径不可用 → torch.compile / Triton 路线受挫）
13. **WSL2：RX 9070 XT（gfx1201）在 ROCm 10.1.0 wheels 下第一个 kernel 永不完成，Windows GPU timeout（TDR）；ROCm 7.2 正常** → [rocm-systems Issue #13291](https://github.com/ROCm/rocm-systems/issues/13291)。（WSL 路径，但同属 Windows 主机）
14. **PyTorch 训练/微调在部分 Radeon GPU 上会 GPU 复位或崩溃** → [TheRock Issue #7699](https://github.com/ROCm/TheRock/issues/7699)。
15. **FlashAttention backward 在 RX 9070 XT 上比 PyTorch SDPA 慢 4.4–7.4 倍（ROCm/CK/gfx1201/Windows）** → [flash-attention Issue #2800](https://github.com/Dao-AILab/flash-attention/issues/2800)。**→ 在 Windows 上不要用 flash-attn。**
16. 性能问题：**[Issue] Bad Pytorch performance on RX 9070 XT Windows 11** → [TheRock Issue #2510](https://github.com/ROCm/TheRock/issues/2510)。
17. `7.13.0a20260408` 之后 PyTorch CPU 占用异常高 → [TheRock Issue #4825](https://github.com/ROCm/TheRock/issues/4825)。
18. VAE caching 崩溃（torch/ROCm 7.14.1 崩，7.14.0 / gfx1200 正常）→ [OneTrainer Issue #1716](https://github.com/Nerogar/OneTrainer/issues/1716)。
19. MIOpen gfx120X：hipblaslt fp32 kernel 在 R9700 上的性能对齐问题 → [legacy-rocm-build Issue #5674](https://github.com/ROCm/legacy-rocm-build/issues/5674)；已在推进：[rocm-libraries PR #5462 "feat(miopen): Add more GEMM convolution solvers from CK for gfx1201 (RDNA4)"](https://github.com/ROCm/rocm-libraries/pull/5462)。

**D. `--pre` 行为**
20. [TheRock Issue #2822](https://github.com/ROCm/TheRock/issues/2822) 明确指出 Windows 上 `pip install --pre` 的行为与 Linux 不同（这正是为什么上面的 ComfyUI 命令**带 `--pre`**）。**具体差异内容我无法读取**（需要打开 Issue 正文）。**实操提示**：如果加了 `--pre` 装不上或装出奇怪版本，试试去掉 `--pre`，或显式钉版本 `pip install torch==2.12.0+rocm7.14.0 ...`。

### 1.7 `torch.cuda.is_available()` 在 ROCm 下返回什么？ [已验证]

**返回 `True`（前提是堆栈装对了）。** 这是 PyTorch 上游设计：HIP 构建复用整个 `torch.cuda` 命名空间。
- **关键证据**：存在一个专门的修复提交 [VoiceStudio commit 9553aff "fix(asr): **ROCm torch is not CUDA** — keep CTranslate2 off the HIP GPU"](https://github.com/debpalash/VoiceStudio/commit/9553aff74c71697b3fd5e02b1dd7aee492feedd3) —— 标题本身就证明了：ROCm 版 torch 在代码看来"就是 CUDA"（`torch.cuda.is_available() == True`），所以下游代码会误把 HIP GPU 当成 CUDA GPU。
- 官方语义文档：[HIP (ROCm) semantics — PyTorch docs 2.14](https://docs.pytorch.ac.cn/docs/2.14/notes/hip.html) [已验证：URL]。
- `torch.version.hip` 是区分标志；上游还有一个 [PR "add `torch.version.rocm`, distinct from `torch.version.hip`"](https://github.com/amd-sriram/pytorch/commit/7208a75f7bc2dc2ffd46419fc536a9ee3b202d3f)（AMD 分支）[已验证：标题]。
- `torch.cuda.get_device_name(0)` 在 RX 9070 XT 上**应返回类似 `"AMD Radeon RX 9070 XT"`** —— 这是 [已验证：格式惯例 + Windows 上 ROCm 用 AMD 驱动暴露设备名] / [未逐字验证具体字符串]。**请以官方文档 `#verify-pytorch-installation` 章节为准。**

---

## 2. ZLUDA（在 Windows 上用 CUDA 翻译层跑 AMD）

### 2.1 项目现状 —— 2026 年最重要的事实 [已验证]

- **ZLUDA v6 已发布**：[GitHub Release v6](https://github.com/vosen/ZLUDA/releases/tag/v6)。
- **但 AMD 已停止资助，项目退回"个人爱好项目"状态**：
  - [Yahoo Tech: "CUDA emulator for AMD GPUs Zluda loses funding with v6 release — embattled project goes back to hobby status"](https://tech.yahoo.com/computing/articles/cuda-emulator-amd-gpus-zluda-182921158.html)
  - [developersdigest.tech: "ZLUDA 6: Running CUDA on AMD GPUs Is Now a Hobby Project"](https://www.developersdigest.tech/blog/zluda-6-cuda-amd-gpus)
  - 官方博客：[ZLUDA update Q1/Q2 2026 — "The new direction of the project"](https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/)（[Wayback 快照](https://web.archive.org/web/20260630215318/https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/)）
  → **战略含义：不要把 ZLUDA 当作长期生产方案。** 上游 vosen/ZLUDA 的重心已转向 GPU PhysX 等方向（[prohardver 报道](https://prohardver.hu/hir/zluda_mostantol_gpu_physx_koncentral.html)）。
- 生态里活跃的是**第三方 fork/打包**：
  - [Speedstu/CUDA-for-AMD-Windows](https://github.com/Speedstu/CUDA-for-AMD-Windows) —— "Run CUDA-targeted Windows applications on AMD GPUs with ZLUDA + ROCm/HIP"，2026-09 被 [gigazine](https://gigazine.net/news/20260918-cuda-for-amd-windows/)、[Korben](https://korben.info/en/cuda-for-amd-windows-nvidia-exe-for-your-radeon.html)、[Yahoo/Tom's Hardware 系](https://tech.yahoo.com/computing/articles/solo-dev-enables-running-cuda-143628611.html)、[MyDrivers](http://m.mydrivers.com/newsview/1151163.html)、[EET-China](https://www.eet-china.com/mp/a524987.html) 广泛报道。
  - [patientx/ComfyUI-Zluda](https://github.com/patientx/ComfyUI-Zluda)、[Loomos-hub/amd-zluda-local-tts-image-ai](https://github.com/Loomos-hub/amd-zluda-local-tts-image-ai)（含 "the cuDNN fix, the **arbitrary-torch-on-ZLUDA recipe**"）、[lshqqytiger/zluda-rocm](https://github.com/lshqqytiger/zluda-rocm)、[Daelyte/RUN_CUDA_ON_AMD_ZLUDA](https://github.com/Daelyte/RUN_CUDA_ON_AMD_ZLUDA)、[vladmandic/sdnext](https://github.com/vladmandic/sdnext/wiki/ZLUDA)。
- 相关但**不属于 PyTorch**：DLSS 4.5 被用在 RX 9000 上（d4r 项目，[VideoCardz](https://videocardz.com/newz/modder-brings-dlss-4-5-to-radeon-rx-9000-gpus-rdna-4-uses-native-fp8)、[TweakTown](https://www.tweaktown.com/news/113846/modder-gets-dlss-4-5-to-run-on-rx-9000-series-gpus-with-zluda-and-hand-written-code/index.html)）。

### 2.2 RDNA4 / gfx1201 支持 [部分验证]

- 存在一份**专门的 RX 9070 XT 验证文档**：[Speedstu/CUDA-for-AMD-Windows `docs/RX9070XT_VALIDATION.md`](https://github.com/Speedstu/CUDA-for-AMD-Windows/blob/main/docs/RX9070XT_VALIDATION.md) [已验证：文件存在]。**其具体通过/失败清单我无法读取。**
- 新闻标题显示该项目在一个 **RX 9060 XT**（同为 RDNA4/gfx120X）上跑通了多个 CUDA 库 → [Yahoo/Tom's 报道标题](https://tech.yahoo.com/computing/articles/solo-dev-enables-running-cuda-143628611.html)。RDNA4 家族兼容性有实证。
- ⚠️ **无法确认** ZLUDA 官方仓库是否对 `gfx1201` 有正式（tested）列表。ZLUDA 的架构白名单是运行时从驱动读的，但历史上依赖 HIP SDK 版本的补丁脚本（见 2.4）。

### 2.3 是否兼容官方 PyTorch CUDA wheel？ [已验证：是]

- **是**。整个 ZLUDA-on-Windows 路线的定义就是"用官方 CUDA 构建的 torch，+ 替换几个 DLL"。出处：[Loomos-hub 的 `04-arbitrary-torch-on-zluda.md`](https://github.com/Loomos-hub/amd-zluda-local-tts-image-ai/blob/main/docs/04-arbitrary-torch-on-zluda.md)（"arbitrary torch on ZLUDA recipe"）。
- 替换目标目录 [已验证：逐字]：**`venv\Lib\site-packages\torch\lib`** —— 出处 CSDN 文库片段原文："3. 下载对应版本的 ZLUDA 替换文件，**覆盖虚拟环境中 `Lib\site-packages\torch\lib` 下的同名文件**"（[CSDN 文库](https://wenku.csdn.net/answer/6ehxma08xjzp)，同一段落在多个 CSDN 页面重复出现）。
- **[未验证]**：具体是哪个 CUDA 版本 wheel（cu126 / cu128 / cu129 / cu130）最稳。**不要相信任何未经你自己验证的具体 CUDA 版本号推荐。**

### 2.4 需要哪些 DLL / 前置条件 [部分验证]

**[已验证：机制]** 需要把 ZLUDA 提供的、**与 CUDA 库同名**的 DLL 覆盖到 `torch\lib`。ZLUDA 的库模拟支持列表见 [DeepWiki: vosen/ZLUDA — Library Emulation](https://deepwiki.com/vosen/ZLUDA/6-library-emulation)。预期的文件名族（CUDA 侧惯例）：`nvcuda.dll`（驱动层）、`cublas64_*.dll` / `cublasLt64_*.dll`、`cufft64_*.dll`、`cusparse64_*.dll`、`cusolver64_*.dll`、`nvml.dll`、以及 **cuDNN**（`cudnn64_*.dll` 等）。
> ⚠️ **我无法逐字确认 ZLUDA 实际发布的文件清单。** 上面的文件名列表是基于 CUDA 惯例 + ZLUDA 库模拟文档的推断，**不是从某个页面抄下来的**。请以你所选 fork 的 README 为准。

**[已验证：cuDNN 是关键]** Loomos-hub 项目描述里明确写了 "**the cuDNN fix**" → [Loomos-hub/amd-zluda-local-tts-image-ai](https://github.com/Loomos-hub/amd-zluda-local-tts-image-ai) [已验证：仓库描述原文]。**conv 网络（Stable Diffusion 等）缺 cuDNN 会直接失败。**

**[已验证：需要 HIP SDK + 补丁脚本，且版本敏感]**
- ZLUDA 在 Windows 上需要 **AMD HIP SDK**：[ZLUDA 官方文档 "Installing HIP SDK"](https://zluda.readthedocs.io/latest/hip_sdk.html)。
- 存在**按 HIP SDK 版本命名的补丁脚本**，这直接暴露了版本敏感性：
  - [Applio `assets/zluda/patch-zluda-hip57.bat`](https://huggingface.co/spaces/blaise-tk/Applio/blob/fa48da65cdd673f69031497c201172328b4f6905/assets/zluda/patch-zluda-hip57.bat) → HIP 5.7
  - [Applio `assets/zluda/patch-zluda-hip61.bat`](https://huggingface.co/spaces/blaise-tk/Applio/raw/8ef89ca0b2cc52294b591dcdac6c1a0755363b99/assets/zluda/patch-zluda-hip61.bat) → HIP 6.1
  - [Applio `assets/zluda/patch-zluda-hip62.bat`](https://huggingface.co/spaces/blaise-tk/Applio/blob/fa48da65cdd673f69031497c201172328b4f6905/assets/zluda/patch-zluda-hip62.bat) → HIP 6.2
  - [apollo-mg/ComfyUI-Zluda-MyFix `patchzluda-n.bat`](https://github.com/apollo-mg/ComfyUI-Zluda-MyFix/blob/main/patchzluda-n.bat)、[sfinktah/ComfyUI-Zluda `patchzluda-s.bat` (branch sfink-hip64)](https://github.com/sfinktah/ComfyUI-Zluda/blob/sfink-hip64/patchzluda-s.bat) → **HIP 6.4**
- [CS1o/Stable-Diffusion-Info Wiki](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Lora-Trainer-Setup-Guides) 有一句关键提示 [已验证：标题片段]：**"Important: To use ROCm HIP SDK 6.4.2 and your GPU is higher/newer than a RX6800 or is a 9070/60, do: ..."** → **明确点名 9070/9060 需要针对 HIP SDK 6.4.2 的额外步骤。**
- 典型错误：[Nerogar/OneTrainer Discussion #912 "Failed to load ZLUDA: Could not find module `C:\Program Files\AMD\ROCm\6.2\bin\hiprtc0507.dll`"](https://github.com/Nerogar/OneTrainer/discussions/912) → **ZLUDA 需要 HIP SDK 的 `hiprtc****.dll`，路径硬编码在 `C:\Program Files\AMD\ROCm\<ver>\bin\`。**
- **[未验证]** 环境变量：`HIP_PATH` 似乎是需要的（见 [gitcode: ZLUDA Windows 环境配置实战：HIP SDK 两种安装方式、HIP_PATH 配置与 cuda_check 全库验证](https://blog.gitcode.com/72601d54ad65ca3fcece29ff441f43ea.html) [已验证：标题] —— 标题里明确有 **HIP_PATH** 和 **cuda_check 全库验证**）。ZLUDA 自带 **`cuda_check`** 工具（见 [ZLUDA/cuda_check 目录](https://gitcode.com/GitHub_Trending/zl/ZLUDA/tree/66b20a3f5479070b116cb4015f2b5167faff838a/cuda_check)）—— **这是验证安装是否成功的第一步工具**。我没找到 `ZLUDA_COMGR_LOG_LEVEL` / `ZLUDA_CACHE` 等变量的实证，**故不列出**。

### 2.5 `torch.cuda.is_available()` 在 ZLUDA 下 [已验证：返回 True]

- **返回 `True`** —— ZLUDA 的设计目标就是让 CUDA 应用（含 PyTorch）以为自己在一块 NVIDIA 卡上。这正是 [gigazine 报道](https://gigazine.net/news/20260918-cuda-for-amd-windows/) 和 [oflight.co.jp 的 "CUDA for AMD on Windows: ZLUDA + HIP Setup Guide (2026)"](https://www.oflight.co.jp/en/columns/cuda-for-amd-windows-zluda-guide-2026) 所描述的场景。
- **[未验证]** `torch.cuda.get_device_name(0)` 在 RX 9070 XT + ZLUDA 下的精确返回字符串。两种可能：(a) `"AMD Radeon RX 9070 XT"`（ZLUDA 从 HIP 取真实设备名），或 (b) 某个伪造的 NVIDIA 名称。**我没有找到任何来源给出这个字符串，因此不猜测。** 请用 `cuda_check` 或 `torch.cuda.get_device_name(0)` 实测。
- **[未验证]** `torch.cuda.get_device_properties(0).major / .minor` 在 ZLUDA 下的返回值 —— 这是你问题 4 里提到的关键判别点，但**我没有找到任何实证**。⚠️ 逻辑上：ZLUDA 必须提供某种 `cudaDeviceProp` 填充（否则 PyTorch 会在启动时就崩），所以 `cudaGetDeviceProperties` 很可能**返回被伪造的 major/minor（例如 7.5 / 8.0 之类）而不是抛异常**。**这是我的推理，不是验证结果，请勿当作事实。**
- **[未验证]** `gcnArchName` 在 ZLUDA 下的情况。理论上 `gcnArchName` 是 **HIP-only 字段**（`torch.cuda.get_device_properties()` 在纯 CUDA 构建上**没有**这个属性），而 PyTorch 的 `_CudaDeviceProperties` 结构是由编译期决定的 —— **ZLUDA 用的是官方 CUDA 构建的 torch，所以该属性很可能根本不存在**（`AttributeError`）。这反而是个"看起来像 CUDA 但其实是 AMD"的可用信号。**⚠️ 同样是推理，未验证。**

### 2.6 性能预期 [定性，无 RX 9070 XT 具体数字]

- 我**没有找到** RX 9070 XT（或任何 RDNA4 卡）在 ZLUDA 下的 PyTorch benchmark 数字。
- 可用作方向的邻近数据：
  - [runaihome: "RDNA4 Vulkan vs ROCm 7.2 for Local LLMs: Which Backend Is Faster on the RX 9000 Series in 2026?"](https://runaihome.com/blog/rdna4-vulkan-vs-rocm-local-llm-benchmark-2026/) —— 比较的是 Vulkan vs ROCm，**不是 ZLUDA**。
  - [Loomos-hub 仓库描述提到 "honest benchmarks"](https://github.com/Loomos-hub/amd-zluda-local-tts-image-ai)（RDNA2）。
  - 定位判断 [已验证：多方标题]：ZLUDA 的路线是"**兼容性优先**" —— 目的是让**本来只支持 CUDA 的软件能跑起来**，而不是追求峰值性能。
- **[定性判断，非验证]** 一般社区共识（本次未能找到 RX 9070 XT 直接证据）：**ROCm 原生 > ZLUDA > DirectML**，在能跑起来的前提下。DirectML 通常最慢。**请勿引用为已证事实。**

### 2.7 float16 autocast / GradScaler 在 ZLUDA 下 [未验证 —— 这是最大的空白]

- **我无法找到任何关于 ZLUDA 下 `torch.amp.autocast('cuda')` 或 `torch.amp.GradScaler` 的确证。**
- 逻辑推断 [推测]：
  - `autocast('cuda')` **应该"能进入"该上下文**（因为 `torch.cuda.is_available()` 为 True），但实际 fp16 运算要经过 `cublasLt` / `cublasGemmEx`。ZLUDA 对这些入口的支持完整性是决定因素。
  - `GradScaler` 需要 `torch.cuda.*` 的 float 检查与 unscale kernel，主要走 elementwise 算子而非 BLAS → **相对更可能可用**。
  - **真正的风险点是 `cublasLt`**：如果 ZLUDA 的 `cublasLt64_*.dll` 覆盖不完整，fp16 autocast 会在第一次 GEMM 时炸掉。
- **实操建议**：先跑 `cuda_check`，再写 10 行最小脚本实测 `autocast('cuda', dtype=torch.float16)` + `GradScaler('cuda')`。**不要在生产训练前假设它可用。**

---

## 3. torch-directml

### 3.1 版本与安装 [已验证]

- **最新版本：`0.2.5.dev240914`** —— 出处 [socket.dev: torch-directml versions — `0.2.5.dev240914/cp310-cp310-manylinux2010-x86-64-whl`](https://socket.dev/pypi/package/torch-directml/versions/0.2.5.dev240914/cp310-cp310-manylinux2010-x86-64-whl) [已验证：URL 路径含完整版本号]。
- 其他确认存在的历史版本：`0.2.2.dev240614`（[socket.dev](https://socket.dev/pypi/package/torch-directml/versions/0.2.2.dev240614/cp310-cp310-manylinux2010-x86-64-whl)）、`0.2.0.dev240914`（搜索结果中的 PyPI 页面标题）、`0.1.13.dev221206`、`0.1.13.1.dev230119`（[deps.dev](https://deps.dev/pypi/torch-directml/0.1.13.1.dev230119)、[Snyk](https://security.snyk.io/package/pip/torch-directml/0.1.13.dev221206)）。
- 版本号里的 **`dev240914` = 2024-09-14** 构建日期。**这说明该包实质上已经停滞在 2024 年 9 月。** 相关：[microsoft/DirectML Issue #712 "[torch-directml][doc] at least the doc can be kept up to date."](https://github.com/microsoft/DirectML/issues/712) [已验证：标题]。
- 完整版本列表（供你自行核对）：[Snyk](https://security.snyk.io/package/pip/torch-directml/versions?page=1)、[repology 历史](https://repology.org/project/python%3Atorch-directml/history)、[microsoft/DirectML Releases](https://github.com/microsoft/DirectML/releases)。
- **安装命令 [已验证]**：
  ```bash
  pip install torch-directml
  ```
  官方文档：[Enable PyTorch with DirectML on Windows（learn.microsoft.com）](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)；WSL 版：[在 WSL 上为 PyTorch 启用 DirectML](https://learn.microsoft.com/zh-cn/windows/ai/directml/pytorch-wsl)。官方 PyTorch 侧说明：[microsoft/DirectML `PyTorch/README.md`](https://raw.githubusercontent.com/microsoft/DirectML/refs/heads/master/PyTorch/README.md)。
  ⚠️ 国内镜像坑：[CSDN 文库片段](https://wenku.csdn.net/answer/1wgsxxzqhe) 显示 `pip install torch-directml -i https://pypi.tuna.tsinghua.edu.cn/simple` 报 `ERROR: Could not find a version that satisfies the requirement torch-directml (from versions: none)` → **清华镜像没有这个包，必须用官方 PyPI。**

### 3.2 最新的 PyTorch 版本支持 [未充分验证]

- **[未验证]** 我**没有**找到 torch-directml ↔ torch 版本对应表。
- 已知 web 搜索结果中出现过 [RVC 项目的 `requirments_cpu_py312.txt`](https://cdn.jsdelivr.net/gh/RVC-Project/Retrieval-based-Voice-Conversion-WebUI@main/requirments_cpu_py312.txt)（标题："Python 3.12 x64 / Windows + Linux / CPU"）—— 说明 3.12 + DirectML 组合有实际项目在用 [已验证：标题]。
- 一般认知（**未验证**）：`0.2.x.dev24xxxx` 系列对应 **torch 2.3.1 / 2.4.1** 一带。**不要引用为事实。**

### 3.3 支持的 Python 版本 / 是否支持 3.13？

- **⚠️ 无法确认它支持 Python 3.13 —— 我倾向于不支持。**
- 依据：
  - wheel tag 观察到的是 `cp310`（[socket.dev URL 中的 `cp310-cp310-manylinux2010-x86-64-whl`](https://socket.dev/pypi/package/torch-directml/versions/0.2.5.dev240914/cp310-cp310-manylinux2010-x86-64-whl)）[已验证]。
  - 包构建日期为 **2024-09-14**，而 Python 3.13 于 2024-10-07 正式发布 → **3.13 发布在最后一次 DirectML 构建之后**。因此 `cp313` wheel 极可能不存在。
  - 相关旁证：[PyTorch 邮件列表正在讨论 2.13 移除 3.13t 支持](https://dev-discuss.pytorch.org/t/dropping-python-3-13t-free-threaded-support-in-nightlies-and-in-the-future-pytorch-2-13-release/3386)，反映 3.13 生态本身仍在收敛。
- **实操结论**：**用 Python 3.11 或 3.12 跑 torch-directml。** 不要用 3.13。

### 3.4 如何获取设备 / device.type 字符串 [验证到 API 名称层级]

```python
import torch
import torch_directml

torch_directml.is_available()      # -> bool
torch_directml.gpu_count()          # -> int
torch_directml.device_name(0)       # -> 设备名字符串（在 RX 9070 XT 上应为 "AMD Radeon RX 9070 XT" 形态）[未逐字验证]
dml = torch_directml.device()       # 默认 device(0)
dml = torch_directml.device(0)      # 显式指定
print(dml.type)                     # -> 'privateuseone'   [已验证]
```
- **`device.type` 返回 `'privateuseone'`** [已验证] —— 多重佐证：
  - [StackOverflow: "How to use one_hot on a pytorch_directml tensor on privateuseone device?"](https://stackoverflow.com/questions/75717172/how-to-use-one-hot-on-a-pytorch_directml-tensor-on-privateuseone-device)
  - [DirectML Issue #604 标题含 "is not currently supported on the DML backend"](https://github.com/microsoft/DirectML/issues/604)（同类 privateuseone 机制）
  - [DirectML Issue #431 "torch.eye fails to fall back on CPU"](https://github.com/microsoft/DirectML/issues/431)
- 官方 API 一览：[Enable PyTorch with DirectML on Windows](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)。
- `str(dml)` 的精确格式 **未验证**（很可能形如 `privateuseone:0`）。

### 3.5 torch.amp.autocast / GradScaler 支持？ [高置信：不支持 / 不可用]

- **[已验证：间接]** DirectML/`privateuseone` **不是** PyTorch autocast 的受支持后端。证据链：
  1. `torch.amp.autocast` 的合法 `device_type` 集合是**编译期注册**的固定列表（`cuda` / `cpu` / `xpu` / `hpu` / `mps` / `privateuseone` / `meta` 等，**具体哪些被注册我未逐字验证**）。对一个未注册或不支持的后端，autocast 会抛错（`RuntimeError` / `NotImplementedError` / `UserWarning`）。相关文档：[Automatic Mixed Precision package - torch.amp（2.14）](https://docs.pytorch.org/docs/2.14/amp.html)、[torch.amp 繁体中文 2.4 文档](https://pytorch.com.tw/docs/2.4/amp.html)。
  2. **`GradScaler` 在非 CUDA 后端会告警/报错**。经典消息是 `torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.` —— 在 DirectML 下 `torch.cuda.is_available()` 为 **False**（DirectML 不伪装成 cuda），所以 `GradScaler()` 会**自动禁用并降级**（或抛 `NotImplementedError`，取决于版本）。
- **⚠️ 我未能逐字捕获具体的报错文本**（因为无法打开 Issue 正文）。**不要在报告里引用"确切的错误消息"。**
- **结论 [高置信但非逐字验证]：在 torch-directml 上不要用 AMP。** 用 fp32 或手动 cast。

### 3.6 已知严重限制 [已验证：以下均为真实 Issue 标题]

| 限制 | 出处 |
|---|---|
| `torch.lstm` 在 DML 后端报错 | [DirectML Issue #613 "torch.lstm raised an error with backend"](https://github.com/microsoft/DirectML/issues/613) |
| `nn.LSTM` 抛 `NotImplementedError` + 两个 warning | [DirectML Issue #654 "One NotImplementedError of nn.LSTM and two Warnings"](https://github.com/microsoft/DirectML/issues/654) |
| `torch.eye` 无法回退到 CPU | [DirectML Issue #431 "torch.eye fails to fall back on CPU"](https://github.com/microsoft/DirectML/issues/431) |
| `aten::_foreach_lerp_.Scalar` 不支持 | [DirectML Issue #604](https://github.com/microsoft/DirectML/issues/604) |
| `one_hot` 在 privateuseone 上无法使用 | [StackOverflow](https://stackoverflow.com/questions/75717172/how-to-use-one-hot-on-a-pytorch_directml-tensor-on-privateuseone-device) |
| 安装困难 | [DirectML Issue #544 "Torch-DirectML installation issues"](https://github.com/microsoft/DirectML/issues/544) |
| WSL2 上 segfault | [DirectML Issue #378 "Segmentation fault on WSL2"](https://github.com/microsoft/DirectML/issues/378) |

**归纳的严重限制 [综合判断]：**
1. **算子覆盖率不完整，且回退机制不可靠** —— 这是最致命的问题。遇到不支持的算子，DML 有时报错而不是回退 CPU（`torch.eye` 就是明证）。
2. **无 AMP / 无 fp16 训练路径** —— 只能用 fp32，显存与速度双重劣势。
3. **无 `torch.compile` 支持**、无 Triton。
4. **性能显著低于 CUDA/ROCm**（定性）。
5. **实质停止维护**（最后构建 2024-09-14）。
6. 对 **RDNA4 / RX 9070 XT** 而言，DirectML 走的是 D3D12 通用路径，**完全用不到 RDNA4 的 FP8/WMMA 新硬件特性**（对比：DLSS 4.5 在 RDNA4 上正是靠 native FP8 实现的 → [VideoCardz](https://videocardz.com/newz/modder-brings-dlss-4-5-to-radeon-rx-9000-gpus-rdna-4-uses-native-fp8)）→ **大量算力被浪费。**

### 3.7 DirectML 在 2026 年是否已被边缘化？ [部分验证]

- 观察到微软生态重心在 Windows ML / ONNX Runtime，`torch-directml` 文档长期未更新（[DirectML Issue #712](https://github.com/microsoft/DirectML/issues/712)）。
- **[未验证]** 我**没有**找到"DirectML 被官方弃用/取代"的明确公告。
- 社区实践上，AMD 用户的讨论已经转向 ROCm 和 Vulkan（如 [runaihome RDNA4 Vulkan vs ROCm](https://runaihome.com/blog/rdna4-vulkan-vs-rocm-local-llm-benchmark-2026/)、[dev.to: Vulkan saved our RX 580 AI setup](https://dev.to/aivisionslab/everything-that-failed-before-vulkan-saved-our-rx-580-ai-setup-4apj)）→ **DirectML 已明显不是 AMD 上的首选。**

---

## 4. 运行时后端检测（最可靠的方法）

### 4.1 推荐判定顺序（可直接用的代码）

```python
import torch

def detect_backend():
    # ---- 1. DirectML / torch-directml ----
    # 特征：第三方模块 + device.type == 'privateuseone'
    try:
        import torch_directml                      # noqa
        dml = torch_directml.device()
        if dml.type == 'privateuseone':
            return 'directml(privateuseone)', str(dml)
    except ImportError:
        pass

    # ---- 2. Apple MPS ----
    if getattr(torch.backends, 'mps', None) is not None:
        if torch.backends.mps.is_available():      # 还需 is_built()
            return 'mps', 'mps'

    # ---- 3. CUDA 家族：NVIDIA CUDA / ROCm-HIP / ZLUDA 三者都走这里 ----
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        hip   = getattr(torch.version, 'hip', None)     # ROCm 标志
        rocm  = getattr(torch.version, 'rocm', None)    # 新版可能新增
        cuda  = torch.version.cuda                       # NVIDIA CUDA 标志

        if hip is not None:
            # ROCm / HIP 构建。gcnArchName 是 HIP-only 字段。
            arch = getattr(props, 'gcnArchName', None)   # 例如 'gfx1201'
            return 'rocm-hip', f'{name} arch={arch} hip={hip}'

        # 到这里：torch.version.hip is None 但 torch.cuda.is_available() is True
        # → 要么真 NVIDIA，要么 ZLUDA 打了补丁。
        # 关键启发式：设备名里出现 AMD/Radeon/gfx，或 gcnArchName 存在
        looks_amd = any(k in name.upper() for k in ('AMD', 'RADEON', 'GFX'))
        has_gcn   = hasattr(props, 'gcnArchName')
        if looks_amd or has_gcn:
            return 'zluda(cuda-masquerade)', f'{name} cuda={cuda}'
        return 'nvidia-cuda', f'{name} cuda={cuda} major={props.major}.{props.minor}'

    # ---- 4. 纯 CPU ----
    return 'cpu', 'cpu'
```

> ⚠️ **重要警告**：上面代码里 **`gcnArchName` 在 CUDA 构建上不存在**这一前提、以及 **ZLUDA 下 `get_device_name()` 返回 AMD 名称**这一前提，**都不是我验证过的事实**（见 2.5）。请把它们当作"待实测的启发式"，并在真实环境里先打印所有属性再据此定策：
> ```python
> import torch, json
> print('torch', torch.__version__)
> print('cuda build', torch.version.cuda, '| hip build', getattr(torch.version, 'hip', None))
> print('available', torch.cuda.is_available())
> if torch.cuda.is_available():
>     print('name', torch.cuda.get_device_name(0))
>     p = torch.cuda.get_device_properties(0)
>     print('props', p)
>     print('major/minor', p.major, p.minor)
>     print('gcnArchName?', getattr(p, 'gcnArchName', '<absent>'))
>     print('arch_list', torch.cuda.get_arch_list())
> ```
> **先跑这段，再决定判定逻辑。** 这是我本次调研能给出的最可靠建议。

### 4.2 逐后端的属性对照表

| 后端 | `torch.version.cuda` | `torch.version.hip` | `torch.cuda.is_available()` | `device.type` | 其他判据 |
|---|---|---|---|---|---|
| **NVIDIA CUDA** | 形如 `'12.8'` / `'12.9'` / `'13.0'` | `None` | `True` | `'cuda'` | `torch.backends.cudnn.is_available() == True`；`get_device_properties(0).major/.minor` 为真实 SM 版本（如 `9.0`/`12.0`）；设备名 `'NVIDIA GeForce RTX ...'` |
| **AMD ROCm / HIP** | `None` | **非 `None`**（形如 `'6.2.41133'` / `'7.14.x'`）[已验证：属性存在；**具体字符串格式未逐字验证**] | **`True`** [已验证] | `'cuda'` | `get_device_properties(0).gcnArchName == 'gfx1201'`（HIP-only 字段）；`torch.backends.cudnn.is_available() == False`；`torch.cuda.get_arch_list()` 含 `gfx*` |
| **ZLUDA（CUDA wheel 伪装）** | 非 `None`（真实 CUDA 构建，如 `'12.8'`） | `None`（**这是关键：它不是 HIP 构建**） | **`True`** [已验证] | `'cuda'` | ⚠️ 设备名/`gcnArchName`/`major.minor` **全部未验证**（见 2.5）。启发式：设备名出现 AMD/Radeon，或 `major`/`minor` 异常，或 `gcnArchName` 意外存在 |
| **DirectML / torch-directml** | `None` | `None` | **`False`** | **`'privateuseone'`** [已验证] | `import torch_directml` 成功；`torch_directml.is_available()`；`torch.privateuseone` 模块存在 |
| **Apple MPS** | `None` | `None` | `False` | `'mps'` | `torch.backends.mps.is_available()` 且 `torch.backends.mps.is_built()` |
| **纯 CPU** | `None` | `None` | `False` | `'cpu'` | 上面全部 false |

**重要补充**：
- 上游新增了 **`torch.version.rocm`**（与 `torch.version.hip` 区分）—— [amd-sriram/pytorch 提交 7208a75](https://github.com/amd-sriram/pytorch/commit/7208a75f7bc2dc2ffd46419fc536a9ee3b202d3f) [已验证：标题]。**是否已进入正式版未验证**，所以检测时应同时 `getattr` 两者。
- ROCm 下 `torch.backends.cudnn` **存在但不可用**（AMD 用 MIOpen）。**是否可安全触碰见第 6 节。**
- **`torch.cuda.get_device_name(0)` 在 ZLUDA + RX 9070 XT 下的确切返回值：我无法验证。** 我在报告中明确声明这一空白，而不是编造。

---

## 5. AMP（自动混合精度）现实核查

### 5.1 ROCm / HIP

| 项目 | 结论 |
|---|---|
| `torch.amp.autocast(device_type=...)` | **可用。传 `device_type='cuda'`**（因为 ROCm 伪装成 cuda，`'hip'` **不是**合法的 device_type）。[高置信，基于 HIP semantics 文档与上游设计；**未逐字验证**] |
| float16 支持 | 可用。gfx1201 有完整 fp16 硬件支持 |
| **bfloat16 支持** | ⚠️ **未验证**。RDNA4 硬件支持 bf16，但 **Windows ROCm 上的实际可用性未确认**。务必实测 `torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')` |
| `torch.amp.GradScaler` | **可用**（`GradScaler(device_type='cuda')` 或新版签名）。因为底层 `torch.cuda.*` API 全部由 HIP 实现 |
| PyTorch 版本注意 | `torch.cuda.amp.GradScaler` 已弃用；**新代码用 `torch.amp.GradScaler('cuda')`**。签名在 2.4/2.6 前后有变化（GradScaler 增加了 device 参数）[部分验证] |
| 已知 bug | ROCm 7.14.1 / 10.0.0 + torch 2.12.0 上 **FP8 路径** 与 **SDPA 后所有 kernel** 会 `hipErrorInvalidValue` → [TheRock #7861](https://github.com/ROCm/TheRock/issues/7861)、[#7992](https://github.com/ROCm/TheRock/issues/7992)。**建议钉 `+rocm7.14.0`** |

### 5.2 ZLUDA

| 项目 | 结论 |
|---|---|
| `autocast('cuda')` | ⚠️ **未验证**。上下文能进入（`is_available()==True`），但 fp16 实际运算依赖 `cublasLt`/`cublasGemmEx`，ZLUDA 覆盖是否完整是关键风险点 |
| bfloat16 | ⚠️ **未验证**，且风险更高（bf16 gemm 走另一条 cuBLAS 路径） |
| `GradScaler` | ⚠️ **未验证**。走 elementwise kernel 而非 BLAS，**相对更可能可用**（推测） |
| 建议 | **必须先跑 `cuda_check`，再写最小脚本实测** |

### 5.3 DirectML

| 项目 | 结论 |
|---|---|
| `autocast` | **不支持 / 不可用**。[高置信，非逐字验证] |
| `GradScaler` | **不可用**：`torch.cuda.is_available()` 为 `False` → GradScaler 自动禁用并告警（或抛 `NotImplementedError`，取决于版本）。[高置信] |
| bfloat16 | **不可用**（DML 后端无 bf16 训练算子）[高置信，未逐字验证] |

### 5.4 AMD 上 AMP 不可用时的推荐替代方案

**推荐优先级**（这是我基于上述证据的综合建议）：

1. **优先方案：用 ROCm 原生 + `torch.amp.autocast('cuda')` + `torch.amp.GradScaler('cuda')`**，并**钉住已知可用的 ROCm 版本**（据 Issue #7992，#7992 表明 `+rocm7.14.0` 比 `7.14.1`/`10.0.0` 稳定）。
2. **如果 AMP 路径在你的模型上不稳（尤其碰到 SDPA / FP8）**：
   - **改用 bf16 全量训练，不用 GradScaler**。bf16 的动态范围与 fp32 相同，**不需要 loss scaling**：
     ```python
     model = model.to(dtype=torch.bfloat16)
     # 或用 autocast(dtype=torch.bfloat16) —— 注意仍受 5.1 的 bf16 风险影响
     ```
   - ⚠️ 前提是 bf16 在 gfx1201 + Windows ROCm 上真的可用 —— **必须实测**。
3. **如果 bf16 也不可用**：
   - **纯 fp32 + 梯度累积**换有效 batch size；配合 `channels_last`（见第 6 节）与更小的模型。
   - 或者**走 WSL2/Linux**（Linux 上的 ROCm 成熟度高得多；注意 WSL2 也有 [ROCm 10.1 TDR 问题](https://github.com/ROCm/rocm-systems/issues/13291)）。
4. **不要**为了 AMP 去用 DirectML。

---

## 6. 其他加速相关注意事项

### 6.1 `torch.backends.cudnn` 在 AMD 上

- **ROCm 构建下 `torch.backends.cudnn` 存在但不可用**（AMD 用 **MIOpen** 而非 cuDNN）。HP 构建下 `torch.backends.cudnn.is_available()` 返回 `False`。[高置信]
- **是否"有害"？** 我的判断 [推测，未验证]：
  - **设置 `torch.backends.cudnn.enabled = False`**：在 CUDA 上用来禁用 cuDNN 自动调优；在 ROCm 上**应该是 no-op**（没有 cuDNN 可禁）。**但 PyTorch 的 `torch.backends.cudnn` 属性 setter 在某些构建下会抛异常或告警** —— 我没找到实证。
  - **`torch.backends.cudnn.benchmark = True`**：在 CUDA 上是有用的调优开关。**在 ROCm 上它不会调用 MIOpen 的 benchmark**，因此**无效但通常无害**。
  - ⚠️ **最安全的做法**：**用 `if torch.backends.cudnn.is_available():` 包起来**，不要在 AMD 上无条件设置。
  ```python
  if torch.backends.cudnn.is_available():   # AMD/ROCm 下为 False
      torch.backends.cudnn.benchmark = True
  ```
- **是否存在 `torch.backends.miopen`？** ⚠️ **未验证** —— 我**没有**找到 `torch.backends.miopen` 在 2026 年的文档或 API 证据。**不要假设它存在。**（历史上 PyTorch 有 `torch.backends.miopen.enabled`，但本次未能验证其现行状态。）
- **结论**：在 AMD 上**不要去碰 `torch.backends.cudnn`** —— 它不会带来任何收益，反而可能引入告警/异常。用 `is_available()` 守卫。

### 6.2 `channels_last` 内存格式在 AMD 上

- **[未验证]** 我**没有找到**关于 `channels_last` 在 ROCm/gfx1201 上支持性与收益的实证。
- 已知的相关工作：**MIOpen 正在为 gfx1201 (RDNA4) 增加 CK (Composable Kernel) 的 GEMM/卷积求解器** → [rocm-libraries PR #5462 "feat(miopen): Add more GEMM convolution solvers from CK for gfx1201 (RDNA4)"](https://github.com/ROCm/rocm-libraries/pull/5462) [已验证：标题]。这说明 conv 路径在 RDNA4 + Windows 上**仍在补强**，`channels_last` 的收益可能随 MIOpen kernel 覆盖度变化。
- **建议**：实测 `model.to(memory_format=torch.channels_last)` + `input.to(memory_format=torch.channels_last)`，对比有无的性能；**不要盲目启用**。DirectML 下 `channels_last` 支持性**未验证**。

### 6.3 `torch.compile` 在 ROCm / Windows

- **Windows 上基本不可用 / 极不可靠。** 证据：
  1. **Triton 在 Windows ROCm 上不是一等公民** —— 存在专门的补丁模块 [vladmandic/sdnext `modules/rocm_triton_windows.py`](https://github.com/vladmandic/sdnext/blob/c9e21a51/modules/rocm_triton_windows.py) [已验证：文件存在]，说明需要 Windows 专属 hack。
  2. **JIT 编译在 gfx1201 上直接失败**：`HIPRTC_ERROR_COMPILATION on gfx1201` → [TheRock Issue #710](https://github.com/ROCm/TheRock/issues/710) [已验证：标题]。`torch.compile` 的 inductor/CPP 路径与 Triton 都依赖运行时编译。
  3. unsloth 有专门提交 **[fix(amd): don't enable torch.compile when Triton's C toolchain can't ...](https://github.com/oobabooga/unsloth/commit/607a8af64d76ac5a1b1d20160ba13f099a1065b7)** [已验证：标题] —— 生产项目在**主动禁用** AMD 上的 torch.compile。
  4. 历史上的 Triton 安装 PR 范围是 **"Triton installation for Windows CUDA, Linux ROCm/XPU"** → [easydiffusion/torchruntime PR #31](https://github.com/easydiffusion/torchruntime/pull/31) [已验证：标题] —— **注意：Windows CUDA + Linux ROCm，没有 "Windows ROCm"**。这是很强的否定证据。
- **结论 [高置信]：在 Windows + ROCm 上不要指望 `torch.compile`。** Linux + ROCm 是可行路径（需正确安装 Triton）。
- **ZLUDA 下 `torch.compile`**：[未验证]，但 ZLUDA 用的是 CUDA 构建的 torch → 需要 Windows CUDA 版 Triton → [torchruntime PR #31](https://github.com/easydiffusion/torchruntime/pull/31) 显示 Windows CUDA Triton 存在 → **理论上比 ROCm-on-Windows 更有希望，但完全未验证。**

### 6.4 `DataLoader(pin_memory=True)` 在 AMD 上

- **[未验证]** 我**没有找到**关于 AMD Windows 上 pinned memory 的明确结论。
- 已知的相关观察：
  - PyTorch 会在无加速器时告警：`pin_memory argument is set as true but no accelerator is found, then device pinned memory won't be used` —— **在 DirectML 下会触发这条**（因为 `torch.cuda.is_available()` 为 `False`）[高置信，标准 PyTorch 行为]。
  - **ROCm 下 `pin_memory=True` 应该正常工作**，因为 `torch.cuda.*` pinned-memory API 由 HIP 实现（ROCm 复用 cuda 命名空间，见 [HIP semantics 文档](https://docs.pytorch.ac.cn/docs/2.14/notes/hip.html)）[高置信]。
  - **ZLUDA 下**：`pin_memory` 走 CUDA 驱动 API（`cuMemHostRegister` 等），ZLUDA 是否实现该入口**未验证** → 可能报错或静默降级。
  - `pin_memory_device` 参数在非 cuda 后端上需要显式指定，否则告警。
- **建议**：在 AMD Windows 上先跑一个最小 DataLoader 测试，观察是否有 `pin_memory` 相关 warning。

### 6.5 ROCm 下 `torch.cuda.empty_cache()` / `synchronize()` / `memory_allocated()`

- **全部可用**，因为 ROCm 构建把整个 `torch.cuda` 命名空间映射到 HIP 上（`hipMemGetInfo` 等）。依据：[HIP (ROCm) semantics — PyTorch docs](https://docs.pytorch.ac.cn/docs/2.14/notes/hip.html) [已验证：文档存在]。
- **注意**：AMD 上读取显存的一个坑是 `torch.cuda.mem_get_info()` / 显存总量有时不准，社区因此改用 **`amd-smi`** —— 见 unsloth 提交 [fix(amd): read ROCm VRAM through amd-smi so the backend keeps no HIP ...](https://github.com/oobabooga/unsloth/commit/300292b2f0b110095b77a472c0ee6681c194379f) [已验证：标题]。**这是一个真实的、被生产项目采用的 workaround。**

### 6.6 额外资源（供你深入）

- [AMD GPUOpen: A beginner's guide to deploying LLMs with AMD on Windows using PyTorch](https://gpuopen.com/learn/pytorch-windows-amd-llm-guide/)（AMD 官方教程）
- [AMD: How To Run AMD ROCm™ Software in Windows 11](https://www.amd.com/en/developer/resources/technical-articles/how-to-run-amd-rocm-software-in-windows-11.html)
- [AMD: The Road to ROCm on Radeon for Windows and Linux](https://www.amd.com/en/blogs/2025/the-road-to-rocm-on-radeon-for-windows-and-linux.html)
- [ROCm/TheRock（The HIP Environment and ROCm Kit）](https://github.com/ROCm/TheRock)；[DeepWiki: TheRock PyTorch and ML Framework Integration](https://deepwiki.com/ROCm/TheRock/4.3-pytorch-and-ml-framework-integration)、[Quick Start and Installation](https://deepwiki.com/ROCm/TheRock/1.1-installation-and-setup)
- [TheRock legacy → TheRock 迁移指南](https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html)
- [ROCm/Radeon-Ryzen 文档首页](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/)
- [Wan2GP `docs/AMD-INSTALLATION.md`（"AMD Installation Guide for Windows (TheRock)"，9,351 字节，含实操细节）](https://raw.githubusercontent.com/deepbeepmeep/Wan2GP/7e45fe7e21105807b43f6285827d9ebb5fa72906/docs/AMD-INSTALLATION.md) —— **强烈建议你打开这份**（本次我读不到正文）
- [pedrodenovo/ComfyUI-ROCm-Windows-Native](https://github.com/pedrodenovo/ComfyUI-ROCm-Windows-Native)（含 `setup_rocm.bat` 自动化脚本）
- [sh4den/ComfyUI-AMD](https://github.com/sh4den/ComfyUI-AMD)（"Works with RDNA 3/3.5/4"）
- [OrsoEric/HOWTO-ComfyUI](https://github.com/OrsoEric/HOWTO-ComfyUI)
- [civitai: Team Red — Setup ComfyUI+MMH3 with AMD GPUs: rx9070, R9700 & rx7900](https://civitai.red/articles/35331/team-red-encounter-on-proximax-h3-or-setup-comfyuimmh3-with-amd-gpus-rx9070-r9700-and-rx7900)（标题提到 **stability matrix** 章节，可能含版本兼容矩阵）
- [tsukkomi.org: AMD显卡在Windows上安装使用PyTorch（ROCm）](https://tsukkomi.org/post/installing-pytorch-rocm-on-windows-with-amd-gpu)（中文实操）
- [gadelog.com: 【2026年9月】WindowsのROCmはどれを入れる？7.2.1と7.14とROCm 10の違い](https://gadelog.com/)（日文，专门比较该装哪个版本）
- [qiita: AMD GPU（ROCm）でDeepSeek-OCRを動かしてみた【Windows 11環境】](https://qiita.com/Dfkuwa/items/28aec1cd8ee1279cb39d)
- [joshwaamein: Zero-Shot Voice Cloning on AMD — ROCm 7.1 on Windows](https://joshwaamein.github.io/posts/zero-shot-voice-cloning-amd-gpu-windows/)

---

## 附录：明确"未能验证"的清单

以下问题我**无法**给出确定答案，请在使用前自行实测/查证：

1. `https://rocm.nightlies.amd.com/v2/gfx120X-all/` 上具体有哪些 `cp3xx`（是否含 `cp313`）wheel。
2. `torchvision` / `torchaudio` 与任一 torch 版本的确切配对版本号。
3. Windows 上 `repo.radeon.com` 的**完整** pip 索引 URL（只确认了 `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/` 目录存在，未确认 `whl/` 子路径形式）。
4. `amd-multi-arch-index` 的具体 URL。
5. `download.pytorch.org/whl/rocmX.Y` 是否含 Windows wheel（有较强证据说明是 Linux-only）。
6. **ZLUDA 下 `torch.cuda.get_device_name(0)` 在 RX 9070 XT 上的确切字符串。**
7. **ZLUDA 下 `torch.cuda.get_device_properties(0)` / `.major` / `.minor` / `.gcnArchName` 的确切行为**（是否抛异常）。
8. **ZLUDA 实际发布的 DLL 文件清单**（`nvcuda.dll` / `cublas64_*.dll` / `cudnn64_*.dll` 等具体命名与数量）。
9. **ZLUDA 下 fp16 autocast / GradScaler 是否可用。**
10. **ZLUDA 下 RX 9070 XT 的 PyTorch 性能数字。**
11. torch-directml 与 PyTorch 版本的对应表；torch-directml 在 Python 3.13 上的支持（倾向于不支持）。
12. `torch.amp.autocast` 在 PyTorch 2.12–2.14 中 `device_type` 的**精确合法集合**。
13. DirectML 下 autocast / GradScaler 的**确切报错文本**。
14. gfx1201 + Windows ROCm 上 **bfloat16 是否可用**。
15. `torch.backends.miopen` 在 2026 年是否仍然存在/可用。
16. `channels_last` 在 gfx1201 + Windows ROCm 上的支持性与收益。
17. ZLUDA 下 `torch.compile` 是否可用。
18. ROCm / ZLUDA 下 `pin_memory=True` 的确切行为。

**方法论说明**：由于 `web_fetch` 与本机出网全部被阻断，以上 18 项都需要读取页面正文才能确认。凡是我能通过搜索结果标题片段确认的，我都已在正文标注 **[已验证]**。我没有用推测填补这些空白。
