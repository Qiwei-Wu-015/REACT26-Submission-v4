# REACT 2026 Improved Baseline — Offline & Online Facial Reaction Generation

> English | [中文](#中文)

## Overview

This repository contains an improved version of the REACT 2026 baseline, supporting **both offline (Task 1) and online (Task 2)** facial reaction generation — including generic and personalized modes. It is built upon the **PerFRDiff** architecture with the following key enhancements:

- **Flow Matching** — Replaced DDIM with a Flow Matching formulation for more efficient and higher-quality sampling.
- **Removed Redundant Prior Module** — Eliminated the diffusion prior network to reduce model complexity without sacrificing generation quality.
- **Low-Rank Decomposition (LoRA) for Personalization** — Proposed a low-rank strategy to drastically reduce GPU memory usage of the personalized hypernetwork, enabling personalization on a single 24 GB RTX 3090.
- **Stitch Encoding Module** *(Online)* — A cross-modal feature fusion encoder that integrates audio, 3DMM, emotion, and future speaker emotion predictions for richer conditioning in the online setting.
- **Speaker Future Behavior Prediction** *(Online)* — A dedicated predictor module to forecast the speaker's upcoming 10-frame emotion, enhancing online reaction coherence and anticipation.

---

## Supported Tasks

| Task | Generic | Personalized |
|------|:-------:|:------------:|
| **Offline (Task 1)** | ✅ `generic_offline/motion_diffusion` | ✅ `personalized_offline/perfrdiff_rewrite_weight` |
| **Online (Task 2)** | ✅ `generic_online/motion_diffusion` | ✅ `personalized_online/perfrdiff_rewrite_weight` |

---

## Results (Online, Bidirectional)

We compare against the official REACT 2026 baseline under the same bidirectional online setting.

| | Training Log | Test Log | FRC ↑ |
|---|:---:|:---:|:---:|
| **Baseline** | `outputs/.../260604191108_sxuzs9dh/main.log` | `outputs/.../260605092541_uadhj9dt/main.log` | 0.6028 |
| **Ours** | `outputs/.../260624144915_5m1kmoy2/main.log` | `outputs/.../260624223823_iooc7ldj/main.log` | **0.6546** |

**→ +5.2 points improvement in FRC.**

---

## Hardware Constraints

This codebase fully supports **both offline and online tasks** (generic + personalized). However, we only have access to a single **RTX 3090 (24 GB)**, which is sufficient for all online experiments but insufficient for offline training/testing (which requires processing full-length clips with larger memory footprints). All reported results and provided checkpoints are therefore for the online setting only. The offline code paths are complete and ready to run on GPUs with larger memory. If any bugs are found, please contact us for immediate fixes.

---

## Low-Rank Personalization

Our low-rank decomposition enables personalized hypernetwork training on a 24 GB GPU. To disable it and match the baseline exactly:

```
# In this file, change:
configs/personalized_online/trainer/perfrdiff_rewrite_weight.yaml
configs/personalized_offline/trainer/perfrdiff_rewrite_weight.yaml

    lora_rank: 4   →   lora_rank: 0
```

All other hyperparameters are identical to the baseline.

---

## Pretrained Weights

**Generic online pretrained weights** (TransformerDenoiser + EEGPredictionHead) are available via Quark Drive:

> https://pan.quark.cn/s/cf449d1de0d4

Personalized low-rank weights are currently training and will be provided at the same link when ready.

---

## Usage

### Generic Online Training

```bash
nohup python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=8 \
    trainer.generic.bidirectional=true \
    stage=fit \
    data_dir=/data2/REACT2025-NEW \
    trainer.model.diff_model.eeg_head.enabled=true \
    trainer.generic.train_eeg_head_only=false \
    > train.log 2>&1 &
```

### Generic Online Testing

```bash
nohup python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=1 \
    trainer.generic.bidirectional=true \
    stage=test \
    data_dir=/data2/REACT2025-NEW \
    resume_id=260624144915_5m1kmoy2 \
    trainer.generic.eval_eeg=true \
    trainer.model.diff_model.eeg_head.enabled=true \
    > test.log 2>&1 &
```

### Personalized Online Training

```bash
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    stage=fit \
    data_dir=/data2/REACT2025-NEW \
    trainer.generic.train_eeg=true \
    trainer.generic.train_eeg_head_only=false \
    trainer.main_model.args.personal_condition_mode=personality_only
```

### Personalized Online Testing

```bash
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    trainer.batch_size=1 \
    stage=test \
    data_dir=/data2/REACT2025-NEW \
    resume_id=260625124536_5xn3nrjr \
    trainer.generic.eval_eeg=true \
    trainer.main_model.args.personal_condition_mode=personality_only
```

---

## Environment Setup

```bash
pip install -r requirements.txt
```

Required pretrained models should be placed under `pretrained_models/` (weights) and `external/` (FaceVerse utilities). See the Quark Drive link above for downloads.

---

---

## 中文

## 概述

本仓库是 REACT 2026 的改进基线，支持**离线（Task 1）和在线（Task 2）**面部反应生成任务，含通用和个性化模式。基于 **PerFRDiff** 架构改造，主要改进如下：

- **Flow Matching 采样** — 将 DDIM 替换为 Flow Matching，采样更高效、质量更高。
- **去除冗余 Prior 模块** — 移除了 Diffusion Prior Network，降低模型复杂度，无损生成质量。
- **低秩分解（LoRA）个性化** — 提出低秩策略，大幅降低个性化超网络的显存占用，使单张 24 GB RTX 3090 即可完成个性化训练。
- **Stitch 特征融合编码模块**（在线） — 跨模态特征交互融合，整合音频、3DMM、情感及未来说话者情感预测，为在线模式提供更丰富的条件输入。
- **说话者未来行为预测**（在线） — 新增预测模块，预测说话者未来 10 帧情感，提升在线反应连贯性和预判能力。

---

## 支持的任务

| 任务 | 通用 | 个性化 |
|------|:---:|:---:|
| **离线 (Task 1)** | ✅ `generic_offline/motion_diffusion` | ✅ `personalized_offline/perfrdiff_rewrite_weight` |
| **在线 (Task 2)** | ✅ `generic_online/motion_diffusion` | ✅ `personalized_online/perfrdiff_rewrite_weight` |

---

## 实验结果（在线、双向）

在相同 bidirectional 在线设置下与官方 REACT 2026 基线对比。

| | 训练日志 | 测试日志 | FRC ↑ |
|---|:---:|:---:|:---:|
| **基线** | `outputs/.../260604191108_sxuzs9dh/main.log` | `outputs/.../260605092541_uadhj9dt/main.log` | 0.6028 |
| **我们的** | `outputs/.../260624144915_5m1kmoy2/main.log` | `outputs/.../260624223823_iooc7ldj/main.log` | **0.6546** |

**→ FRC 提升约 5.2 个点。**

---

## 硬件限制说明

本代码完整支持**离线与在线全部任务**（通用 + 个性化）。但由于我们仅有一张 **RTX 3090 (24 GB)**，在线任务可正常运行，离线训练/测试因需处理完整长序列而显存不足，无法完成。因此所有报告结果和提供的权重仅针对在线设置。离线代码路径是完整的，可以在更大显存的 GPU 上运行。若提交代码存在 bug，可第一时间联系我们修复。

---

## 低秩个性化配置

低秩分解使个性化训练可在 24 GB 显卡上运行。如需关闭该选项与基线对齐：

```
# 修改以下文件中的参数：
configs/personalized_online/trainer/perfrdiff_rewrite_weight.yaml
configs/personalized_offline/trainer/perfrdiff_rewrite_weight.yaml

    lora_rank: 4   →   lora_rank: 0
```

其余所有超参与基线完全一致。

---

## 预训练权重

**在线通用预训练权重**（TransformerDenoiser + EEGPredictionHead）已上传至夸克网盘：

> https://pan.quark.cn/s/cf449d1de0d4

个性化低秩分解权重正在训练中，训练完成后将更新至同一链接。

---

## 使用方法

### 通用在线训练

```bash
nohup python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=8 \
    trainer.generic.bidirectional=true \
    stage=fit \
    data_dir=/data2/REACT2025-NEW \
    trainer.model.diff_model.eeg_head.enabled=true \
    trainer.generic.train_eeg_head_only=false \
    > train.log 2>&1 &
```

### 通用在线测试

```bash
nohup python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=1 \
    trainer.generic.bidirectional=true \
    stage=test \
    data_dir=/data2/REACT2025-NEW \
    resume_id=260624144915_5m1kmoy2 \
    trainer.generic.eval_eeg=true \
    trainer.model.diff_model.eeg_head.enabled=true \
    > test.log 2>&1 &
```

### 个性化在线训练

```bash
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    stage=fit \
    data_dir=/data2/REACT2025-NEW \
    trainer.generic.train_eeg=true \
    trainer.generic.train_eeg_head_only=false \
    trainer.main_model.args.personal_condition_mode=personality_only
```

### 个性化在线测试

```bash
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    trainer.batch_size=1 \
    stage=test \
    data_dir=/data2/REACT2025-NEW \
    resume_id=260625124536_5xn3nrjr \
    trainer.generic.eval_eeg=true \
    trainer.main_model.args.personal_condition_mode=personality_only
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

所需预训练模型请放入 `pretrained_models/`，FaceVerse 工具文件放入 `external/`。详见上方夸克网盘链接。


