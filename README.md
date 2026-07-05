# REACT 2026 Submission-V2

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

## online训练日志


| 方法 | 训练日志 |
|------|----------|
| MAFRG | `outputs/.../260624144915_5m1kmoy2/main.log` |
| PMAFRG | `outputs/.../260705102821_uyh7q1p4/main.log` |



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

**在线预训练权重**（TransformerDenoiser + EEGPredictionHead）已上传至夸克网盘：

> https://pan.quark.cn/s/cf449d1de0d4


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
    resume_id=260701154446_uikda4ar \
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
    resume_id=260705102821_uyh7q1p4 \
    trainer.generic.eval_eeg=true \
    trainer.main_model.args.personal_condition_mode=personality_only
```

---


