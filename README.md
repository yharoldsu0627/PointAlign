# PointAlign (CVPR 2026)
### [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Code](https://github.com/yharoldsu0627/PointAlign)

> PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models\
> Yuanhao Su, Shaofeng Zhang, Xiaosong Jia, Qi Fan\
> CVPR 2026

✨ **A step towards better 3D vision-language understanding by preserving geometric information throughout the language modeling process.**

| Model | Objaverse (I) | Objaverse (C) | ModelNet40 Avg |
|-|-|-|-|
| MiniGPT-3D | 65.00 | 68.50 | 61.24 |
| **PointAlign (Ours)** | **72.50** | **69.50** | **61.17** |

### ✅ Project Status

🎉 **Accepted to CVPR 2026!**

- [x] Release introduction & results
- [x] Release training & inference code
- [x] Upload pretrained weights

If you find PointAlign useful, please consider giving us a star ⭐.

### Introduction

<!-- 在这里插入你的 Introduction 图片，步骤见文末 -->

🔍
**Geometric Information Degradation:** Existing 3D VLMs rely solely on next-token prediction loss, causing valuable geometric cues to be discarded during training. PointAlign addresses this by explicitly supervising intermediate point cloud tokens within the LLM to preserve fine-grained 3D geometric-semantic information throughout the language modeling process.

### Overview

<!-- 在这里插入你的 Framework Overview 图片，步骤见文末 -->

PointAlign enhances 3D vision-language understanding through **feature-level alignment regularization** — a cosine similarity loss that aligns intermediate point cloud tokens in the LLM with Q-Former outputs. The alignment projector is used **only during training** and discarded at inference, introducing **zero additional inference overhead**.

### Qualitative Results

<!-- 在这里插入定性对比图/表，步骤见文末 -->

## ⚙️ Quick Start

### Installation and Data Preparation

```bash
git clone https://github.com/yharoldsu0627/PointAlign.git
cd PointAlign
```

For detailed setup instructions, please refer to [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D).

### Pretrained Weights

Download the pretrained weights from [Baidu Pan](https://pan.baidu.com/s/1QQ65gGhagQmrVjt96GDY3A?pwd=x7n5) (extraction code: `x7n5`) and place them under `./params_weight/` before training:

```
./params_weight/
└── <weight_file>
```

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path finetune.yaml
```

### Evaluation

Please follow the evaluation instructions from [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D).

## Contact

If you have any questions related to the code or the paper, feel free to email Yuanhao (`yharoldsu0627@gmail.com`).

## Acknowledgement

This project builds upon [**MiniGPT-3D**](https://github.com/TangYuan96/MiniGPT-3D). We thank the authors for their excellent codebase.

## Citation

```bibtex
@inproceedings{su2026pointalign,
  title     = {PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models},
  author    = {Su, Yuanhao and Zhang, Shaofeng and Jia, Xiaosong and Fan, Qi},
  booktitle = {CVPR},
  year      = {2026}
}
```
