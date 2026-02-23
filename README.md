# PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models

## Installation and Data Preparation
```bash
git clone https://github.com/HaroldYUANHAOSU/PointAlign.git
cd PointAlign
```

For detailed setup instructions, please refer to [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D).

## Pretrained Weights

Download the pretrained weights from [Baidu Pan](https://pan.baidu.com/s/1QQ65gGhagQmrVjt96GDY3A?pwd=x7n5) (extraction code: `x7n5`) and place them under `./params_weight/` before training:
```
./params_weight/
└── <weight_file>
```

## Training

Modify `finetune.yaml` to set your data paths, then run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path finetune.yaml
```

## Evaluation

Please follow the evaluation instructions from [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D).

## Acknowledgement

This work builds upon [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D). We thank the authors for their excellent codebase.
