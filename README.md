# LightCollections⚡️: modular-pytorch-lightning-collections
[WIP] modular-pytorch-lightning, **WARNING: The repository is currently under development, and is unstable.**

What is `modular-pytorch-Lightning-Collections⚡`(LightCollections⚡️) for?
- LightCollection aims at extending the features of `pytorch-lightning`. We aim to provide training procedures of various subtasks in Computer Vision with a collection of `LightningModule` and utilities for easily using model architecture, metrics, dataset, and training algorithms.
- This repository is designed to utilize many amazing and robust and open-source projects such as `timm`, `torchmetrics`, and more. Currently, the following frameworks are integrated into `LightCollections` and can be easily applied through the config files:
  - `torchvision.models` for models, `torchvision.transforms` for transforms, optimizers and learning rate schedules from `pytorch`.
  - Network backbones from `timm`.
  - `inagenet21k` [pretrained weights](https://github.com/Alibaba-MIIL/ImageNet21K) and feature to load model weights from url / `.pth` file.
  - `torch-ema` for saving EMA weights.
  - `torchmetrics` for logging metrics. For example, see `configs/vision/training/resnet-cifar10.yaml`
  - WIP & future TODO:
    - Data augmentation from `albumentations`
    - TTA and TTAch


## How to run experiments
1. Run experiments using `train.py`

- CIFAR10 image classification with ResNet18``.
```
!python train.py --config configs/vision/training/resnet-cifar10.yaml configs/vision/models/resnet/resnet18-custom.yaml configs/vision/data/cifar10.yaml configs/utils/wandb.yaml configs/utils/train.yaml
```

2. Use the `Experiment` class to run complex experiments for research, hyperparameter sweep, ...:

For example, please have a look at the `study/dataset_size_experiment.py`.

### How does config files work?

Training involves many configs. `LightCollections` implements a cascading config system where we use multiple layers of config
files to define differnt parts of the experiment. For example, in the CIFAR10 example above, we use 5 config files.
```
configs/vision/training/resnet-cifar10.yaml
configs/vision/models/resnet/resnet18-custom.yaml
configs/vision/data/cifar10.yaml
configs/utils/wandb.yaml
configs/utils/train.yaml
```
here, if we want to log to `TensorBoard` instead of `wandb`, you may replace
```
configs/utils/wandb.yaml
-> configs/utils/tensorboard.yaml
```
these cascading config files are baked at the start of `train.py`, where configs are overriden in inverse order. These baked config files are logged under `configs/logs/{experiment_name}.(yaml/pkl/json)` for logging purpose.

# List of techniques implemented and how to use

## Data Augmentation

### RandomResizeCrop(ImageNet augmentation)

- Paper: https://arxiv.org/abs/1409.4842
- Note: Common data augmentation strategy for ImageNet using RandomResizedCrop.
- Refer to: `configs/vision/data/imagenet.yaml`

```
transform: [
    [
      "trn",
      [
        {
          "name": "RandomResizedCropAndInterpolation",
          "args":
            {
              "scale": [0.08, 1.0],
              "ratio": [0.75, 1.3333],
              "interpolation": "random",
            },
        },
        {
          "name": "TorchTransforms",
          "args": { "NAME": "RandomHorizontalFlip" },
        },
        # DATA AUGMENTATION(rand augment, auto augment, ...)
        {
          "name": "TorchTransforms",
          "args":
            {
              "NAME": "ColorJitter",
              "ARGS": { "brightness": 0.4, "contrast": 0.4, "saturation": 0.4 },
            },
        },
      ],
    ],
    [
      "val",
      [
        {
          "name": "Resize",
          "args": { "img_size": 224, "interpolation": "bilinear" },
        },
        {
          "name": "TorchTransforms",
          "args": { "NAME": "CenterCrop", "ARGS": { "size": 224 } },
        },
      ],
    ],
    [
      "trn,val",
      [
        { "name": "ToTensor", "args": {} },
        {
          "name": "Normalize",
          "args":
            { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] },
        },
      ],
    ],
  ]
```

### RandAugment

- Paper: https://arxiv.org/abs/1909.13719
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/randaugment.yaml`

```
        {
          "name": "TorchTransforms",
          "args":
            { 
              "NAME": "RandAugment",
              "ARGS": { "num_ops": 2, "magnitude": 9 } 
            },
        },
...
```
Refer to: `configs/algorithms/data_augmentation/randaugment.yaml`

### TrivialAugmentation
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```
        {
          "name": "TorchTransforms",
          "args":
            {
              "NAME": "TrivialAugmentWide",
              "ARGS": { "num_magnitude_bins": 31 },
            },
        },
...
```

### Mixup
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### CutMix
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### CutOut
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

## Regularization

### Label smoothing
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### Weight decay
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### DropOut(classification)
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### R-Drop(classification)
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

## Loss

### Knowledge Distillation soft loss
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### Sharpness-aware minimization(SAM)
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### PolyLoss
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```


## Training

### Gradient Clipping
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### Stocahstic Weight Averaging(SWA)
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

## Pre-training

### Fine-tuning from checkpoint
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

## Learning rate schedule

### Cosine learning rate decay
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

### WarmUp
- Paper: https://arxiv.org/abs/2103.10158
- Note: Commonly used data augmentation strategy for image classification.
- Refer to: `configs/algorithms/data_augmentation/trivialaugment.yaml`
```

```

# Using more from other libraries

## Network architecture

### timm 

### torchvision.models

## 

### torchmetrics

### 


# Overview

## LightningModules

### Classifi
- If not implemented yet, you may take an instance of `main.py: Experiment` and override any part of it.
- Training procedure (`LightningModule`): List of available training procedures are listed in `lightning/trainers.py`
- Model architectures:
  - Backbone models implemented in `torchvision.models` can be used.
  - Backbone models implemented in `timm` can be used.
  - Although we highly recommend using `timm`, as it is throughly evaluated and managed, custom implementations of some architectures are listed in `models/backbone/__init__.py`.
## Data
Currently `torchvision` datasets are supported by `Experiment`, however you could use `torchvision.datasets.ImageFolder` to load from custom dataset.

### Transformations(data augmentation): Transforms must be listed in one in [`data/transforms/vision/__init__.py`]


## Utils

### Optimizers

####

### Learning-rate schedules

#### Warmup

### Metrics

- Other features
  - Optimizers
  - Metrics / loss

### `debug` flag


## About me

- Contact: sieunpark77@gmail.com / sp380@student.london.ac.uk / +82) 10-6651-6432
