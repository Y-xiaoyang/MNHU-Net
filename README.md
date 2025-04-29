# MNHU-Net: A Multi-Scale Feature Fusion and Nested Structure-Based High-Order U-Net for Infrared Small Target Detection [[📄 Paper Link]](https://ieeexplore.ieee.org/document/10979417)
### Xiaoyang Yuan, Chunling Yang, Yu Chen, Yan Zhang, IEEE Transactions on Aerospace and Electronic Systems 2025. 

# Introduction
We present a multi-scale feature fusion and nested structurebased High-order UNet (MNHU) for IRSTD. We evaluate the proposed high-order UNet-based
methods (MNHU-E, MNHU-D, and MNHU) on three public datasets (e.g. SIRST, IRSTD-1k, NUDT-SIRST), which demonstrates the effectiveness of our methods. Our main contributions are as follows:

1. A novel high-order UNet paradigm is proposed. This paradigm enhances feature representation by leveraging high-order skip connections to fuse the feature maps from adjacent layers with strong correlations.
2. Building upon the paradigm, we propose three high-order U-Net architectures (MNHU-E, MNHU-D, and MNHU) to calibrate infrared feature maps by distinguishing small targets from background textures and modeling long-range dependencies.
3. To enhance segmentation performance, a high-order interactive feature extractor integrating a residual channel-spatial attention module and a high-order fusion supervision module is incorporated.

# Usage
### 1. Data
#### Our project has the following structure:
```text
├──./dataset/
│    ├── IRSTD-1K
│    │    ├── images
│    │    │    ├── XDU0.png
│    │    │    ├── XDU1.png
│    │    │    ├── ...
│    │    ├── masks
│    │    │    ├── XDU0.png
│    │    │    ├── XDU1.png
│    │    │    ├── ...
│    │    ├── 80_20
│    │    │    ├── train.txt
│    │    │    ├── test.txt
│    ├── NUDT-SIRST
│    │    ├── images
│    │    │    ├── 000001.png
│    │    │    ├── 000002.png
│    │    │    ├── ...
│    │    ├── masks
│    │    │    ├── 000001.png
│    │    │    ├── 000002.png
│    │    │    ├── ...
│    │    ├── 80_20
│    │    │    ├── train.txt
│    │    │    ├── test.txt
│    ├── ...
│    ├── ...
│    ├── NUAA-SIRST
│    │    ├── images
│    │    │    ├── Misc_1.png
│    │    │    ├── Misc_2.png
│    │    │    ├── ...
│    │    ├── masks
│    │    │    ├── Misc_1.png
│    │    │    ├── Misc_2.png
│    │    │    ├── ...
│    │    ├── 80_20
│    │    │    ├── train.txt
│    │    │    ├── test.txt
```
### 2. Train.
```bash
python train.py 
```
### 3. Test.
```bash
python test.py 
```

# If the implementation of this repo is helpful to you, just star it！⭐⭐⭐

 
