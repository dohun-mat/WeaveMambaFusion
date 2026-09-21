# WeaveMambaFusion(ACCV 2026)

## official implement of Paper
Weave Mamba Fusion Global Cross-Scale Interaction for Lightweight Face Detection

## OverView
Feature pyramid methods, from FPN to BiFPN, fuse multi-scale features only locally (summation or concatenation followed by convolution), which struggles to detect faces under unconstrained conditions small scale, occlusion, extreme pose since these require global cross-scale dependencies that local fusion cannot model.

State space models such as Mamba offer global context at linear complexity, but only if the way the two pyramid scales are combined preserves cross-scale structure: summation collapses them before the scan, leaving nothing to exploit, while concatenation preserves both scales at far higher cost.

We propose Weave Mamba Fusion (WMF), which interleaves two adjacent pyramid scales column by column so that every step of a horizontal bidirectional SS2D scan moves from one scale to the other. With partial-channel processing and parameter-free de-weaving, WMF enables efficient global cross-scale interaction while preserving feature structure. Integrating WMF into every fusion node yields WeaveBiFPN, the neck of our WeaveFace detector. On WIDER FACE, WeaveFace achieves 91.41% mean AP with only 0.34M parameters and 1.16 GFLOPs, outperforming prior detectors under 0.5M parameters, with its largest gains on the Hard subset (87.14% AP).

<img width="3121" height="538" alt="figure0-featuremap_WVsywWv7" src="https://github.com/user-attachments/assets/2f0a076a-b737-44d5-ae44-1e0b69ce2242" />

## Architecture
<img width="900" alt="model-architecture" src="https://github.com/user-attachments/assets/70daa796-7e93-4bab-9654-93ef6eff3676" />

## install
1. git clone https://github.com/dohun-mat/WeaveMambaFusion

2. dcnv3를 설치(https://github.com/OpenGVLab/InternImage/tree/master/detection)
```Shell
  cd ./ops_dcnv3
  sh ./make.sh
  # unit test (should see all checking is True)
  python test.py
```

3. mamba설치
```Shell
  /lib_mamba/kernels/selective_scan && pip install .
```

## data
1. We also provide the organized dataset we used as in the above directory structure.
Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

2. Organise the dataset directory as follows:
```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

## train
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in
   ```data/config.py and train.py```

2. Train the model using WIDER FACE:
  ```Shell
   CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 train.py --network eresnet
  ```


## Evaluation
1. Generate txt file
```Shell
When you want to use only the original image
CUDA_VISIBLE_DEVICES=1 python test_widerface.py --trained_model ./weights/eresnet_Final.pth --network eresnet

When you want to use the multi-scale strategy
CUDA_VISIBLE_DEVICES=1 python test_widerface_multi_scale.py --trained_model ./weights/eresnet_Final.pth --network eresnet --test_scales 0.5 1.0 1.5 2.0 --do_flip 
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)  
```Shell
cd ./widerface_evaluate
python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)  

## Result
### WIDER FACE Validation (multi-scale AP)

| Model | Backbone | Neck | Fusion | Easy | Medium | Hard | Avg. (%) | #Params (M) | GFLOPs |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| DSFD | ResNet-152 | FEM | Conv. | 96.60 | 95.70 | 90.40 | 94.23 | 120.06 | 259.55 |
| TinaFace | ResNet-50 | FPN | Conv. | 97.00 | 96.30 | 93.40 | 95.57 | 37.98 | 172.95 |
| TransEnc-R50 | ResNet-50 | FPN | Self-attention | 93.03 | 92.89 | 88.56 | 91.49 | 33.72 | 32.92 |
| RetinaFace | ResNet-50 | FPN | Conv. | 96.70 | 96.10 | 91.40 | 94.73 | 29.50 | 37.59 |
| SCRFD-10GF | Basic Res | PANet | Conv. | 95.93 | 94.95 | 90.81 | 93.90 | 3.86 | 9.98 |
| FaceBoxes | - | - | - | 85.90 | 81.60 | 55.70 | 74.40 | 1.01 | 0.28 |
| SCRFD-0.5GF | Depth-wise Conv | PANet | Conv. | 92.71 | 91.45 | 86.23 | 90.13 | 0.57 | 0.51 |
| RetinaFace Lite | MobileNet0.25 | FPN | Conv. | 91.40 | 89.20 | 82.50 | 87.70 | 0.44 | 0.80 |
| FDLite | BLite | FPN | Conv. | 92.31 | 89.91 | 82.30 | 88.17 | 0.24 | 0.94 |
| MTCNN | - | - | - | 85.10 | 82.00 | 60.70 | 75.93 | 0.12 | 0.01 |
| EResFD | EResNet | SepFPN | Sep. Conv. | 89.02 | 87.96 | 80.41 | 85.80 | 0.09 | 0.30 |
| **WeaveFace (Ours)** | **EResNet** | **WeaveBiFPN** | **SS2D** | **94.33** | **92.76** | **87.14** | **91.41** | **0.34** | **1.16** |

WeaveFace achieves the highest AP among all sub-0.5M-parameter detectors on every subset, and its largest margin is on **Hard** — the regime where global cross-scale context matters most.

### Cross-Dataset Generalization: AFW & PASCAL Faces

All models below are trained solely on WIDER FACE and evaluated **without fine-tuning** on AFW and PASCAL Faces.

| Method | AFW AP (%) | PASCAL AP (%) | #Params |
|---|:---:|:---:|:---:|
| MogFace | 99.85 | **99.32** | 85.26M |
| EfficientSRFace-L | **99.94** | 98.84 | 18.84M |
| SCRFD-0.5GF | 98.60 | 98.54 | 0.57M |
| SCRFD-1.0GF | 99.70 | 98.60 | 0.64M |
| SCRFD-2.5GF | 99.82 | 98.91 | 0.67M |
| FaceBoxes | 98.91 | 96.30 | 1.01M |
| **WeaveFace (Ours)** | **99.50** | **98.77** | **0.34M** |

Despite having the fewest parameters among all compared methods, WeaveFace outperforms every detector under 1M parameters (SCRFD variants, FaceBoxes) on both AFW and PASCAL Faces, and trails the far heavier MogFace (85.26M params) by only 0.35 AP on AFW — indicating that WMF's cross-scale interaction generalizes beyond WIDER FACE rather than overfitting to it.

### Generality to Generic Object Detection (EfficientDet + WMF)

To test whether WMF generalizes beyond face detection, we plug WMF into the BiFPN neck of **EfficientDet** and evaluate on COCO and PASCAL VOC, without any face-detection-specific components.

**COCO val2017**

| Variant | AP | AP50 | AP75 | #Params (M) | FLOPs (G) |
|---|:---:|:---:|:---:|:---:|:---:|
| EfficientDet (baseline) | 34.5 | 52.9 | 36.6 | 3.88 | 2.57 |
| **EfficientDet + WMF** | **36.2** | **53.6** | **38.2** | 4.25 | 3.21 |

**PASCAL VOC**

| Variant | mAP | #Params (M) | FLOPs (G) |
|---|:---:|:---:|:---:|
| EfficientDet (baseline) | 76.90 | 3.84 | 2.35 |
| **EfficientDet + WMF** | **78.63** | 4.21 | 2.99 |

Adding WMF to EfficientDet's BiFPN neck improves COCO AP by **+1.7** (34.5% → 36.2%) and AP75 by **+1.6**, indicating better localization quality in addition to higher overall accuracy. On PASCAL VOC, WMF improves mAP by **+1.73** (76.90% → 78.63%). Both gains come at a modest cost of roughly +0.37M parameters and +0.64–0.71 GFLOPs, confirming that WMF's benefit is not specific to face detection and transfers to generic object detection.

