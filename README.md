# WeaveMambaFusion

## official implement of Paper
Weave Mamba Fusion Global Cross-Scale Interaction for Lightweight Face Detection

## OverView
<img width="3121" height="538" alt="figure0-featuremap_WVsywWv7" src="https://github.com/user-attachments/assets/2f0a076a-b737-44d5-ae44-1e0b69ce2242" />

## Architecture
<img width="900" alt="model-architecture" src="https://github.com/user-attachments/assets/70daa796-7e93-4bab-9654-93ef6eff3676" />

## install
1. git clone https://github.com/dohun-mat/WeaveMambaFusion

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


