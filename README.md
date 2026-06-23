Cross-Layer Adaptive Knowledge Distillation for Lightweight Metal Surface Scratch Detection
=============================================================================
This paper has been submitted to [The Visual Computer].

Authors: Bin Xu, Jianhong Huang, Yuyang Du, Tong Zhou, Jiangwen Tang, Hongyu Li, Qishi Hu, Bo Zhang, Kai Liu

Repository: https://github.com/asdu1x/Lightweight-metal-scratch-detector 

This repository contains code for our lightweight metal scratch detector. It is intended to provide a demonstration of the model described in the paper.

-------
Environment
-------
The code has been tested on:
  - Windows 10
  - Python 3.7+
  - PyTorch 1.6.0+ (CUDA 10.1 or later recommended)
  - A GPU with at least 6 GB memory is recommended for smooth inference. CPU-only inference is also supported but will be slower.

-------
Quick Start
-------
1. Install dependencies: pip install -r requirements.txt

2. Install detectron2 [https://github.com/facebookresearch/detectron2]    

3. Change the configuration and dataset paths in train.py, then run it to get the models.

4. Change the configuration, model, and dataset paths in val.py, then run it to get the evaluation results.
   
5. Change the image and model paths in visualization.py, then run it to get the visualization results.

---
Data Availability
---
The data that support the findings of this study are available from the corresponding author upon reasonable request.

---
Contact
---
Bo Zhang
bo_zh@scu.edu.cn
