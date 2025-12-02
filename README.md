# CTCMrailwaySeg: Railway segmentation of video captures with a transfer of contextual information with unsupervised evaluation

## Introduction
This repository contains the implementation and definitions of methods for our research on railway-track segmentation from forward-facing locomotive video recordings. The goal of the project is to accurately segment the specific rails on which the train is traveling, even in challenging real-world conditions and with limited annotated data. 

To address this problem, we explore multiple neural-network-based approaches with a focus on minimizing model size and preventing overfitting. Alongside two baseline methods, a convolutional neural network (CNN) and a CNN enhanced by morphological postprocessing and connected component labeling, we introduce a novel composite model that transfers contextual information between consecutive video frames. This context-aware design improves segmentation stability and consistency while keeping the number of trainable parameters low.

A key contribution of this work is an unsupervised evaluation methodology designed for scenarios where annotated data are scarce. The proposed metrics assess segmentation quality using colour distinguishability, geometric properties, and colour-variance analysis, enabling more robust comparison of models without relying on ground-truth labels.

All models were trained exclusively on a synthetic dataset created specifically for this project, while evaluations were performed on real-world video sequences to test generalization capabilities. 
