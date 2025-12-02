# CTCMrailwaySeg: Railway segmentation of video captures with a transfer of contextual information with unsupervised evaluation

## Introduction
This repository contains the implementation and definitions of methods for our research on railway-track segmentation from forward-facing locomotive video recordings. The goal of the project is to accurately segment the specific rails on which the train is traveling, even in challenging real-world conditions and with limited annotated data. 

To address this problem, we explore multiple neural-network-based approaches with a focus on minimising model size and preventing overfitting. Alongside two baseline methods, a convolutional neural network (CNN) and a CNN enhanced by morphological postprocessing and connected component labelling, we introduce a novel composite model that transfers contextual information between consecutive video frames. This context-aware design improves segmentation stability and consistency while keeping the number of trainable parameters low.

A key contribution of this work is an unsupervised evaluation methodology designed for scenarios where annotated data are scarce. The proposed metrics assess segmentation quality using colour distinguishability, geometric properties, and colour-variance analysis, enabling more robust comparison of models without relying on ground-truth labels.

All models were trained exclusively on a synthetic dataset created specifically for this project, while evaluations were performed on real-world video sequences to test generalisation capabilities. 

## process_images.py
This script implements an end-to-end image-processing pipeline that performs semantic segmentation and track-mask extraction using three models in parallel: (1) a baseline semantic segmentation using BiSeNetV2, (2) a refined BiSeNetV2 variant with additional sem-cc post-processing, and (3) a CTCM-based rail/track extraction model operating with temporal context. It loads an input image, normalises it, and infers semantic masks via the models. The CTCM blocks to segment rail components, applies connected-component analysis with a correction policy to refine the track region, and finally predicts rail/track masks. The results are composed into a visual comparison grid and saved for each processed image.

## process_video.py
This script implements a video evaluation pipeline that runs three segmentation models in parallel on railway-scene frames: (1) a baseline semantic segmentation using BiSeNetV2, (2) a refined BiSeNetV2 variant with additional sem-cc post-processing, and (3) a CTCM-based rail/track extraction model operating with temporal context. It preprocesses the image, infers the BiSeNetV2 mask, derives an improved mask via sem-cc, and in parallel feeds the frame through a two-stage CTCM pipeline with connected-component selection, morphological erosion, correction heuristics, and optional context propagation/reset based on quality metrics. The three resulting masks are then visualised side by side with the original frame and written into an output video, while detailed per-stage timing (BiSeNet, sem-cc, CTCM) is optionally logged to CSV for performance analysis.
