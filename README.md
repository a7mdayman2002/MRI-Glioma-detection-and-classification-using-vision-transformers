# MRI-Glioma-detection-and-classification-using-vision-transformers

## Introduction
This project focuses on the detection and classification of gliomas in MRI images using Vision Transformer (ViT) models. Gliomas are a type of tumor that occurs in the brain and spinal cord, making accurate and early detection crucial for treatment. The goal of this project is to apply state-of-the-art deep learning techniques for the segmentation and classification of gliomas, enabling better clinical decision-making.

## Features:
- Segmentation of brain tumors in MRI images
- Classification of gliomas based on tumor characteristics
- Leveraging Vision Transformer models for better accuracy
- Detailed visualization of the results
Model Overview
This project uses a Vision Transformer (ViT) model, a powerful architecture for image processing tasks. The transformer model is known for its capability to capture global context across the image, making it particularly effective for tasks like tumor segmentation and classification.

The model pipeline consists of:

- Preprocessing MRI images (rescaling, normalization)
- Feeding the images into the Vision Transformer
- Performing segmentation to identify tumor boundaries
- Classifying the glioma based on the segmented area

## Results
The model has shown promising results in both segmentation and classification tasks. Below is a sample output showing the segmented tumor and the classification label:

## Workflow
The overall workflow for the project is as follows:

## Data Collection: 
MRI scans are collected and stored in the data/ directory.
Preprocessing: Images are preprocessed to fit the input requirements of the Vision Transformer.
Training and Inference: The model is trained on the preprocessed data, and inference is performed to classify the glioma.
Post-processing: Results are visualized and analyzed.

## Documentation
For more detailed information about the project, please refer to the Documentation included in the repository.
- I uploaded the flask code with everything needed, just use " git clone https://huggingface.co/Unknown6197/res_classification " and it will work perfectly
- I also Uploaded the best 2 segmentation codes, and the classification code
