# asl_fingerspell

<img width="1700" height="493" alt="image" src="https://github.com/user-attachments/assets/e87271be-a363-4edc-ae71-6e42a03ecf75" />

# ASL Image Classification Project

This project trains machine learning models to classify American Sign Language (ASL) alphabet images. Two deep learning approaches are compared: a custom-built Convolutional Neural Network (CNN) and a transfer learning model based on MobileNetV2. The project also includes a Gradio web application for interactive inference.

## Dataset Organization

The dataset initially consists of a single folder containing training images. The script automatically partitions the data into three folders:

- `train` (approximately 80% of the data)
- `valid` (10% of the data)
- `test` (10% of the data)

Images are randomly assigned to the validation and test sets, and the corresponding directory structure for each ASL class is created as needed.

## Data Preprocessing and Augmentation

To improve model generalization, the training set undergoes data augmentation including rotation, shifting, shearing, zooming, and pixel rescaling. Validation and test images are only rescaled to ensure consistent evaluation.

## Model 1: Custom CNN

The first model is a compact Convolutional Neural Network built from scratch. It includes two convolutional-pooling blocks followed by a flattening step, a dense feature-learning layer, and a softmax classifier. This model provides a baseline for performance and is efficient to train due to its small size.

**Custom CNN Test Accuracy:** 92.72%

## Model 2: MobileNetV2 Transfer Learning

The second model uses MobileNetV2 pretrained on ImageNet. The pretrained layers are initially frozen to retain established image features. A custom classification head is added, consisting of global pooling, a dense layer, dropout, and a softmax output layer for 29 ASL classes. After initial training, a portion of the MobileNetV2 layers is unfrozen for fine-tuning with a low learning rate. This significantly improves accuracy and robustness compared to the custom CNN.

**MobileNetV2 Test Accuracy:** 74.81%

## Model Comparison

Although the MobileNetV2 model benefits from transfer learning and typically excels at general image classification tasks, in this specific problem setup the custom CNN achieved higher accuracy than MobileNetV2. The custom CNN reached 92.72% test accuracy, while MobileNetV2 reached 74.81%. The difference may be due to input resolution, limited training time, or the suitability of the smaller custom model for this specific dataset.

## Gradio Web Application

A Gradio-based web interface allows users to upload an image and receive an ASL letter prediction. To launch the web application, navigate to the source folder and execute:

