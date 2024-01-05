# Breast Cancer Image Classification with DenseNet121

## Overview
This repository focuses on a project dedicated to classifying **Breast Ultrasound Images** into three categories: 
- **benign**
- **malignant**
- **normal**

The classification model used in this project relies on deep learning techniques, particularly **Convolutional Neural Networks (CNN)** and **Transfer Learning**, harnessing the power of a base model called **DenseNet121**. It is trained on a comprehensive dataset of annotated breast ultrasound images, leveraging these deep learning approaches to enhance its performance.

Transfer learning involves adapting a pre-trained model to a specific task by fine-tuning it with a smaller, task-specific dataset. Here, the initial layers of DenseNet121, having acquired general features from a diverse dataset, remain frozen. Subsequent layers, responsible for learning task-specific details, are adjusted to better align with the specific characteristics of breast ultrasound image classification. 

----------------------

## Dataset
#### Breast Ultrasound Images
- Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.

The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant.

**Source:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

**Download Dataset:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

----------------------

## Algorithm:
#### Neural Networks (Supervised Learning)
- 
----------------------

### Architecture 
#### Convolutional Neural Network (CNN)
- 

----------------------

## Base Model
#### DenseNet121
-

**Download Weights File Here:** https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5

----------------------

## Dependencies

***Main***
- **Pandas** - a data analysis library that provides data structures like DataFrames for efficient data handling. 
- **NumPy** - a library for scientific computing with Python.
- **Matplotlib** - a comprehensive library for creating static, interactive, and animated plots in Python, facilitating data visualization tasks.
- **Seaborn**
- **Pillow**
- **TensorFlow**
- **Keras** - a high-level neural networks API written in Python, capable of running on top of TensorFlow. This library is utilized for building and training deep learning models.
- **scikit-learn**

***Others***
- **h5py**
- **Requests**
- **Streamlit**

**TXT File:** [requirements.txt](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/blob/f8a7a3b747ab8b3d81acb7bbda0251ff1063ee14/requirements.txt)

----------------------

## Model 
After evaluating the model using the test set (test_images), it demonstrated an overall accuracy of "**85.44%**". 

**Download Here:**
Whole | [model.h5](https://drive.google.com/file/d/14tfAoUQDBRwJKL-5ooWFXqMLy-Ki2lX7/view?usp=drive_link)
Splitted File | [splitted_model](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/tree/fcb1bcfedd11d733427bd96ae791ed0fbeefdcd5/splitted_model)

----------------------

## Streamlit App
**Click this link to open:** https://breast-cancer-image-classification-with-densenet121-v9dybugp4h.streamlit.app/
