# Breast Cancer Image Classification with DenseNet121

### Overview:
This repository focuses on a project dedicated to classifying **Breast Ultrasound Images** into three categories: 
- **benign**
- **malignant**
- **normal**

The classification model used in this project relies on deep learning techniques, particularly **Convolutional Neural Networks (CNN)** and **Transfer Learning**, harnessing the power of a base model called **DenseNet121**. It is trained on a comprehensive dataset of annotated breast ultrasound images, leveraging these deep learning approaches to enhance its performance.

Transfer learning involves adapting a pre-trained model to a specific task by fine-tuning it with a smaller, task-specific dataset. Here, the initial layers of DenseNet121, having acquired general features from a diverse dataset, remain frozen. Subsequent layers, responsible for learning task-specific details, are adjusted to better align with the specific characteristics of breast ultrasound image classification. This approach enables the model to benefit from the broader dataset's knowledge, improving its effectiveness in addressing the unique challenges posed by breast cancer classification.

----------------------

### Dataset Used: 
#### Breast Ultrasound Images
- Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.

The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant.

**Source:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

**Download Dataset:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

----------------------

### Algorithm Used:
#### Neural Networks (Supervised Learning)
- 
----------------------

### Architecture Used: 
#### Convolutional Neural Network (CNN)
- 

----------------------

### Base Model: 
#### DenseNet121
-

**Download Weights File Here:** https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5

----------------------

### Dependencies: 

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

**Installation:**
1. Download the **'requirements.txt'** file from this repository.
2. Open a terminal or command prompt and change the current working directory to where your **requirements.txt** file is located.
3. Run the following command: **pip install -r requirements.txt**
4. Verify the installed dependencies by running the following command: **pip list**

----------------------

### Model: 
After evaluating the model using the test set (test_images), it demonstrated an overall accuracy of "**85.44%**". 

Download Here: 

----------------------

### Streamlit App:
https://breast-cancer-image-classification-with-densenet121-v9dybugp4h.streamlit.app/
