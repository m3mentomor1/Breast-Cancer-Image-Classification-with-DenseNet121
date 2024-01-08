# Breast Cancer Image Classification with DenseNet121

### 🧐 I. Overview
This repository focuses on a project dedicated to classifying **Breast Ultrasound Images** into three categories: 
- **benign**
- **malignant**
- **normal**

This aims to accurately determine if there is a presence of **Breast Cancer**.

The classification model used in this project relies on deep learning techniques, particularly **Convolutional Neural Networks (CNN)** and **Transfer Learning**, harnessing the power of a base model called **DenseNet121**. It is trained on a comprehensive dataset of annotated breast ultrasound images, leveraging these deep learning approaches to enhance its performance.

Transfer learning involves adapting a pre-trained model to a specific task by fine-tuning it with a smaller, task-specific dataset. Here, the initial layers of DenseNet121, having acquired general features from a diverse dataset, remain frozen. Subsequent layers, responsible for learning task-specific details, are adjusted to better align with the specific characteristics of breast ultrasound image classification. 

This project involves **Multiclass Classification** (a type of machine learning task where the goal is to categorize input data points into three or more classes or categories).

----------------------

### 🗂️ II. Dataset
![image](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/assets/95956735/47e76b23-1542-4576-9127-6f30077733f3)

#### Breast Ultrasound Images
- Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.
- The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 1578 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are **normal**, **benign**, and **malignant**.

**Source:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

**Download Dataset Here:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

----------------------

### 🧑🏻‍💻 III. Learning Approach Used
#### Supervised Learning
- a type of learning in machine learning that involves training an algorithm on labeled data, where input samples are paired with corresponding output labels. The objective is to learn a mapping from input data to correct output labels by adjusting internal parameters during training, minimizing the difference between predicted outputs and true labels.
##
**Why Supervised Learning?**

The model's training adopted a supervised learning approach, as the dataset included explicit labels for each image. 

----------------------

### 🧮 IV. Algorithm Used
#### Neural Networks 
- also known as Artificial Neural Networks (ANNs), are a class of algorithms inspired by the structure and functioning of the human brain. It consists of interconnected nodes organized into layers. These layers typically include an input layer, one or more hidden layers, and an output layer. Each connection between nodes has an associated weight, and nodes within a layer may have activation functions.
##
**Why Neural Networks?**

In image classification, neural networks excel at capturing subtle patterns and variations that conventional algorithms may find challenging. Their hierarchical structure allows them to proficiently learn and represent features across diverse levels of abstraction, proving highly effective for the specific requirements of image classification tasks.

----------------------

### 📐 V. Model Architecture 
#### Convolutional Neural Network (CNN)
- a type of deep learning model architecture designed specifically for processing structured grid data, such as images. CNNs have proven to be highly effective in computer vision tasks, including image classification, object detection, segmentation, and more. They are characterized by their ability to automatically and adaptively learn spatial hierarchies of features directly from the data.
##
**Why CNN?**

CNNs have proven to be highly effective in computer vision tasks due to their ability to automatically and adaptively learn spatial hierarchies of features directly from diverse data sources such as images and videos.
##
The model's architecture is implemented using the **Keras Sequential API**. Below is a detailed summary of its layers:

1. **DenseNet121 Base Model (Functional):**
   - Output Shape: (None, 8, 8, 1024)
   - Parameters: 7,037,504

2. **Flatten Layer:**
   - Output Shape: (None, 65536)

3. **Dense Layer 1:**
   - Output Shape: (None, 1024)
   - Activation Function: ReLU (Rectified Linear Unit)
   - Parameters: 67,109,888

4. **Dropout Layer 1:**
   - Rate: 0.5

5. **Dense Layer 2:**
   - Output Shape: (None, 1024)
   - Activation Function: ReLU (Rectified Linear Unit)
   - Parameters: 1,049,600

6. **Dropout Layer 2:**
   - Rate: 0.3

7. **Dense Layer 3:**
   - Output Shape: (None, 512)
   - Activation Function: ReLU (Rectified Linear Unit)
   - Parameters: 524,800

8. **Dense Layer 4:**
   - Output Shape: (None, 128)
   - Activation Function: ReLU (Rectified Linear Unit)
   - Parameters: 65,664

9. **Dense Output Layer:**
   - Output Shape: (None, 3)
   - Activation Function: Softmax
   - Parameters: 387

**Total Trainable Parameters:** 68,750,339 (262.26 MB)  
- These are the parameters that the model learns and adjusts during the training process. They include the weights and biases in the dense and fully connected layers. The number **68,750,339** represents the total count of trainable parameters, and the value **262.26 MB** is an approximation of the memory required to store these parameters.

**Total Non-trainable Parameters:** 7,037,504 (26.85 MB)
- These are the parameters that are not updated during training. In this case, they likely correspond to the weights of the pre-trained DenseNet121 base model. The number **7,037,504** represents the total count of non-trainable parameters, and the value **26.85 MB** is an approximation of the memory required to store these parameters.

----------------------

### 🚩 VI. Base Model
#### DenseNet121
- a CNN architecture widely used as a base model for image classification tasks. It is part of the DenseNet (Densely Connected Convolutional Networks) family of models. These architectures are characterized by dense connectivity patterns, where each layer receives direct input from all preceding layers. The "121" in DenseNet121 denotes the total number of layers in the network.

**Download Weights File Here:** https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5

----------------------

### 📦 VII. Dependencies

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

**Download TXT File Here:** [requirements.txt](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/blob/f8a7a3b747ab8b3d81acb7bbda0251ff1063ee14/requirements.txt)

----------------------

### 🧾 VIII. Model Evaluation

**DataFrame**
| Set          | Loss      | Accuracy  |
|--------------|-----------|-----------|
| Train        | 0.038702  | 99.04%    |
| Validation   | 0.188313  | 90.97%    |
| Test         | 0.136501  | 94.94%    |
##
**Confusion Matrix (Test Set)**

![image](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/assets/95956735/48203e05-3c58-441c-aa56-7064245d77d9)

- Row 1: This corresponds to the "benign" class.
   > Column 1 (89): The model predicted "benign," and the true class was also "benign." There are 89 instances (image/s) where both the prediction and the true class are "benign."
   
   > Column 2 (0): The model predicted "malignant," but the true class was "benign." There are 0 instances (image/s) of this case.
   
   > Column 3 (1): The model predicted "normal," but the true class was "benign." There is 1 instance of this case.

- Row 2: This corresponds to the "malignant" class.
   > Column 1 (5): The model predicted "benign," but the true class was "malignant." There are 5 instances (image/s) of this case.
   
   > Column 2 (37): The model predicted "malignant," and the true class was also "malignant." There are 37 instances (image/s) where both the prediction and the true class are "malignant."
   
   > Column 3 (0): The model predicted "normal," but the true class was "malignant." There are 0 instances (image/s) of this case.

- Row 3: This corresponds to the "normal" class.
   > Column 1 (2): The model predicted "benign," but the true class was "normal." There are 2 instances (image/s) of this case.
   
   > Column 2 (0): The model predicted "malignant," but the true class was "normal." There are 0 instances (image/s) of this case.
   
   > Column 3 (24): The model predicted "normal," and the true class was also "normal." There are 24 instances (image/s) where both the prediction and the true class are "normal."
##
**Classification Report (Test Set)**
|            | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Benign     | 0.93      | 0.99   | 0.96     | 90      |
| Malignant  | 1.00      | 0.88   | 0.94     | 42      |
| Normal     | 0.96      | 0.92   | 0.94     | 26      |
|------------|-----------|--------|----------|---------|
| Accuracy   |           |        | 0.95     | 158     |
| Macro Avg  | 0.96      | 0.93   | 0.94     | 158     |
| Weighted Avg| 0.95      | 0.95   | 0.95     | 158     |

- **Precision** - a measure of the accuracy of the positive predictions. It is the ratio of true positive predictions to the sum of true positives and false positives. In the table, it is presented for each class (Benign, Malignant, Normal) and also as Macro Avg (average across classes) and Weighted Avg (weighted average based on support).

- **Recall** - also known as sensitivity or true positive rate, is the ratio of true positive predictions to the sum of true positives and false negatives. It is presented for each class, as well as Macro Avg and Weighted Avg.

- **F1-Score** - the harmonic mean of precision and recall. It is a single metric that combines both precision and recall. Like precision and recall, it is presented for each class, Macro Avg, and Weighted Avg.

- **Support** - the number of actual occurrences of the class in the specified dataset. It is the last column for each class.

- **Accuracy** - the ratio of correctly predicted instances to the total instances. It is presented as an overall accuracy in the last column of the Accuracy row.

- **Macro Avg** - this row presents the macro-averaged values for precision, recall, and F1-Score. Macro-averaging calculates the metric independently for each class and then takes the average. It treats all classes equally.

- **Weighted Avg** - this row presents the weighted-averaged values for precision, recall, and F1-Score. Weighted averaging is similar to macro-averaging, but it takes into account the number of instances for each class. It is often used when there is an imbalance in class distribution.
##
**Overall Accuracy**

After evaluating the model using the test set ([test_images](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/tree/7eb700a208efcd84beafbf412274f594bf1b85ce/test_images)), it demonstrated an overall accuracy of:
### "94.94%"

(For reference see: [model_evaluation.ipynb](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/blob/8b2bca760563ec8af7e8245f871679e49711777f/model_evaluation.ipynb))
##
**Download Model Here:**
- Single File: [model.h5](https://drive.google.com/file/d/1lxY2mH7dQ9hVh5mYipKWLCBuKJRDD1sj/view?usp=drive_link)
- Splitted File: [splitted_model](https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/tree/fcb1bcfedd11d733427bd96ae791ed0fbeefdcd5/splitted_model) (Splitted using [PineTools File Splitter](https://pinetools.com/split-files))

----------------------

### 🚀 IX. Model Deployment
<img src="https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/assets/95956735/6d0001fd-6890-44aa-8f25-223b21e8ab39" width="300" />

- a free and open-source Python framework to rapidly build and share beautiful machine learning and data science web apps. 
##
The model is deployed on **Streamlit**, allowing for a straightforward and accessible user interface where users can conveniently do breast cancer image classification.

**Access the app here:** https://breast-cancer-image-classification-with-densenet121-v9dybugp4h.streamlit.app/

----------------------

### 🛠️ X. How to Use?

**1. Clone this repository**

   Paste this command on your terminal: 
   ```
   git clone https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121.git
   ```

**2. Go to the repository's main directory**
   
   Paste this command on your terminal:
   ```
   cd Breast-Cancer-Image-Classification-with-DenseNet121
   ```

**3. Install dependencies**

   Paste this command on your terminal:
   ```
   pip install -r requirements.txt
   ```
