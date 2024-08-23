# Ensemble Deep Learning for Blood Pressure Estimation Using Facial Videos

This project is based on the research results of the paper titled "Ensemble Deep Learning for Blood Pressure Estimation Using Facial Videos." This paper was the first-place winner at "The 3rd RePSS Track2."

## Introduction

This project implements an ensemble deep learning model that estimates blood pressure using facial videos. The models used include Physformer and Physnet.

## Paper Information

- **Paper Title:** Ensemble Deep Learning for Blood Pressure Estimation Using Facial Videos
- **Authors:**
  - Wei Liu¹, Bingjie Wu¹, Menghan Zhou¹, Xingjian Zheng¹, Xingyao Wang¹, Liangli Zhen¹,∗, Yiping Xie², Chaoqi Luo³

- **Affiliations:**
  - ¹Institute of High Performance Computing, Agency for Science, Technology and Research (A*STAR), Singapore
  - ²College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China
  - ³School of Electrical Engineering, Southwest Jiaotong University, Chengdu, China

- **Conference:** The 3rd RePSS Track2
- **Award:** 1st Place

You can find the full paper [here](https://liangli-zhen.github.io/assets/pdf/RePPS_BP.pdf).

## Model Architecture
![image](https://github.com/user-attachments/assets/a011e1c3-7a4e-459d-bffa-3dc905df0a76)

## Usage

1. **Install Required Packages**:  
   Install the dependencies listed in the `requirements.txt` file using the following command:  
   *Note:* The `requirements.txt` may contain unnecessary packages.
   ```bash
   pip install -r requirements.txt

2. **Set Paths in `config.yaml`**:  
   Edit the paths in the `config.yaml` file as shown below:  
   Make sure to modify the `DATA_PATH` and `JSON_PATH` fields to reflect the locations of your dataset and the JSON output file.
   ```bash
   DATA:
    TYPE: 'vital-video'  # Data Type
    DATA_PATH: "your/path"  # Data Path
    JSON_PATH: 'your/path/numpy_path_with_gt.json'  # Data processing save path (including the JSON file name).
    FS: 30

3. **Run the Project**:  
   Once the paths are set, run the project by executing the following command in the terminal:
   ```bash
   python main.py


## Result

1. **Train Curve**:
   ![train_valid_loss_plot](https://github.com/user-attachments/assets/32dc0d61-632a-4ec6-8152-d3f02c5b8e94)
  Train Epoch: 50  
  Best Epoch: 50(last)  
  Dataset: Vital Video(250 subject)  
           &emsp;Processing: Split in 160 frames  
           &emsp;RGB, YUV (3 channels)  
           &emsp;Face Detection (Use Retina Face)  
           &emsp;128 * 128 resized  
           &emsp;Input Vector Size [160, 128, 128, 3]   
  Split Data: Train : Valid : Test = 7 : 2 : 1  

   
2. **Scatter**:  
   ![Both_Both_Scatter](https://github.com/user-attachments/assets/8c2903a7-5db9-4b3f-93f3-41ee08554ad9)
  Using Model: Both(Physforemr, Physnet)  
  Using Data Type: Both(RGB, YUV)  

3. **Test Table**:  
   Test Table in using model type and data type  
   (Both : Using both model or data type)  
   Metrics : MAE, RMSE, Pearson Correlation(r)  
   ![GetImage](https://github.com/user-attachments/assets/36345414-9120-4ed3-be8d-f1866ac9561e)
