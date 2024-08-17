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
   
   ```bash
   pip install -r requirements.txt

  *Note:* The `requirements.txt` may contain unnecessary packages.

3. **Set Paths in `config.yaml`**:  
   Edit the paths in the `config.yaml` file as shown below:

   ```bash
   DATA:
    TYPE: 'vital-video'  # Data Type
    DATA_PATH: "your/path"  # Data Path
    JSON_PATH: 'your/path/numpy_path_with_gt.json'  # Data processing save path (including the JSON file name).
    FS: 30
   

  Make sure to modify the `DATA_PATH` and `JSON_PATH` fields to reflect the locations of your dataset and the JSON output file.

5. **Run the Project**:  
   Once the paths are set, run the project by executing the following command in the terminal:
   ```bash
   python main.py
