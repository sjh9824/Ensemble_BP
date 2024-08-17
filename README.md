# Ensemble Deep Learning for Blood Pressure Estimation Using Facial Videos

이 프로젝트는 "Ensemble Deep Learning for Blood Pressure Estimation Using Facial Videos"라는 제목의 논문에 기반한 연구 결과를 포함하고 있습니다. 이 논문은 "The 3rd RePSS Track2"에서 1등을 수상한 수상작입니다.

## 소개

이 프로젝트는 얼굴 비디오를 사용하여 혈압을 추정하는 딥러닝 기반의 앙상블 모델을 구현합니다. 사용 모델로는 Physformer와 Physnet를 사용했습니다.

## 논문 정보

- **논문 제목:** Ensemble Deep Learning for Blood Pressure Estimation Using Facial Videos
- **저자:**
  - Wei Liu¹, Bingjie Wu¹, Menghan Zhou¹, Xingjian Zheng¹, Xingyao Wang¹, Liangli Zhen¹,∗, Yiping Xie², Chaoqi Luo³

- **소속 정보:**
  - ¹Institute of High Performance Computing, Agency for Science, Technology and Research (A*STAR), Singapore
  - ²College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China
  - ³School of Electrical Engineering, Southwest Jiaotong University, Chengdu, China

- **컨퍼런스:** The 3rd RePSS Track2
- **수상:** 1등

논문 전문은 "(https://liangli-zhen.github.io/assets/pdf/RePPS_BP.pdf)" 에서 확인하실 수 있습니다.

## 모델 구조
![image](https://github.com/user-attachments/assets/a011e1c3-7a4e-459d-bffa-3dc905df0a76)


## Usage

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
