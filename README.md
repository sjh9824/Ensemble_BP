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


## 사용법

1. Data_Preprocess.ipynb 를 통해 vital_video dataset의 위치 정보 및 Path 설정을 진행합니다.
   
2. Create_npy_path_to_json.ipynb 를 통해 vital_video를 preprocessing한 RGB 정보 numpy와 YUV 정보 numpy의 경로와 ground_truth가 정리된 json 파일을 얻습니다.

3. main.py에 2번을 통해 얻은 json파일의 경로를 적어주고 실행합니다.

본 논문에서는 Vital_Video 데이터셋을 학습하고 OBF Dataset으로 Train을 진행하지만 아직 Vital_video dataset만을 이용해 train, valid, test를 진행하고 있습니다.
OBF test는 추후 추가하겠습니다.

이 코드는 개인 연구를 위해 구현하였습니다.
편의성 부분은 고려하지 못했음을 알려드립니다.
