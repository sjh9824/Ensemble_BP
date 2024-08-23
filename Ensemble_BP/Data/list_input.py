import os
import json
from natsort import natsorted

def get_bp_data(bp_json_path):
    with open(bp_json_path, 'r') as f:
        bp_data = json.load(f)
    return bp_data

def create_data_structure(root_dir, bp_json_path, output_json_path):
    bp_data = get_bp_data(bp_json_path)
    data_structure = {}

    # subject 폴더를 자연스럽게 오름차순으로 정렬
    subject_folders = natsorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f.startswith("subject")])

    for subject in subject_folders:
        subject_dir = os.path.join(root_dir, subject)
        subject_id = subject

        # 초기화
        data_structure[subject_id] = {}
        
        # RGB와 YUV 폴더 경로
        rgb_dir = os.path.join(subject_dir, "RGB")
        yuv_dir = os.path.join(subject_dir, "YUV")
        
        # BP 데이터 가져오기
        if subject_id in bp_data and "BP" in bp_data[subject_id]:
            bp_sys = bp_data[subject_id]["BP"]["bp_sys"]
            bp_dia = bp_data[subject_id]["BP"]["bp_dia"]
            bp = [bp_sys, bp_dia]
        else:
            print(f"Warning: BP data not found for {subject_id}.")
            bp = [None, None]
        
        # npy 파일 경로 수집
        for i, (rgb_file, yuv_file) in enumerate(zip(sorted(os.listdir(rgb_dir)), sorted(os.listdir(yuv_dir)))):
            if rgb_file.endswith(".npy") and yuv_file.endswith(".npy"):
                data_structure[subject_id][f"input{i}"] = {
                    "RGB": os.path.join(rgb_dir, rgb_file),
                    "YUV": os.path.join(yuv_dir, yuv_file),
                    "BP": bp
                }

    # JSON 파일로 저장
    with open(output_json_path, 'w') as f:
        print("The input JSON file will be saved.")
        print(f"-Save Path: {output_json_path}")
        json.dump(data_structure, f, indent=4)