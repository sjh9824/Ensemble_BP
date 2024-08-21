import os
import glob
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from Data.vid_to_npy import convert_vid2npy
from Data.list_input import create_data_structure
from sklearn.model_selection import train_test_split
import yaml
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import shutil

class DataProcess():
    def __init__(self, config):
        self.data_type = config['DATA']['TYPE']
        self.json_path = config['DATA']['JSON_PATH']
        self.raw_data_path = config['DATA']['DATA_PATH']
        self.is_first = config['PREPROCESS']['DO']
        self.data_load_json_path = ''
        self.config = config
        
    def data_process(self):
        dir = self.raw_data_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if self.data_type == 'vital-video':
            file_path = self.find_videos_and_ground_truths(dir)
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_load_json_path = os.path.join(current_dir, "Data.json")
            
            self.save_json(file_path, self.data_load_json_path)
            print("Raw Data Path and BP Ground Truth information will be saved as a JSON file at the following path.")
            print(f"-Data Type: {self.data_type}")
            count_videos_in_json(self.data_load_json_path)
            print(f"-Save Path: {self.data_load_json_path}")
        else:
            print(f"{self.data_type} is not supported.")
            
        convert = convert_vid2npy(self.config, self.data_load_json_path)
        convert.vid2npy()
        root_dir = os.path.join(current_dir, 'Process')
        bp_json_path = self.data_load_json_path
        output_json_path = os.path.join(current_dir, "input_numpy_path.json")   
        create_data_structure(root_dir=root_dir,bp_json_path=bp_json_path,output_json_path=output_json_path)

    def save_json(self, data, path):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent = 4)
        
    def find_value_in_json(self, data, key, target_key='value'):
        if isinstance(data, dict):
            if data.get('parameter') == key:
                return data.get(target_key)
            for v in data.values():
                result = self.find_value_in_json(v, key, target_key)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self.find_value_in_json(item, key, target_key)
                if result is not None:
                    return result
        return None
    
    def find_videos_and_ground_truths(self, base_dir):
        dataset = defaultdict(dict)
        subject_index = 0
        
        # 모든 mp4 파일 탐색
        mp4_files = glob.glob(os.path.join(base_dir, '**', '*.mp4'), recursive=True)
        json_files = glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True)

        # Subject 이름을 단순화하여 매핑
        subject_map = {}

        # mp4 파일과 json 파일을 매칭
        for mp4_file in mp4_files:
            # 파일명에서 기본 subject 이름 추출 (0a8ab78a2c1e44718a467c400e78a910_1.mp4 -> 0a8ab78a2c1e44718a467c400e78a910)
            filename = os.path.basename(mp4_file)
            base_name = filename.rsplit('_', 1)[0]
            
            # Subject 이름이 이미 매핑되어 있는지 확인
            if base_name not in subject_map:
                subject_map[base_name] = f"subject{subject_index}"
                subject_index += 1
                
            subject_name = subject_map[base_name]
            
            # 해당 subject의 비디오 리스트에 추가
            if "video" not in dataset[subject_name]:
                dataset[subject_name]["video"] = []
            dataset[subject_name]["video"].append(mp4_file)
        
        # 비디오 파일의 순서를 -1, -2로 정렬
        for subject_name, data in dataset.items():
            data["video"].sort(key=lambda x: x.split('_')[-1])  # -1.mp4가 먼저 오도록 정렬
        
        # json 파일을 해당 subject와 연결
        for json_file in json_files:
            filename = os.path.basename(json_file)
            base_name = filename.split('.')[0]
            
            if base_name in subject_map:
                subject_name = subject_map[base_name]
                dataset[subject_name]["ground_truth"] = json_file
                
                with open(dataset[subject_name]["ground_truth"], 'r') as file:
                    data = json.load(file)
                bp_sys_value = self.find_value_in_json(data, 'bp_sys')
                bp_dia_value = self.find_value_in_json(data,'bp_dia')
                if bp_sys_value is not None and bp_dia_value is not None:
                    dataset[subject_name]["BP"] = {"bp_sys": bp_sys_value, "bp_dia": bp_dia_value}
                    
                
        return dataset
    
class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, yuv_path, bp = self.samples[idx]
        rgb_data = np.load(rgb_path)
        yuv_data = np.load(yuv_path)
        
        # Convert to torch tensors
        rgb_data = torch.tensor(rgb_data, dtype=torch.float32)
        yuv_data = torch.tensor(yuv_data, dtype=torch.float32)
        bp = torch.tensor(bp, dtype=torch.float32)
        
        return rgb_data, yuv_data, bp
            
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error while loading YAML file: {exc}")
            return None

# Custom Dataset to load npy files based on JSON configuration

def load_and_split_data(json_path):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Flatten the data structure for easier access
    samples = []
    for subject, inputs in data.items():
        for input_name, input_data in inputs.items():
            rgb_path = input_data['RGB']
            yuv_path = input_data['YUV']
            bp = input_data['BP']
            samples.append((rgb_path, yuv_path, bp))
    
    # Extract subjects and corresponding samples
    subjects = list(data.keys())
    
    # Split subjects into train, valid, and test sets
    train_subjects, temp_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
    valid_subjects, test_subjects = train_test_split(temp_subjects, test_size=1/3, random_state=42)
    
    # Filter samples by subjects for each set
    train_samples = [s for s in samples if any(subject in s[0] for subject in train_subjects)]
    valid_samples = [s for s in samples if any(subject in s[0] for subject in valid_subjects)]
    test_samples = [s for s in samples if any(subject in s[0] for subject in test_subjects)]
    
    return train_samples, valid_samples, test_samples

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create a general generator for use with the validation dataloader,
    # the test dataloader, and the unsupervised dataloader
    general_generator = torch.Generator()
    general_generator.manual_seed(seed)
    
    # Create a training generator to isolate the train dataloader from
    # other dataloaders and better control non-deterministic behavior
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    
    return general_generator, train_generator

def count_videos_in_json(json_file_path):
    # JSON 파일 로드
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    total_subjects = len(data)
    total_videos = 0
    
    # 각 subject에 대해 비디오 개수 계산
    for subject, details in data.items():
        video_count = len(details['video'])
        total_videos += video_count
    
    print(f"-Total subjects: {total_subjects}")
    print(f"-Total videos: {total_videos}")
    
def filter_and_save_json(input_json_path, output_json_path, config):
    """
    주어진 JSON 파일을 로드하여 조건에 맞는 npy 파일들만 필터링하고,
    새로운 JSON 구조로 저장하는 함수.

    Parameters:
    - input_json_path (str): 입력 JSON 파일 경로.
    - output_json_path (str): 출력 JSON 파일 경로.
    - config (dict): 처리에 사용할 설정을 담은 사전.
    """

    # JSON 파일 로드
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # 새로 저장할 JSON 데이터 구조
    new_data = {}

    # 통계 정보를 저장하기 위한 변수들
    total_paths = 0
    removed_paths = 0
    saved_paths = 0
    removed_info = []

    # config에서 shape 정보와 필터링 기준 가져오기
    expected_shape = (
        config['PREPROCESS']['CHUNK_LENGTH'],
        config['TRAIN']['RESIZE']['H'],
        config['TRAIN']['RESIZE']['W'],
        3
    )

    # 유효 프레임 수를 계산
    filter_percentage = config['PREPROCESS']['FILTER_PERCENTAGE']
    chunk_length = config['PREPROCESS']['CHUNK_LENGTH']
    frame_threshold = int(chunk_length * (filter_percentage / 100))

    total_paths = sum(len(subject_value) for subject_value in data.values())

    # tqdm을 사용하여 진행률 표시
    with tqdm(total=total_paths, desc="Processing", unit="file") as pbar:
        # 각 subject를 처리
        for subject_key, subject_value in data.items():
            new_data[subject_key] = {}
            for input_key, input_value in subject_value.items():
                remove_reason = None

                # npy 파일 로드
                rgb_path = input_value['RGB']
                yuv_path = input_value['YUV']

                try:
                    rgb_data = np.load(rgb_path)
                    yuv_data = np.load(yuv_path)
                except Exception as e:
                    remove_reason = "Error loading file"
                    removed_paths += 1
                    removed_info.append((subject_key, input_key, remove_reason))
                    pbar.update(1)
                    continue

                # 크기 체크
                if rgb_data.shape != expected_shape or yuv_data.shape != expected_shape:
                    remove_reason = "Incorrect shape"
                    removed_paths += 1
                    removed_info.append((subject_key, input_key, remove_reason))
                    pbar.update(1)
                    continue

                # 이미지들이 모두 0으로 채워진 경우 체크
                zero_frames_rgb = np.sum(np.all(rgb_data == 0, axis=(1, 2, 3)))
                zero_frames_yuv = np.sum(np.all(yuv_data == 0, axis=(1, 2, 3)))

                # 지정된 프레임 이상이 유효해야 함
                if zero_frames_rgb > (chunk_length - frame_threshold) or zero_frames_yuv > (chunk_length - frame_threshold):
                    remove_reason = "Too many zero frames"
                    removed_paths += 1
                    removed_info.append((subject_key, input_key, remove_reason))
                    pbar.update(1)
                    continue

                # 유효한 파일만 저장
                new_data[subject_key][input_key] = {
                    "RGB": rgb_path,
                    "YUV": yuv_path,
                    "BP": normalize_bp(input_value['BP'])
                }
                saved_paths += 1
                pbar.update(1)

    # 새로운 JSON 파일로 저장
    with open(output_json_path, 'w') as file:
        json.dump(new_data, file, indent=4)

    # 처리 통계 출력
    print(f"\nTotal paths processed: {total_paths}")
    print(f'The percentage of valid frames needed for filtering: {filter_percentage}')
    print(f"Total paths removed: {removed_paths}")
    print("Removed subject and inputs:")
    for subject, input_key, reason in removed_info:
        print(f" - {subject}: {input_key} (Reason: {reason})")
    print(f"Total paths saved: {saved_paths}")

def normalize_bp(bp, min_sbp=70, max_sbp=180, min_dbp=40, max_dbp=120):
    sbp, dbp = bp
    normalized_sbp = (sbp - min_sbp) / (max_sbp - min_sbp)
    normalized_dbp = (dbp - min_dbp) / (max_dbp - min_dbp)
    return [normalized_sbp, normalized_dbp]