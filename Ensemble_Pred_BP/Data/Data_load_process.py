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
        else:
            print(f"{self.data_type} is not supported.")
            
        convert = convert_vid2npy(self.config, self.data_load_json_path)
        convert.vid2npy()
        root_dir = os.path.join(current_dir, 'Preprocess')
        bp_json_path = self.data_load_json_path
        output_json_path = self.json_path
        create_data_structure(root_dir=root_dir,bp_json_path=bp_json_path,output_json_path=output_json_path)

    def save_json(self, data, path):
        print("Raw Data Path and BP Ground Truth information will be saved as a JSON file at the following path.")
        print(f"-Data Type: {self.data_type}")
        count_videos_in_json(path)
        print(f"-Save Path: {path}")
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

def check_npy_shapes(json_file_path):
    # JSON 파일 로드
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 모든 npy 파일의 경로를 가져오기
    subject_info = {}
    npy_paths = []
    for subject, inputs in data.items():
        rgb_paths = []
        yuv_paths = []
        for input_name, input_data in inputs.items():
            rgb_paths.append(input_data['RGB'])
            yuv_paths.append(input_data['YUV'])
            npy_paths.append(input_data['RGB'])
            npy_paths.append(input_data['YUV'])
        
        subject_info[subject] = {
            'RGB_count': len(rgb_paths),
            'YUV_count': len(yuv_paths),
            'RGB_paths': rgb_paths,
            'YUV_paths': yuv_paths
        }

    # 각 subject의 RGB와 YUV 파일 개수 비교
    for subject, info in subject_info.items():
        print(f"{subject}: RGB files = {info['RGB_count']}, YUV files = {info['YUV_count']}")
        if info['RGB_count'] != info['YUV_count']:
            print(f"Warning: {subject} has a different number of RGB and YUV files.")

    # npy 파일의 형태 확인
    shapes = []
    for npy_path in npy_paths:
        npy_array = np.load(npy_path)
        shapes.append((npy_path, npy_array.shape))

    # 첫 번째 npy 파일의 shape을 기준으로 나머지 파일과 비교
    first_shape = shapes[0][1]
    different_shapes = [(path, shape) for path, shape in shapes if shape != first_shape]

    # 결과 출력
    if len(different_shapes) == 0:
        print("All input shapes are identical.")
        print(f"- Input Shape: {first_shape}")
    else:
        print("There are npy files with different shapes.")
        for path, shape in different_shapes:
            print(f"- File: {path}, Shape: {shape}")
            
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