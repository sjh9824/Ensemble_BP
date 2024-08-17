import os
import glob
from collections import defaultdict
import json
import random
import cv2
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace
from mtcnn import MTCNN
import torch

class FaceDetector:
    def __init__(self):
        self.last_face_box_coor = None
    
    def detect_and_resize_face(self, image, frame_count, target_size=(128, 128), backend='HC', use_larger_box=False, larger_box_coef=2.0):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        global device_printed
        
        face_box_coor = None
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hc_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
        
        # HC 백엔드는 기존처럼 처리
        if backend == "HC":
            detector = cv2.CascadeClassifier(hc_path)
            face_zone = detector.detectMultiScale(image_rgb)

            if len(face_zone) < 1:
                return None
            elif len(face_zone) >= 2:
                max_width_index = np.argmax(face_zone[:, 2])
                face_box_coor = face_zone[max_width_index]
            else:
                face_box_coor = face_zone[0]

        # RF와 MT 백엔드는 frame_count에 따라 처리
        elif backend == "RF" or backend == "MT":
            if frame_count == 0 or frame_count % 160 == 0 or self.last_face_box_coor is None:
                if backend == "RF":
                    res = RetinaFace.detect_faces(image_rgb)

                    if isinstance(res, dict) and len(res) > 0:
                        highest_score_face = max(res.values(), key=lambda x: x['score'])
                        face_zone = highest_score_face['facial_area']

                        x_min, y_min, x_max, y_max = face_zone
                        x = x_min
                        y = y_min
                        width = x_max - x_min
                        height = y_max - y_min
                        center_x = x + width // 2
                        center_y = y + height // 2
                        square_size = max(width, height)
                        new_x = center_x - (square_size // 2)
                        new_y = center_y - (square_size // 2)
                        face_box_coor = [new_x, new_y, square_size, square_size]
                    else:
                        print("ERROR: No Face Detected - Using last detected face position")
                        face_box_coor = self.last_face_box_coor

                elif backend == "MT":
                    detector = MTCNN(keep_all=True, device=device)
                    results = detector.detect_faces(image_rgb)
                    if len(results) < 1:
                        return None
                    elif len(results) >= 2:
                        max_width_index = np.argmax([r['box'][2] for r in results])
                        face_box_coor = results[max_width_index]['box']
                    else:
                        face_box_coor = results[0]['box']

                # 얼굴 위치를 저장
                self.last_face_box_coor = face_box_coor
            else:
                # 이전에 검출된 얼굴 위치 사용
                face_box_coor = self.last_face_box_coor

        # 더 큰 박스를 사용할 경우
        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = int(larger_box_coef * face_box_coor[2])
            face_box_coor[3] = int(larger_box_coef * face_box_coor[3])
            
        # 얼굴 크롭 및 리사이즈
        x, y, w, h = face_box_coor
        face_crop = image[int(y):int(y+h), int(x):int(x+w)]
        resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

        return resized_face

class convert_vid2npy():
    def __init__(self, config, path):
        self.vid_path = path
        self.config = config
        self.H = config['TRAIN']['RESIZE']['H']
        self.W = config['TRAIN']['RESIZE']['W']
        self.backend = config['PREPROCESS']['BACKEND']
        self.chunk_size = config['PREPROCESS']['CHUNK_LENGTH']
        
        
    def vid2npy(self):
        with open(self.vid_path, 'r') as f:
            data = json.load(f)

        processed_data = self.process_all_subjects(data, self.vid_path)
        self.save_to_json(processed_data)
        
    def process_all_subjects(self, data, output_dir):
    # 완료된 subject 목록을 로드 또는 초기화        
        with tqdm(total=len(data.keys()), desc="Processing Subjects", unit="subject") as pbar_subjects:
            processed_data = {}
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = config_file_path = os.path.join(current_dir, 'Process')
            for subject in data.keys():
                
                # 기존 코드로 subject 처리
                details = data[subject]
                video_paths = details['video']
                BP = details['BP']
                rgb_files = []
                yuv_files = []
                npy_index = 0

                
                #HC, RF, MT
                for video_path in video_paths:
                    subject_output_dir = os.path.join(output_dir, subject)
                    rgb_npy_files, yuv_npy_files, npy_index = self.process_video(video_path, subject_output_dir, target_size=(self.H,self.W), backend=self.backend, chunk_size=self.chunk_size,start_idx=npy_index,subject_name=subject)
                    rgb_files.extend(rgb_npy_files)
                    yuv_files.extend(yuv_npy_files)

                processed_data[subject] = {
                    "video_npy": {
                        "RGB": rgb_files,
                        "YUV": yuv_files
                    },
                    "BP": BP
                }  
                
                pbar_subjects.update(1) 
                
            return processed_data
    
    def save_to_json(self, data):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "input_numpy_path.json")
        
        print("The JSON file containing the organized paths of the input Numpy files will be saved.")
        print(f"-Save Path: {output_dir}")
        
        with open(output_dir, 'w') as f:
            json.dump(data, f, indent=4)
            
    def process_video(self, video_path, output_dir, target_size, backend, use_larger_box=False, larger_box_coef=1.0, chunk_size=160, start_idx=0, subject_name="subject"):
        face_detector = FaceDetector()
        rgb_output_dir = os.path.join(output_dir, "RGB")
        yuv_output_dir = os.path.join(output_dir, "YUV")
        image_save_dir = os.path.join(output_dir, "Detect and crop and resize")

        if not os.path.exists(rgb_output_dir):
            os.makedirs(rgb_output_dir)
        if not os.path.exists(yuv_output_dir):
            os.makedirs(yuv_output_dir)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        face_frames = []
        rgb_files = []
        yuv_files = []
        npy_index = start_idx
        
        total_num = total_frames - (total_frames % 160)

        with tqdm(total=total_num, desc=f"Processing {subject_name}", unit="frame", leave=False) as pbar_frames:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                
                if total_frames - frame_count < chunk_size:
                    break

                # Crop and resize the face, and update the progress bar
                resized_face = face_detector.detect_and_resize_face(frame, frame_count, target_size, backend, use_larger_box, larger_box_coef )
                #print(resized_face)
                if resized_face is not None:
                    face_frames.append(resized_face)
                else:
                    face_frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                frame_count += 1

                if frame_count % chunk_size == 0:
                    face_numpy = np.array(face_frames)
                    
                    face_frames_np = self.diff_normalize_data(face_numpy)
                    
                    # RGB npy 파일 저장
                    rgb_output_file = os.path.join(rgb_output_dir, f"RGB_{subject_name}_input{npy_index}.npy")
                    np.save(rgb_output_file, face_frames_np)
                    rgb_files.append(rgb_output_file)
                    
                    # YUV로 변환 후 npy 파일 저장
                    yuv_frames_np = self.rgb_to_yuv(face_frames_np)
                    yuv_output_file = os.path.join(yuv_output_dir, f"YUV_{subject_name}_input{npy_index}.npy")
                    np.save(yuv_output_file, yuv_frames_np)
                    yuv_files.append(yuv_output_file)
                    
                    if self.config['PREPROCESS']['RANDOM_FRAME']:
                        # 랜덤 프레임을 이미지로 저장
                        self.save_random_frames(face_frames, image_save_dir, subject_name, npy_index)
                    
                    face_frames = []
                    npy_index += 1

                pbar_frames.update(1)  # Update progress bar after processing each frame

        cap.release()

        return rgb_files, yuv_files, npy_index
    
    def save_random_frames(self, face_frames, save_dir, subject_name, npy_index):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_frames = len(face_frames)
        random_indices = random.sample(range(num_frames), min(1, num_frames))

        for i in random_indices:
            frame = face_frames[i]
            # OpenCV는 BGR 형식을 사용하므로, RGB에서 BGR로 변환하여 저장해야 함
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_file = os.path.join(save_dir, f"{subject_name}_input{npy_index}_frame{i}.png")
            cv2.imwrite(frame_file, frame_bgr)
            
    def diff_normalize_data(self, data):
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
            
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        
        return diffnormalized_data 
    
    def rgb_to_yuv(self, rgb_image):
        transformation_matrix = np.array([[0.299, 0.587, 0.114],
                                        [-0.169, -0.331, 0.5],
                                        [0.5, -0.419, -0.081]])
        offset = np.array([0, 128, 128])
        
        yuv_image = np.dot(rgb_image, transformation_matrix.T) + offset
        return yuv_image.astype(np.uint8)
