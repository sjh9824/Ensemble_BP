import json
import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from trainer.trainer import Trainer
from config import get_config
import random

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

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

current_dir = os.path.dirname(os.path.abspath(__file__))

# config.yaml의 상대 경로 설정
config_file_path = os.path.join(current_dir, 'config.yaml')
config = load_config(config_file_path)
json_path = config['DATA']['JSON_PATH']
#'/home/neuroai/Downloads/ensenble/Ensemble/numpy_path_with_gt.json'
# Load and split data
train_samples, valid_samples, test_samples = load_and_split_data(json_path)

# Create DataLoaders
batch_size = config['TRAIN']['BATCH_SIZE']
num_workers = 16

train_dataset = CustomDataset(train_samples)
valid_dataset = CustomDataset(valid_samples)
test_dataset = CustomDataset(test_samples)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=train_generator
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=train_generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=train_generator
)

# Now you can pass these loaders to your Trainer
data_loaders = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}

# Initialize Trainer and start training
trainer = Trainer(config, data_loaders)
trainer.train(data_loaders)
trainer.test(data_loaders)