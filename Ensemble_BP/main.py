import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from Data.Data_load_process import *
from trainer.trainer import Trainer
from trainer.eval import Evaluation
import random
import time
import gc

if __name__ == "__main__":
    RANDOM_SEED = 100
    general_generator, train_generator = setup_seed(RANDOM_SEED)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_file_path = os.path.join(current_dir, 'config.yaml')
    # config.yaml의 상대 경로 설정
    config = load_config(config_file_path)
    json_path = config['DATA']['JSON_PATH']
    
    if config['PREPROCESS']['DO']:
        # try:
        loader = DataProcess(config)
        print("========Data Processing started========")

        loader.data_process()  # 만약 이 메서드가 데이터를 처리하는 역할이라면 호출합니다.
        print("=======Data Processing completed=======")

        # except Exception as e:
        #     print(f"DataProcess 실행 중 오류가 발생했습니다: {e}")
            
        before_sort_dir = os.path.join(current_dir, "Data/input_numpy_path.json")        
        
        filter_and_save_json(before_sort_dir, json_path, config)
    
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
        shuffle=False, #True
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

    #Initialize Trainer and start training
    
    plot_path = os.path.join(current_dir, 'result')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    existing_dirs = [int(d) for d in os.listdir(plot_path) if os.path.isdir(os.path.join(plot_path, d)) and d.isdigit()]
    if existing_dirs:
        new_dir_num = max(existing_dirs) + 1
    else:
        new_dir_num = 1

    plot_path = os.path.join(plot_path,str(new_dir_num))
    os.makedirs(plot_path)

    print(f"====Total Epoch: {config['TRAIN']['EPOCHS']}")
    trainer = Trainer(config, data_loaders, plot_path)
    trainer.train(data_loaders)
    trainer.test(data_loaders) 
    del trainer
    gc.collect()