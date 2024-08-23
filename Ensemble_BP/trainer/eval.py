import os
import numpy as np
import math
import torch
import torch.optim as optim
import random
from model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from loss.ensemble_loss import RMSE_Loss_function, set_seed
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from .BaseTrainer import BaseTrainer
from torch.autograd import Variable
from evaluation import metrics,save_plot
from tqdm import tqdm
import time

class Evaluation(BaseTrainer):
    def __init__(self, config, data_loader,load_path,plot_path,seed_list, model_type, data_type):
        super().__init__()
        self.seed_list = seed_list # 456, 789, 101112]
        self.plot_path = plot_path
        self.model_type = model_type
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path) 
        print(self.plot_path)

        self.data_type = data_type

        self.weight_load_path = load_path
        self.device = torch.device(config['DEVICE'])
        self.max_epoch_num = config['TRAIN']['EPOCHS']
        self.dropout_rate = config['MODEL']['PHYSFORMER']['DROP_RATE']

        self.patch_size = config['MODEL']['PHYSFORMER']['PATCH_SIZE']
        self.dim = config['MODEL']['PHYSFORMER']['DIM']
        self.ff_dim = config['MODEL']['PHYSFORMER']['FF_DIM']
        self.num_heads = config['MODEL']['PHYSFORMER']['NUM_HEADS']
        self.num_layers = config['MODEL']['PHYSFORMER']['NUM_LAYERS']
        self.theta = config['MODEL']['PHYSFORMER']['THETA']

        self.batch_size = config['TRAIN']['BATCH_SIZE']
        self.num_of_gpu = config['NUM_OF_GPU_TRAIN']
        self.base_len = self.num_of_gpu

        self.chunk_len = config['PREPROCESS']['CHUNK_LENGTH']
        self.frame_rate = config['DATA']['FS']

        self.config = config 
        self.min_valid_loss = None
        self.best_model_1_path = ''
        self.best_model_2_path = ''
        
        self.jump_number = config['ENSEMBLE']['NUM']

        initial_seed = config['INITIAL_SEED']
        set_seed(initial_seed)
        self.criterion = RMSE_Loss_function()
        self.sbp_mae = 0.0
        self.dbp_mae = 0.0
        self.sbp_rmse = 0.0
        self.dbp_rmse = 0.0
        
        
        root = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir =  os.path.join(root,self.config['RUN']['DIR'])
        
        self.both = config['PREPROCESS']['BOTH']
        self.using_model = config['PREPROCESS']['USING_MODEL']
        self.model_type = model_type

        if config['TOOLBOX_MODE'] == 'train and test':
            if self.model_type != 'Both' :
                self.model_num = 1
                if self.model_type == 'Physformer':
                    print('Using Model: Physformer only')
                    self.model_1_name = 'Physformer'
                    self.model_1 = ViT_ST_ST_Compact3_TDC_gra_sharp(
                        image_size=(self.chunk_len, config['TRAIN']['RESIZE']['H'], config['TRAIN']['RESIZE']['W']), 
                        patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                        dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
                elif self.model_type == 'Physnet':
                    self.model_1_name = 'Physnet'
                    print('Using Model: Physnet only')
                    self.model_1 = PhysNet_padding_Encoder_Decoder_MAX(frames=config['MODEL']['PHYSNET']['FRAME_NUM']).to(self.device)

            else :
                self.model_num = 2
                self.model_1_name = 'Physformer'
                self.model_2_name = 'Physnet'
                print('Using Model: Physformer, Physnet')
                self.model_1 = ViT_ST_ST_Compact3_TDC_gra_sharp(
                    image_size=(self.chunk_len, config['TRAIN']['RESIZE']['H'], config['TRAIN']['RESIZE']['W']), 
                    patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                    dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
                self.model_2 = PhysNet_padding_Encoder_Decoder_MAX(frames=config['MODEL']['PHYSNET']['FRAME_NUM']).to(self.device)
                self.model_2 = torch.nn.DataParallel(self.model_2, device_ids=list(range(config['NUM_OF_GPU_TRAIN'])))
                self.optimizer_2 =  optim.AdamW(self.model_2.parameters(), lr = 0.0001, weight_decay=0.0001)
                self.scheduler_2 = optim.lr_scheduler.StepLR(self.optimizer_2, step_size=50, gamma=0.5)

            self.model_1 = torch.nn.DataParallel(self.model_1, device_ids=list(range(config['NUM_OF_GPU_TRAIN'])))
            self.optimizer_1 =  optim.AdamW(self.model_1.parameters(), lr = 0.0001, weight_decay=0.0001)  
            self.scheduler_1 = optim.lr_scheduler.StepLR(self.optimizer_1, step_size=50, gamma=0.5)      
            self.num_train_batches = len(data_loader["train"])

    def eval(self, data_loader):  
        if data_loader["test"] is None:
                raise ValueError("No data for test")
            
        print('')
        print("===Testing===\n")
        
        if not self.config['TEST']['USE_LAST_EPOCH']:
            print("Testing uses last epoch as non-pretrained model!")
            path_1 = self.model_1_name + '.pth'
            model_1_path = os.path.join(self.weight_load_path, path_1)
            print(f"Best {self.model_1_name} path:{model_1_path}")
            self.model_1.load_state_dict(torch.load(model_1_path))
            self.model_1.to(self.device)
            self.model_1.eval()
            if self.model_type == 'Both':
                path_2 = self.model_2_name + '.pth'
                model_2_path = os.path.join(self.weight_load_path,path_2)
                print(f"Best {self.model_2_name} path:{model_2_path}")
                self.model_2.load_state_dict(torch.load(model_2_path))
                self.model_2.to(self.device)
                self.model_2.eval()
        
        pred_sbp = []
        pred_dbp = []
        gt_sbp = []
        gt_dbp = []
        
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                
                batch_sbp_preds = []
                batch_dbp_preds = []
                batch_gt_sbp = []
                batch_gt_dbp = []
                
                RGB_data = test_batch[0].float().to(self.device)
                YUV_data = test_batch[1].float().to(self.device)
                bp_label = test_batch[2].float().to(self.device)

                for i in range(RGB_data.size(0)):  # 배치 내 각 데이터에 대해 독립적으로 처리
                    if self.data_type == 'Both':
                        combined_data = torch.cat((RGB_data[i].unsqueeze(0), YUV_data[i].unsqueeze(0)), dim=0)
                        label = torch.cat((bp_label[i].unsqueeze(0), bp_label[i].unsqueeze(0)), dim=0)
                    elif self.data_type == 'RGB':
                        combined_data = RGB_data
                        label = bp_label
                    elif self.data_type == 'YUV':
                        combined_data = YUV_data
                        label = bp_label
                    all_dbp_preds = []
                    all_sbp_preds = []
                    
                    BP_pred, _, _, _ = self.model_1(combined_data)

                    for j in range(BP_pred.shape[0]):
                        all_dbp_preds.append(BP_pred[j][0].cpu().numpy())
                        all_sbp_preds.append(BP_pred[j][1].cpu().numpy())

                    if self.model_type == 'Both':
                        BP_pred, _, _, _ = self.model_2(combined_data)
                        for j in range(BP_pred.shape[0]):
                            all_dbp_preds.append(BP_pred[j][0].cpu().numpy())
                            all_sbp_preds.append(BP_pred[j][1].cpu().numpy())        

                    aggregated_dbp = self.aggregate_predictions(all_dbp_preds)
                    aggregated_sbp = self.aggregate_predictions(all_sbp_preds)
                
                    batch_dbp_preds.append(aggregated_dbp)
                    batch_sbp_preds.append(aggregated_sbp)
                    batch_gt_dbp.append(label[:, 0].cpu().numpy()[0])
                    batch_gt_sbp.append(label[:, 1].cpu().numpy()[0])

                pred_dbp.extend(batch_dbp_preds)
                pred_sbp.extend(batch_sbp_preds)
                gt_dbp.extend(batch_gt_dbp)
                gt_sbp.extend(batch_gt_sbp)

        pred_dbp = [self.denormalize_bp_value(p, min_val=40, max_val=120) for p in pred_dbp]
        pred_sbp = [self.denormalize_bp_value(p, min_val=70, max_val=180) for p in pred_sbp]
        gt_dbp = [self.denormalize_bp_value(g, min_val=40, max_val=120) for g in gt_dbp]
        gt_sbp = [self.denormalize_bp_value(g, min_val=70, max_val=180) for g in gt_sbp]

        print('')
        print(f'{self.seed_list}')
        print("\nSBP Metrics:")
        (self.sbp_mae, se_sbp_mae), (self.sbp_rmse, se_sbp_rmse) = metrics.calculate_all_metrics(pred_sbp, gt_sbp)
        r_sbp, _ = pearsonr(np.array(gt_sbp), np.array(pred_sbp))

        print("\nDBP Metrics:")
        (self.dbp_mae, se_dbp_mae), (self.dbp_rmse, se_dbp_rmse) = metrics.calculate_all_metrics(pred_dbp, gt_dbp)
        r_dbp, _ = pearsonr(np.array(gt_dbp), np.array(pred_dbp))

        avg_mae = (self.sbp_mae + self.dbp_mae) / 2
        avg_rmse = (self.sbp_rmse + self.dbp_rmse) / 2
        avg_r = (r_dbp + r_sbp) / 2
        
        avg_se_mae = (se_sbp_mae + se_dbp_mae) / 2
        avg_se_rmse = (se_sbp_rmse + se_dbp_rmse) / 2
        
        print("\nAverage Metrics SBP and DBP:")
        print("Average MAE: {0} +/- {1}".format(avg_mae, avg_se_mae))
        print("Average RMSE: {0} +/- {1}".format(avg_rmse, avg_se_rmse))
        
        plot_name = self.model_type + '_' + self.data_type
        save_plot.scatter_plots(pred_sbp, pred_dbp, gt_sbp, gt_dbp, self.plot_path, plot_name= plot_name)
        
        return (self.sbp_mae, self.sbp_rmse, r_sbp), (self.dbp_mae, self.dbp_rmse, r_dbp) , (avg_mae, avg_rmse, avg_r)


    def aggregate_predictions(self, predictions):
        n = 0 #self.jump_number
        predictions = [float(p) for p in predictions]
        sorted_predictions = np.sort(predictions)
        if n < 2:
            trimmed_predictions = sorted_predictions
        else:
            trimmed_predictions = sorted_predictions[n:-n]

        aggregated_prediction = np.mean(trimmed_predictions)
        return aggregated_prediction
    
    def denormalize_bp_value(self, normalized_bp, min_val, max_val):
        """정규화된 BP 값을 역정규화하여 실제 값으로 변환"""
        return normalized_bp * (max_val - min_val) + min_val
    

    
