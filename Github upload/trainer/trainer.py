import os
import numpy as np
import math
import torch
import torch.optim as optim
import random
from model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from loss.ensemble_loss import RMSE_Loss_function, set_seed
from .BaseTrainer import BaseTrainer
from torch.autograd import Variable
from evaluation import metrics
from tqdm import tqdm



class Trainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config['DEVICE'])
        self.max_epoch_num = config['TRAIN']['EPOCHS']
        #self.model_dir = config['MODEL'].get('MODEL_DIR', None)  # MODEL_DIR가 없을 수도 있으므로 get 사용
        self.dropout_rate = config['MODEL']['PHYSFORMER']['DROP_RATE']

        self.patch_size = config['MODEL']['PHYSFORMER']['PATCH_SIZE']
        self.dim = config['MODEL']['PHYSFORMER']['DIM']
        self.ff_dim = config['MODEL']['PHYSFORMER']['FF_DIM']
        self.num_heads = config['MODEL']['PHYSFORMER']['NUM_HEADS']
        self.num_layers = config['MODEL']['PHYSFORMER']['NUM_LAYERS']
        self.theta = config['MODEL']['PHYSFORMER']['THETA']

        #self.model_file_name = config['TRAIN']['MODEL_FILE_NAME']
        self.batch_size = config['TRAIN']['BATCH_SIZE']
        self.num_of_gpu = config['NUM_OF_GPU_TRAIN']
        self.base_len = self.num_of_gpu

        self.chunk_len = config['PREPROCESS']['CHUNK_LENGTH']
        self.frame_rate = config['DATA']['FS']

        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0
        self.best_model_physformer_path = ''
        self.best_model_phynet_path = ''
        
        self.jump_number = config['ENSEMBLE']['NUM']

        initial_seed = config['INITIAL_SEED']
        set_seed(initial_seed)
        self.criterion = RMSE_Loss_function()
        
        self.base_dir = config['RUN']['DIR']
        #"/home/neuroai/Downloads/ensenble/Ensemble/run" # 홈 디렉토리 기준으로 "run" 폴더 경로 설정
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # 존재하는 폴더 중 가장 큰 번호를 찾아 새로운 폴더 이름 설정
        existing_dirs = [int(d) for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) and d.isdigit()]
        if existing_dirs:
            self.new_dir_num = max(existing_dirs) + 1
        else:
            self.new_dir_num = 1
        
        # 새 폴더 생성
        self.new_dir = os.path.join(self.base_dir, str(self.new_dir_num))
        os.makedirs(self.new_dir)
        
        
        if config['TOOLBOX_MODE'] == 'train and test':
            self.model_p = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(self.chunk_len, config['TRAIN']['RESIZE']['H'], config['TRAIN']['RESIZE']['W']), 
                patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
            self.model_physformer = torch.nn.DataParallel(self.model_p, device_ids=list(range(config['NUM_OF_GPU_TRAIN'])))
            
            self.model_physnet = PhysNet_padding_Encoder_Decoder_MAX(
                                frames=config['MODEL']['PHYSNET']['FRAME_NUM']).to(self.device)
            
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.AdamW(self.model_physformer.parameters(), lr=0.00001, weight_decay=0.00001)
            self.optimizer = optim.AdamW(self.model_physnet.parameters(), lr=0.00001, weight_decay=0.00001)
            
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
            # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     self.optimizer, max_lr=0.00001, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
    
    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            
            seed_list = [42, 123, 456, 789, 101112]
            
            train_loss = []
            
            all_dbp_preds = []
            all_sbp_preds = []
            all_dbp_gt = []
            all_sbp_gt = []
            
            epoch_loss = []
            
            self.model_physformer.train()
            self.model_physnet.train()
                
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                running_loss = 0.0
                training_loss = []
                tbar = tqdm(data_loader["train"], ncols=80)
                for seed_value in seed_list:
                    print(f"Training with random seed: {seed_value}")
                    set_seed(seed_value)
                    
                    RGB_data = batch[0].float().to(self.device)
                    YUV_data = batch[1].float().to(self.device)
                    label = batch[2].float().to(self.device)
                    
                    self.optimizer.zero_grad()
                    gra_sharp = 2.0
                    
                    RGB_BP_physformer, _, _, _ = self.model_physformer(RGB_data, gra_sharp)
                    RGB_BP_physnet, _, _, _ = self.model_physnet(RGB_data)
                    YUV_BP_physformer, _, _, _ = self.model_physformer(YUV_data,gra_sharp)
                    YUV_BP_physnet, _, _, _ = self.model_physnet(YUV_data)

                    all_dbp_preds.extend([RGB_BP_physformer[:, 0], RGB_BP_physnet[:, 0], YUV_BP_physformer[:, 0], YUV_BP_physnet[:, 0]])
                    all_sbp_preds.extend([RGB_BP_physformer[:, 1], RGB_BP_physnet[:, 1], YUV_BP_physformer[:, 1], YUV_BP_physnet[:, 1]])
                    all_dbp_gt.extend([label[:, 0]] * 4)
                    all_sbp_gt.extend([label[:, 1]] * 4)
                    
                loss = self.criterion(all_dbp_preds, all_dbp_gt, all_sbp_preds, all_sbp_gt)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                training_loss.append(loss.item())
                
                self.optimizer.step()
                self.scheduler.step()
                
                if idx % 100 == 99:
                    print(f'loss : {loss.item():.4f}')
                    
            epoch_loss.append(np.mean(training_loss))
            print(f"Epoch [{epoch+1}/{self.max_epoch_num}], Loss: {loss.item():.4f}")
        
            lrs.append(self.scheduler.get_last_lr())
            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(loss))
            self.save_model(epoch)
            self.scheduler.step()
            self.model_physnet.eval()
            self.model_physformer.eval()
            
            if not self.config['TEST']['USE_LAST_EPOCH']: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print(f'Validation RMSE:{valid_loss:.3f}, batch:{idx+1}')
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config['TEST']['USE_LAST_EPOCH']:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        # if self.config.TRAIN.PLOT_LOSSES_AND_LR:
        #    self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)
            
    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        self.optimizer.zero_grad()
        with torch.no_grad():
            val_dbp_preds = []
            val_sbp_preds = []
            val_dbp_gt = []
            val_sbp_gt = []
            
            vbar = tqdm(data_loader["valid"], ncols=80)
            for val_idx, val_batch in enumerate(vbar):
                RGB_data = val_batch[0].float().to(self.device)
                YUV_data = val_batch[1].float().to(self.device)
                label = val_batch[2].float().to(self.device)
                
                gra_sharp = 2.0
                RGB_BP_physformer, _, _, _ = self.model_physformer(RGB_data, gra_sharp)
                RGB_BP_physnet, _, _, _ = self.model_physnet(RGB_data)
                YUV_BP_physformer, _, _, _ = self.model_physformer(YUV_data,gra_sharp)
                YUV_BP_physnet, _, _, _ = self.model_physnet(YUV_data)
                
                val_dbp_preds.extend([RGB_BP_physformer[:, 0], RGB_BP_physnet[:, 0], YUV_BP_physformer[:, 0], YUV_BP_physnet[:, 0]])
                val_sbp_preds.extend([RGB_BP_physformer[:, 1], RGB_BP_physnet[:, 1], YUV_BP_physformer[:, 1], YUV_BP_physnet[:, 1]])
                val_dbp_gt.extend([label[:, 0]] * 4)
                val_sbp_gt.extend([label[:, 1]] * 4)
            #############################################
            RMSE = self.criterion(val_dbp_preds, val_dbp_gt, val_sbp_preds, val_sbp_gt)
            ##############################################
        return RMSE
    
    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        
        if self.config['TEST']['USE_LAST_EPOCH']:
            print("Testing uses last epoch as non-pretrained model!")
            print(self.best_model_physformer_path)
            print(self.best_model_physformer_path)
            self.model_physnet.load_state_dict(torch.load(self.best_model_phynet_path))
            self.model_physformer.load_state_dict(torch.load(self.best_model_physformer_path))
        # else:
        #     best_model_path = os.path.join(
        #         self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
        #     print("Testing uses best epoch selected using model selection as non-pretrained model!")
        #     print(best_model_path)
        #     self.model.load_state_dict(torch.load(best_model_path))

        self.model_physnet = self.model_physnet.to(self.config['DEVICE'])
        self.model_phyformer = self.model_phyformer.to(self.config['DEVICE'])
        self.model_physformer.eval()
        self.model_physnet.eval()
        
        pred_sbp = []
        pred_dbp = []
        gt_sbp = []
        gt_dbp = []
        
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                
                all_sbp_pred = []
                all_dbp_pred = []
                
                batch_size = test_batch[0].shape[0]
                RGB_data = test_batch[0].float().to(self.device)
                YUV_data = test_batch[1].float().to(self.device)
                label = test_batch[2].float().to(self.device)
                gra_sharp = 2.0
                
                RGB_BP_physformer, _, _, _ = self.model_physformer(RGB_data, gra_sharp)
                RGB_BP_physnet, _, _, _ = self.model_physnet(RGB_data)
                YUV_BP_physformer, _, _, _ = self.model_physformer(YUV_data,gra_sharp)
                YUV_BP_physnet, _, _, _ = self.model_physnet(YUV_data)
                
                all_sbp_pred.extend([RGB_BP_physformer[:, 0], RGB_BP_physnet[:, 0], YUV_BP_physformer[:, 0], YUV_BP_physnet[:, 0]])
                all_dbp_pred.extend([RGB_BP_physformer[:, 1], RGB_BP_physnet[:, 1], YUV_BP_physformer[:, 1], YUV_BP_physnet[:, 1]])
                
                # After collecting all predictions, combine them into a single tensor for SBP and DBP
                all_sbp_pred = torch.cat(all_sbp_pred, dim=0)
                all_dbp_pred = torch.cat(all_dbp_pred, dim=0)
                
                pred_sbp.append(self.aggregate_predictions(all_sbp_pred, self.jump_number))
                pred_dbp.append(self.aggregate_predictions(all_dbp_pred, self.jump_number))
                gt_sbp.extend([label[:, 0]] * 4)
                gt_dbp.extend([label[:, 1]] * 4)
                

        print('')
        print("\nSBP Metrics:")
        (sbp_mae, se_sbp_mae), (sbp_rmse, se_sbp_rmse), (sbp_pearson, se_sbp_pearson) = metrics.calculate_all_metrics(pred_sbp, gt_sbp)
        
        # DBP에 대한 메트릭 계산
        print("\nDBP Metrics:")
        (dbp_mae, se_dbp_mae), (dbp_rmse, se_dbp_rmse), (dbp_pearson, se_dbp_pearson) = metrics.calculate_all_metrics(pred_dbp, gt_dbp)
        
        # SBP와 DBP 메트릭의 평균 계산
        avg_mae = (sbp_mae + dbp_mae) / 2
        avg_rmse = (sbp_rmse + dbp_rmse) / 2
        avg_pearson = (sbp_pearson + dbp_pearson) / 2
        
        avg_se_mae = (se_sbp_mae + se_dbp_mae) / 2
        avg_se_rmse = (se_sbp_rmse + se_dbp_rmse) / 2
        avg_se_pearson = (se_sbp_pearson + se_dbp_pearson) / 2
        
        print("\nAverage Metrics (SBP and DBP):")
        print("Average MAE: {0:.4f} +/- {1:.4f}".format(avg_mae, avg_se_mae))
        print("Average RMSE: {0:.4f} +/- {1:.4f}".format(avg_rmse, avg_se_rmse))
        print("Average Pearson Correlation: {0:.4f} +/- {1:.4f}".format(avg_pearson, avg_se_pearson))
          
    def save_model(self, index):
        model_path = os.path.join(self.new_dir + 'Physformer_', f'Epoch{index}.pth')
        torch.save(self.model_physformer.state_dict(), model_path)
        self.best_model_physformer_path = model_path
        model_path = os.path.join(self.new_dir + 'Physnet_', f'Epoch{index}.pth')
        torch.save(self.model_physnet.state_dict(), model_path)
        self.best_model_physnet_path = model_path
        print('Saved Model Path: ', model_path)
        
    def aggregate_predictions(self, predictions, n):
        if len(predictions) < 2 * n:
            raise ValueError("The number of predictions should be greater than 2*n.")
        # Sort the predictions in ascending order
        sorted_predictions = np.sort(predictions)
        # Remove the top-n and bottom-n values
        trimmed_predictions = sorted_predictions[n:-n]
        # Calculate the average of the remaining values
        aggregated_prediction = np.mean(trimmed_predictions)
        return aggregated_prediction
                
                
                
                
            
        
