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
import time



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
        self.best_model_1_path = ''
        self.best_model_2_path = ''
        
        self.jump_number = config['ENSEMBLE']['NUM']
        self.seed_list = [42, 123, 789] # 456, 789, 101112]

        initial_seed = config['INITIAL_SEED']
        set_seed(initial_seed)
        self.criterion = RMSE_Loss_function()
        
        root = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir =  os.path.join(root,self.config['RUN']['DIR'])
        #"/home/neuroai/Downloads/ensenble/Ensemble/run" # 홈 디렉토리 기준으로 "run" 폴더 경로 설정
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # 존재하는 폴더 중 가장 큰 번호를 찾아 새로운 폴더 이름 설정
        existing_dirs = [int(d) for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) and d.isdigit()]
        if existing_dirs:
            self.new_dir_num = max(existing_dirs) + 1
        else:
            self.new_dir_num = 1
            
        self.new_dir = os.path.join(self.base_dir, str(self.new_dir_num))
        os.makedirs(self.new_dir)
        print(self.new_dir)
        
        self.both = config['PREPROCESS']['BOTH']
        self.using_model = config['PREPROCESS']['USING_MODEL']

        if config['TOOLBOX_MODE'] == 'train and test':
            if not config['PREPROCESS']['BOTH'] :
                self.model_num = 1
                if config['PREPROCESS']['USING_MODEL'] == 'Physformer':
                    print('Using Model: Physformer only')
                    self.model_1_name = 'Physformer'
                    self.model_1 = ViT_ST_ST_Compact3_TDC_gra_sharp(
                        image_size=(self.chunk_len, config['TRAIN']['RESIZE']['H'], config['TRAIN']['RESIZE']['W']), 
                        patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                        dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
                elif config['PREPROCESS']['USING_MODEL'] == 'Physnet':
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
                # del self.before_model_2
                self.optimizer_2 =  optim.AdamW(self.model_2.parameters(), lr = 0.0001, weight_decay=0.0001)
                self.scheduler_2 = optim.lr_scheduler.StepLR(self.optimizer_2, step_size=50, gamma=0.5)

            self.model_1 = torch.nn.DataParallel(self.model_1, device_ids=list(range(config['NUM_OF_GPU_TRAIN'])))
            # del self.before_model_1
            self.optimizer_1 =  optim.AdamW(self.model_1.parameters(), lr = 0.0001, weight_decay=0.0001)  
            self.scheduler_1 = optim.lr_scheduler.StepLR(self.optimizer_1, step_size=50, gamma=0.5)      
            self.num_train_batches = len(data_loader["train"])
            # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     self.optimizer, max_lr=0.00001, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)   
    
    def train(self, data_loader):
        torch.autograd.set_detect_anomaly(True)
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        
        # BP_pred = torch.zeros().to(self.device)
            
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====\n")
            
           #seed_list = [42]
            epoch_loss = []

            self.model_1.train()
            if self.both:
                self.model_2.train()
                
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                all_dbp_preds = []
                all_sbp_preds = []
                all_dbp_gt = []
                all_sbp_gt = []
                running_loss = 0.0
                training_loss = []
                RGB_data = batch[0].float().to(self.device)
                YUV_data = batch[1].float().to(self.device)
                bp_label = batch[2].float().to(self.device)
       
                combined_data = torch.cat((RGB_data, YUV_data), dim = 0)
                label= torch.cat((bp_label, bp_label), dim = 0)     

                 

                for seed_value in self.seed_list:
                    set_seed(seed_value)
                    BP_pred, _, _, _ = self.model_1(combined_data)
                
                    for i in range(BP_pred.shape[0]):
                        all_dbp_preds.append(BP_pred[i][0])
                        all_dbp_gt.append(label[i][0])
                        all_sbp_preds.append(BP_pred[i][1])
                        all_sbp_gt.append(label[i][1])

                    if self.both:
                        BP_pred, _, _, _ = self.model_2(combined_data)
                        for i in range(BP_pred.shape[0]):
                            all_dbp_preds.append(BP_pred[i][0])
                            all_dbp_gt.append(label[i][0])
                            all_sbp_preds.append(BP_pred[i][1])
                            all_sbp_gt.append(label[i][1])
                    del BP_pred

                    # print(f'\n\nsbp_pred:{all_sbp_preds}')
                    # print(f'\n\ndbp_pred:{all_dbp_preds}')
                    # print(f'\n\nsbp_gt:{all_sbp_gt}')
                    # print(f'\n\ndbp_gt:{all_dbp_gt}')

                    # time.sleep(5)

                # # all_dbp_gt.extend([label[:, 0]] * (2 * self.model_num * len(seed_list)))
                # all_sbp_gt.extend([label[:, 1]] * (2 * self.model_num * len(seed_list)))              
                del combined_data
                
                loss = self.criterion.loss_f(all_dbp_preds, all_dbp_gt, all_sbp_preds, all_sbp_gt)
                
                loss.backward()
                training_loss.append(loss.item())
                self.optimizer_1.step()
                self.optimizer_1.zero_grad()
                if self.both:
                    self.optimizer_2.step()
                    self.optimizer_2.zero_grad()

                epoch_loss.append(loss.item())

                # del RGB_data, YUV_data, label, BP_physformer, BP_physnet, combined_data
                
                # if idx % 100 == 99:
                #     print(f'loss : {loss.item():.4f}')
            mean_epoch_loss = np.mean(epoch_loss)
            print(f"Epoch [{epoch+1}/{self.max_epoch_num}], Loss: {loss.item():.4f}")
            mean_training_losses.append(mean_epoch_loss)
            
            # lrs.append(self.scheduler_1.get_last_lr())
            # Append the mean training loss for the epoch
            self.save_model(epoch)
            self.scheduler_1.step()
            self.model_1.eval()
            if self.both:
                self.scheduler_2.step()
                self.model_2.eval()
                
            if not self.config['TEST']['USE_LAST_EPOCH']: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print(f'Validation RMSE:{valid_loss:.3f}, batch:{idx+1}')
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_model_1_path = os.path.join(self.new_dir, f'{self.model_1_name}_Epoch{epoch}.pth')
                    if self.both:
                        self.best_model_2_path = os.path.join(self.new_dir, f'{self.model_2_name}_Epoch{epoch}.pth')
                    print("Update best model! Best epoch: {}".format(epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_model_1_path = os.path.join(self.new_dir, f'{self.model_1_name}_Epoch{epoch}.pth')
                    if self.both:
                        self.best_model_2_path = os.path.join(self.new_dir, f'{self.model_2_name}_Epoch{epoch}.pth')
                    print("Update best model! Best epoch: {}".format(epoch))
            
    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===\n")
        self.optimizer_1.zero_grad()
        if self.both:
            self.optimizer_2.zero_grad()
        with torch.no_grad():
            val_dbp_preds = []
            val_sbp_preds = []
            val_dbp_gt = []
            val_sbp_gt = []
            # BP_preds = torch.zeros().to(self.device)
            
            vbar = tqdm(data_loader["valid"], ncols=80)
            for val_idx, val_batch in enumerate(vbar):
                RGB_data = val_batch[0].float().to(self.device)
                YUV_data = val_batch[1].float().to(self.device)
                bp_label = val_batch[2].float().to(self.device)
                    
                combined_data = torch.cat((RGB_data, YUV_data), dim = 0)
                label= torch.cat((bp_label, bp_label), dim = 0)
                BP_pred, _, _, _ = self.model_1(combined_data)

                for i in range(BP_pred.shape[0]):
                    val_dbp_preds.append(BP_pred[i][0])
                    val_dbp_gt.append(label[i][0])
                    val_sbp_preds.append(BP_pred[i][1])
                    val_sbp_gt.append(label[i][1])

                if self.both:
                    BP_pred, _, _, _ = self.model_2(combined_data)
                    for i in range(BP_pred.shape[0]):
                        val_dbp_preds.append(BP_pred[i][0])
                        val_dbp_gt.append(label[i][0])
                        val_sbp_preds.append(BP_pred[i][1])
                        val_sbp_gt.append(label[i][1])
                
            RMSE = self.criterion.loss_f(val_dbp_preds, val_dbp_gt, val_sbp_preds, val_sbp_gt)
        return RMSE
    
    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===\n")
        # BP_pred = torch.zeros().to(self.device)
        
        if self.config['TEST']['USE_LAST_EPOCH']:
            print("Testing uses last epoch as non-pretrained model!")
            print(f"Best {self.model_1_name} path:{self.best_model_1_path}")
            self.model_1.load_state_dict(torch.load(self.best_model_1_path)).to(self.device)
            self.model_1.eval()
            if self.both:
                print(f"Best {self.model_2_name} path:{self.best_model_2_path}")
                self.model_2.load_state_dict(torch.load(self.best_model_2_path)).to(self.device)
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
                    combined_data = torch.cat((RGB_data[i].unsqueeze(0), YUV_data[i].unsqueeze(0)), dim=0)
                    label = torch.cat((bp_label[i].unsqueeze(0), bp_label[i].unsqueeze(0)), dim=0)

                    all_dbp_preds = []
                    all_sbp_preds = []
                    
                    for seed_value in self.seed_list:
                        set_seed(seed_value)
                        BP_pred, _, _, _ = self.model_1(combined_data)

                        for j in range(BP_pred.shape[0]):
                            all_dbp_preds.append(BP_pred[j][0].cpu().numpy())
                            all_sbp_preds.append(BP_pred[j][1].cpu().numpy())

                        if self.both:
                            BP_pred, _, _, _ = self.model_2(combined_data)
                            for j in range(BP_pred.shape[0]):
                                all_dbp_preds.append(BP_pred[j][0].cpu().numpy())
                                all_sbp_preds.append(BP_pred[j][1].cpu().numpy())

                aggregated_dbp = self.aggregate_predictions(all_dbp_preds)
                aggregated_sbp = self.aggregate_predictions(all_sbp_preds)
                
                # 각 데이터 쌍의 결과 저장
                batch_dbp_preds.append(aggregated_dbp)
                batch_sbp_preds.append(aggregated_sbp)
                batch_gt_dbp.append(label[:, 0].cpu().numpy()[0])
                batch_gt_sbp.append(label[:, 1].cpu().numpy()[0])

            # 배치 내 각 데이터 쌍의 결과를 전체 결과에 추가
            pred_dbp.extend(batch_dbp_preds)
            pred_sbp.extend(batch_sbp_preds)
            gt_dbp.extend(batch_gt_dbp)
            gt_sbp.extend(batch_gt_sbp)

        pred_dbp = [self.denormalize_bp_value(p, min_val=40, max_val=120) for p in pred_dbp]
        pred_sbp = [self.denormalize_bp_value(p, min_val=70, max_val=180) for p in pred_sbp]
        gt_dbp = [self.denormalize_bp_value(g, min_val=40, max_val=120) for g in gt_dbp]
        gt_sbp = [self.denormalize_bp_value(g, min_val=70, max_val=180) for g in gt_sbp]

        print('')
        print("\nSBP Metrics:")
        (sbp_mae, se_sbp_mae), (sbp_rmse, se_sbp_rmse) = metrics.calculate_all_metrics(pred_sbp, gt_sbp)
        
        print("\nDBP Metrics:")
        (dbp_mae, se_dbp_mae), (dbp_rmse, se_dbp_rmse) = metrics.calculate_all_metrics(pred_dbp, gt_dbp)


                    # After collecting all predictions, combine them into a single tensor for SBP and DBP
        #         if all_sbp_pred:
        #             all_sbp_pred = torch.cat(all_sbp_preds, dim=0)
        #         if all_dbp_pred:
        #             all_dbp_pred = torch.cat(all_dbp_preds, dim=0)
        #         aggregated_sbp = self.aggregate_predictions([p.cpu().numpy() for p in all_sbp_pred])
        #         aggregated_dbp = self.aggregate_predictions([p.cpu().numpy() for p in all_dbp_pred])
                
        #         # Append the aggregated results and ground truth
        #         pred_sbp.append(aggregated_sbp)
        #         pred_dbp.append(aggregated_dbp)
        #         gt_sbp.append(label[:, 1].cpu().numpy()[0])
        #         gt_dbp.append(label[:, 0].cpu().numpy()[0])

        
        # print('')
        # print("\nSBP Metrics:")
        # (sbp_mae, se_sbp_mae), (sbp_rmse, se_sbp_rmse) = metrics.calculate_all_metrics(pred_sbp, gt_sbp)
        
        # # DBP에 대한 메트릭 계산
        # print("\nDBP Metrics:")
        # (dbp_mae, se_dbp_mae), (dbp_rmse, se_dbp_rmse) = metrics.calculate_all_metrics(pred_dbp, gt_dbp)
        
        # SBP와 DBP 메트릭의 평균 계산
        avg_mae = (sbp_mae + dbp_mae) / 2
        avg_rmse = (sbp_rmse + dbp_rmse) / 2
        #avg_pearson = (sbp_pearson + dbp_pearson) / 2
        
        avg_se_mae = (se_sbp_mae + se_dbp_mae) / 2
        avg_se_rmse = (se_sbp_rmse + se_dbp_rmse) / 2
        #avg_se_pearson = (se_sbp_pearson + se_dbp_pearson) / 2
        
        print("\nAverage Metrics SBP and DBP:")
        print("Average MAE: {0:.4f} +/- {1:.4f}".format(avg_mae, avg_se_mae))
        print("Average RMSE: {0:.4f} +/- {1:.4f}".format(avg_rmse, avg_se_rmse))
       # print("Average Pearson Correlation: {0:.4f} +/- {1:.4f}".format(avg_pearson, avg_se_pearson))
          
    def save_model(self, index):
    # Ensure the directory exists before saving the models
        if not os.path.exists(self.new_dir):
            os.makedirs(self.new_dir)

        model_1_path= os.path.join(self.new_dir, f'{self.model_1_name}_Epoch{index}.pth')
        torch.save(self.model_1.state_dict(), model_1_path)
        print('Saved Model Paths: ',model_1_path)

        if self.both:
            model_2_path = os.path.join(self.new_dir, f'{self.model_2_name}_Epoch{index}.pth')
            torch.save(self.model_2.state_dict(), model_2_path)
            print('Saved Model Paths: ',model_2_path)
        
    def aggregate_predictions(self, predictions):
        n = 0 #self.jump_number
        predictions = [float(p) for p in predictions]
        # Sort the predictions in ascending order
        sorted_predictions = np.sort(predictions)
        # Remove the top-n and bottom-n values
        if n < 2:
            trimmed_predictions = sorted_predictions
        else:
            trimmed_predictions = sorted_predictions[n:-n]

        # Calculate the average of the remaining values
        aggregated_prediction = np.mean(trimmed_predictions)
        return aggregated_prediction
    
    def denormalize_bp_value(self, normalized_bp, min_val, max_val):
        """정규화된 BP 값을 역정규화하여 실제 값으로 변환"""
        return normalized_bp * (max_val - min_val) + min_val

    