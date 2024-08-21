import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
import random

# class RMSE_Loss_function():
#     def loss_f(self, dbp_pred, dbp_gt, sbp_pred, sbp_gt):
#         # 각 리스트의 길이가 동일한지 확인
#         assert len(dbp_pred) == len(dbp_gt) == len(sbp_pred) == len(sbp_gt), "List lengths must be the same"
        
#         # DBP와 SBP에 대한 RMSE 계산
#         rmse_dbp_list = [torch.sqrt(torch.mean((pred - gt) ** 2)) for pred, gt in zip(dbp_pred, dbp_gt)]
#         rmse_sbp_list = [torch.sqrt(torch.mean((pred - gt) ** 2)) for pred, gt in zip(sbp_pred, sbp_gt)]
        
#         # 평균 RMSE 계산
#         avg_rmse_dbp = torch.mean(torch.stack(rmse_dbp_list))
#         avg_rmse_sbp = torch.mean(torch.stack(rmse_sbp_list))
        
#         # 최종 손실 계산 (DBP와 SBP의 평균 RMSE)
#         loss = 0.5 * avg_rmse_dbp + 0.5 * avg_rmse_sbp
        
#         return loss
# class RMSE_Loss_function():
    # def loss_f(self, dbp_pred, dbp_gt, sbp_pred, sbp_gt):
    #     # 각 리스트의 길이가 동일한지 확인
    #     assert len(dbp_pred) == len(dbp_gt) == len(sbp_pred) == len(sbp_gt), "List lengths must be the same"
        
    #     # DBP와 SBP에 대한 RMSE 계산
    #     rmse_dbp_list = []
    #     rmse_sbp_list = []
        
    #     for pred, gt in zip(dbp_pred, dbp_gt):
    #         diff = pred - gt
    #         squared_diff = diff ** 2
    #         mean_squared_diff = torch.mean(squared_diff)
    #         rmse = torch.sqrt(mean_squared_diff)
    #         rmse_dbp_list.append(rmse)

    #     for pred, gt in zip(sbp_pred, sbp_gt):
    #         diff = pred - gt
    #         squared_diff = diff ** 2
    #         mean_squared_diff = torch.mean(squared_diff)
    #         rmse = torch.sqrt(mean_squared_diff)
    #         rmse_sbp_list.append(rmse)
        
    #     # 평균 RMSE 계산
    #     avg_rmse_dbp = torch.mean(torch.stack(rmse_dbp_list))
    #     avg_rmse_sbp = torch.mean(torch.stack(rmse_sbp_list))
        
    #     # 최종 손실 계산 (DBP와 SBP의 평균 RMSE)
    #     loss = 0.5 * avg_rmse_dbp + 0.5 * avg_rmse_sbp
        
    #     return loss
class RMSE_Loss_function():
    def loss_f(self, dbp_pred, dbp_gt, sbp_pred, sbp_gt):
        # 모든 예측값과 실제값을 텐서로 변환
        dbp_pred = torch.stack(dbp_pred)
        dbp_gt = torch.stack(dbp_gt)
        sbp_pred = torch.stack(sbp_pred)
        sbp_gt = torch.stack(sbp_gt)
        
        # DBP와 SBP에 대한 RMSE 계산
        diff_dbp = dbp_pred - dbp_gt
        diff_sbp = sbp_pred - sbp_gt
        
        rmse_dbp = torch.sqrt(torch.mean(diff_dbp ** 2))
        rmse_sbp = torch.sqrt(torch.mean(diff_sbp ** 2))
        
        # 최종 손실 계산 (DBP와 SBP의 평균 RMSE)
        loss = 0.5 * rmse_dbp + 0.5 * rmse_sbp
        
        return loss

      
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False