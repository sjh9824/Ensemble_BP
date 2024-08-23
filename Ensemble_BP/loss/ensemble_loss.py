import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
import os
import random

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False