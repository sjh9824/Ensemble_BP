import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
import random

class RMSE_Loss_function(nn.Module):
    def __init__(self):
        super(RMSE_Loss_function, self).__init__()

    def forward(self, dbp_pred, dbp_gt, sbp_pred, sbp_gt):
        # 각 리스트의 길이가 동일한지 확인
        assert len(dbp_pred) == len(dbp_gt) == len(sbp_pred) == len(sbp_gt), "List lengths must be the same"
        
        # DBP와 SBP에 대한 RMSE 계산
        rmse_dbp_list = [torch.sqrt(torch.mean((pred - gt) ** 2)) for pred, gt in zip(dbp_pred, dbp_gt)]
        rmse_sbp_list = [torch.sqrt(torch.mean((pred - gt) ** 2)) for pred, gt in zip(sbp_pred, sbp_gt)]
        
        # 평균 RMSE 계산
        avg_rmse_dbp = torch.mean(torch.stack(rmse_dbp_list))
        avg_rmse_sbp = torch.mean(torch.stack(rmse_sbp_list))
        
        # 최종 손실 계산 (DBP와 SBP의 평균 RMSE)
        loss = 0.5 * avg_rmse_dbp + 0.5 * avg_rmse_sbp
        return loss
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False