import os
import os.path as osp
import torch
import numpy as np
import sys
from csrnet import CSRNet
from evaluate import evaluate_model
from dataset import *

rand_seed = 64678
if rand_seed is not None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

EVAL_MODE = 'Rank' #  None | Rank | NoCompare
#pretrained_model = '../results/eval_1_ShanghaiTechA_Combine_CSRNet/809_model_best.pth.tar'
#pretrained_model = '../results/eval_2_ShanghaiTechA_Reg_CSRNet/688_model_best.pth.tar'
#pretrained_model = '../results/eval_5_ShanghaiTechA_AllCombine_CSRNet/188_model_best.pth.tar'
#pretrained_model = '../results/eval_122_ShanghaiTechA_CSRNet/342_model_best.pth.tar'
#pretrained_model = '../results/eval_13_ShanghaiTechA_Combine_CSRNet/model_best.pth.tar'
#pretrained_model = '../results/eval_14_ShanghaiTechA_Combine_CSRNet/model_best.pth.tar'
#pretrained_model = '../results/eval_15_ShanghaiTechA_Combine_CSRNet/model_best.pth.tar'
#pretrained_model = '../results/22_1_UCF-QNRF_Combine_CSRNet/model_best.pth.tar'
pretrained_model = '../results/22_2_UCF-QNRF_AllCombine_CSRNet/model_best.pth.tar'
down_sample = 8
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

net = CSRNet(add_mode=False)
net.cuda()

print("=> loading checkpoint '{}'".format(pretrained_model))
checkpoint = torch.load(pretrained_model)
net.load_state_dict(checkpoint['state_dict'])
#data_val = UCF_eval(down_sample=down_sample)
data_val = ShanghaiTech_eval(down_sample=down_sample, is_A=True)
data_compare = UCF_compare(down_sample=down_sample)
#data_compare = ShanghaiTech_compare(down_sample=down_sample, is_A=True, compare_nums=50)

if EVAL_MODE == 'None':
    is_rank = False
elif EVAL_MODE == 'Rank':
    is_rank = True
elif EVAL_MODE == 'NoCompare':
    is_rank = None
else:
    raise AssertionError("Invalid Evaluation Mode!")

data_loader_val = torch.utils.data.DataLoader(data_val,batch_size=1,shuffle=False)
data_loader_compare = torch.utils.data.DataLoader(data_compare,batch_size=1,shuffle=False)

mae_score, mse_score = evaluate_model(net, data_loader_val, data_loader_compare, is_rank)
print(EVAL_MODE+": ", mae_score, mse_score)
