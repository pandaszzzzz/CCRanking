import os
import os.path as osp
import torch
import numpy as np
import sys
from csrnet import CSRNet
from dataset import *
import math
import tqdm

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
#pretrained_model = '../results/eval_2_1_ShanghaiTechA_Reg_CSRNet/model_best.pth.tar'
#pretrained_model = '../results/22_1_UCF-QNRF_Combine_CSRNet/model_best.pth.tar'
pretrained_model = '../results/22_2_UCF-QNRF_AllCombine_CSRNet/model_best.pth.tar'

down_sample = 8
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

net = CSRNet(add_mode=False)
net.cuda()

print("=> loading checkpoint '{}'".format(pretrained_model))
checkpoint = torch.load(pretrained_model)
net.load_state_dict(checkpoint['state_dict'])

#data_val = ShanghaiTech_eval(down_sample=down_sample, is_A=True)
#data_compare = ShanghaiTech_compare(down_sample=down_sample, is_A=True)

data_compare = UCF_compare(down_sample=down_sample)
data_val = UCF_eval(down_sample=down_sample)

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

def evaluate_model(net, data_loader_val, data_loader_compare, rank, scaling_rate=100):
    net.eval()
    mae_accs = []

    comp_nums = []
    comp_outs = []
    with torch.no_grad():
        if rank is None:
            for blob in tqdm.tqdm(data_loader_val):
                im_data = blob['im'].cuda()
                num = blob['num'][0].numpy()

                out, _ = net(im_data)
                out = out[0].cpu().numpy()
                pred = out * scaling_rate

                mae_accs.append((num, np.abs(min(num, pred))/max(num, pred)))
        else:
            for blob in data_loader_compare:
                im_data = blob['im'].cuda()
                num = blob['num'][0].numpy()
                out, _ = net(im_data)
                out = out[0].cpu().numpy()
                comp_nums.append(num)
                comp_outs.append(out)
            X = np.array(comp_outs)
            y = np.array(comp_nums)
            if rank:
                arg_X = np.argsort(X.squeeze())
                rank_X = arg_X.copy()
                for i,e in enumerate(arg_X):
                    rank_X[e] = i
                arg_y = np.argsort(y.squeeze())
                rank_y = arg_y.copy()
                for i,e in enumerate(arg_y):
                    rank_y[e] = i
                X_b = np.c_[np.ones((len(rank_X), 1)),rank_X.reshape(-1,1)]
                linalg = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(rank_y.reshape(-1,1))
                print(linalg[0], linalg[1])

                for blob in tqdm.tqdm(data_loader_val):

                    im_data = blob['im'].cuda()
                    num = blob['num'][0].numpy()

                    out, _ = net(im_data)
                    out = out[0].cpu().numpy()
                    if out < min(X.squeeze()):
                        out = min(X.squeeze())
                    if out > max(X.squeeze()):
                        out = max(X.squeeze())
                    min_idx = 0
                    max_idx = 0
                    min_diff = 10000
                    max_diff = 10000
                    for i,e in enumerate(X.squeeze()):
                        if e < out:
                            if out - e < min_diff:
                                min_diff = out - e
                                min_idx = i
                        elif e > out:
                            if e - out < max_diff:
                                max_diff = e - out
                                max_idx = i
                        else:
                            min_idx = max_idx = i
                            min_diff = max_diff = 0
                            break
                    min_ = out - min_diff
                    max_ = out + max_diff
                    min_rank = rank_X[min_idx]
                    max_rank = rank_X[max_idx]
                    if min_idx != max_idx:
                        p = (out - min_)/(max_ - min_)
                        out_rank = p * min_rank + (1 - p) * max_rank
                    else:
                        out_rank = min_rank
                    pred_rank = linalg[1] * out_rank + linalg[0]
                    y.sort()
                    if pred_rank < min(rank_y):
                        pred = y[np.argmin(rank_y)]
                    elif pred_rank > max(rank_y):
                        pred = y[np.argmax(rank_y)]
                    else:
                        min_y_idx = math.floor(pred_rank)
                        max_y_idx = math.ceil(pred_rank)
                        pred = y[min_y_idx] * (max_y_idx - pred_rank)+ y[max_y_idx] * (pred_rank - min_y_idx)
                    #print(pred_rank, y[min_y_idx], y[max_y_idx], pred, num)
                    mae_accs.append((num, np.abs(min(num, pred))/max(num, pred)))
            else:
                X_b = np.c_[np.ones((len(X), 1)),X]
                linalg = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
                print(linalg[0],linalg[1])
                for blob in tqdm.tqdm(data_loader_val):

                    im_data = blob['im'].cuda()
                    num = blob['num'][0].numpy()

                    out, _ = net(im_data)
                    out = out[0].cpu().numpy()
                    pred = linalg[1] * out + linalg[0]

                    mae_accs.append((num, np.abs(min(num, pred))/max(num, pred)))

    acc_0_200 = []
    acc_200_400 = []
    acc_400_700 = []
    acc_700_1000 = []
    acc_1000_1500 = []
    acc_1500_3000 = []
    acc_3000_ = []
    for mae in mae_accs:
        if mae[0] >= 0 and mae[0] < 200:
            acc_0_200.append(mae[1])
        elif mae[0] >= 200 and mae[0] < 400:
            acc_200_400.append(mae[1])
        elif mae[0] >= 400 and mae[0] < 700:
            acc_400_700.append(mae[1])
        elif mae[0] >= 700 and mae[0] < 1000:
            acc_700_1000.append(mae[1])
        elif mae[0] >= 1000 and mae[0] < 1500:
            acc_1000_1500.append(mae[1])
        elif mae[0] >= 1500 and mae[0] < 3000:
            acc_1500_3000.append(mae[1])
        else:
            acc_3000_.append(mae[1])
    print(len(acc_0_200), len(acc_200_400), len(acc_400_700), len(acc_700_1000), len(acc_1000_1500), len(acc_1500_3000), len(acc_3000_))
    acc_0_200 = np.average(np.array(acc_0_200))
    acc_200_400 = np.average(np.array(acc_200_400))
    acc_400_700 = np.average(np.array(acc_400_700))
    acc_700_1000 = np.average(np.array(acc_700_1000))
    acc_1000_1500 = np.average(np.array(acc_1000_1500))
    acc_1500_3000 = np.average(np.array(acc_1500_3000))
    acc_3000_ = np.average(np.array(acc_3000_))
    print(acc_0_200, acc_200_400, acc_400_700, acc_700_1000, acc_1000_1500, acc_1500_3000, acc_3000_)

evaluate_model(net, data_loader_val, data_loader_compare, is_rank)
