import numpy as np
import torch
import sys
import math
import tqdm
from utils import Logger, mkdir_if_missing

def evaluate_model(net, data_loader_val, data_loader_compare, rank, scaling_rate=100.):
    net.eval()
    maes = []
    mses = []

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

                maes.append(np.abs(num - pred))
                mses.append((num - pred)*(num - pred))
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
                    maes.append(np.abs(num - pred))
                    mses.append((num - pred)*(num - pred))
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

                    maes.append(np.abs(num - pred))
                    mses.append((num - pred)*(num - pred))
    mae = np.average(np.array(maes))
    mse = np.sqrt(np.average(np.array(mses)))
    return mae, mse
