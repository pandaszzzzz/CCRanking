import os
import os.path as osp
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from csrnet import CSRNet
from evaluate import evaluate_model
from dataset import *

rand_seed = 64678
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

down_sample = 8
#pretrained_model = '../results/eval_1_ShanghaiTechA_Combine_CSRNet/809_model_best.pth.tar'
#pretrained_model = '../results/eval_5_ShanghaiTechA_AllCombine_CSRNet/188_model_best.pth.tar'

#pretrained_model = '../results/22_1_UCF-QNRF_Combine_CSRNet/model_best.pth.tar'
pretrained_model = '../results/22_2_UCF-QNRF_AllCombine_CSRNet/model_best.pth.tar'
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
net = CSRNet(add_mode=False)
net.cuda()

print("=> loading checkpoint '{}'".format(pretrained_model))
checkpoint = torch.load(pretrained_model)
net.load_state_dict(checkpoint['state_dict'])

net.eval()
data = UCF(vis_mode=True, split_ratio=2, split_num=2000, down_sample=down_sample)
data_val = ShanghaiTech_eval(down_sample=down_sample)
data_compare = ShanghaiTech_compare(down_sample=down_sample)
# data = ShanghaiTech(vis_mode=True, split_ratio=2, split_num=500, down_sample=down_sample)
# data_val = ShanghaiTech_eval(down_sample=down_sample, is_A=False)
# data_compare = ShanghaiTech_compare(down_sample=down_sample, is_A=False)

data_loader = torch.utils.data.DataLoader(data,batch_size=1,shuffle=False)
data_loader_val = torch.utils.data.DataLoader(data_val,batch_size=1,shuffle=False)
data_loader_compare = torch.utils.data.DataLoader(data_compare,batch_size=1,shuffle=False)

with torch.no_grad():
    comp_nums = []
    comp_outs = []
    for blob in data_loader_compare:
        im_data = blob['im'].cuda()
        num = blob['num'][0].numpy()
        out, _ = net(im_data)
        out = out[0].cpu().numpy()
        comp_nums.append(num)
        comp_outs.append(out)
    X = np.array(comp_outs)
    y = np.array(comp_nums)
    X_b= np.c_[np.ones((len(X), 1)),X]
    linalg = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # stastical visualization
    print(linalg[0],linalg[1])
    plt.scatter(X, y, marker='*')
    Y = linalg[0] + linalg[1] * X
    plt.plot(X, Y, c='r')
    plt.show()

    # visualize density map
    for ii,blob in enumerate(data_loader):
        if ii == 893: 
            im1_data = blob['im1'].cuda()
            print(im1_data.shape)
            im2_data = blob['im2'].cuda()
            print(im2_data.shape)
        
            out1, dmap1 = net(im1_data)
            out2, dmap2 = net(im2_data)
            
            pred1 = out1.cpu().numpy().item()
            pred1 = linalg[0] + linalg[1] * pred1
            pred2 = out2.cpu().numpy().item()
            pred2 = linalg[0] + linalg[1] * pred2
            
            path1 = blob['path1']
            path2 = blob['path2']
            num1 = blob['num1'].numpy().item()
            num2 = blob['num2'].numpy().item()

            print(ii)
            print(num1, pred1, path1)
            print(num2, pred2, path2)

            # visualize density features
            plt.subplot(121)
            dmap1 = dmap1.squeeze(0).squeeze(0)
            #dmap1 = torch.clamp(dmap1,0,5000)
            print(torch.max(dmap1))
            plt.imshow(dmap1.cpu().numpy(), cmap='plasma_r')
            # plt.subplot(122)
            # dmap2 = dmap2.squeeze(0).squeeze(0)
            # plt.imshow(dmap2.cpu().numpy(), cmap='plasma_r')

            plt.show()
            sys.exit(0)
