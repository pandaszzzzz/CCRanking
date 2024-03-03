import os
import os.path as osp
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
from utils import Timer
from torch.autograd import Variable
from utils import save_checkpoint, save_D_checkpoint
from utils import mkdir_if_missing
from dataset import *
from csrnet import CSRNet, ResBranch
from evaluate import evaluate_model
import argparse

parser = argparse.ArgumentParser(description='Train crowd counting network')
parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float, help="initial learning rate")
parser.add_argument('--lr_D', '--learning-rate-D', default=0.001, type=float, help="initial learning rate")
parser.add_argument('--split_ratio', default=2., type=float, help="set split ratio as 2: difference in the number of people is at least twice as large")
parser.add_argument('--split_num', default=1000, type=int, help="set split num as 500: more than 500 is considered more")
parser.add_argument('--down_sample', default=8, type=int, help="set network downsampling ratio")

parser.add_argument('--vis_mode', default=False, action='store_true', help="use visualization mode, default False")
parser.add_argument('--compare_loss_mode', default=False, action='store_true', help="use compare loss, default False")
parser.add_argument('--add_loss_mode', default=False, action='store_true', help="use sum loss, default False")
parser.add_argument('--rank_loss_mode', default=False, action='store_true', help='use rank loss, default False')
parser.add_argument('--eval_mode', default='Rank', type=str, help="None | Rank | NoCompare")

parser.add_argument('--scaling_rate', default=100, type=float, help="result ratio")

parser.add_argument('--lambda_reg', default=1., type=float)
parser.add_argument('--lambda_comp', default=1., type=float)
parser.add_argument('--lambda_summ', default=1., type=float)
parser.add_argument('--lambda_rank', default=0.2, type=float)
parser.add_argument('--lambda_soft', default=0.5, type=float)
parser.add_argument('--margin_comp', default=0.5, type=float, help="note if adaptive margin mode, it is initial margin")
parser.add_argument('--margin_rank', default=0.2, type=float)

parser.add_argument('--dataset_name', default='ShanghaiTechA', type=str, help="ShanghaiTechA | ShanghaiTechA_JHU | ShanghaiTechA_JHU_Baseline | ShanghaiTechA_JHU_Combine")
parser.add_argument('--model_name', default='CSRNet', type=str)
parser.add_argument('--start_step', default=0, type=int)
parser.add_argument('--end_step', default=500, type=int, help="ShanghaiTechA step = 500 | ShanghaiTechA_JHU step = 70")
parser.add_argument('--eval_step', default=2, type=int)

parser.add_argument('--seed', default=64678, type=int)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--experiment-ID', default='0', type=str, help="ID of the experiment")

# for adaptive
parser.add_argument('--res_mode', default=False, action='store_true', help='use residual branch')
parser.add_argument('--load_checkpoint', default='', type=str, help="pre-trained model for adaptive loss")
parser.add_argument('--adaptive_ratio', default=0.4, type=float)
args = parser.parse_args()

lr = args.lr
lr_D = args.lr_D
split_ratio = args.split_ratio
split_num = args.split_num
down_sample = args.down_sample
seed = args.seed
beta1 = 0.9
beta2 = 0.999

################################################# EXPERIMENTAL MODE #################################################
VIS_MODE = args.vis_mode
COMP_LOSS_MODE = args.compare_loss_mode
ADD_LOSS_MODE = args.add_loss_mode
RANK_LOSS_MODE = args.rank_loss_mode
EVAL_MODE = args.eval_mode

scaling_rate = args.scaling_rate
margin_comp = args.margin_comp
margin_rank = args.margin_rank
lambda_reg = args.lambda_reg
lambda_comp = args.lambda_comp
lambda_summ = args.lambda_summ
lambda_rank = args.lambda_rank
lambda_soft = args.lambda_soft
# for adaptive
adaptive_ratio = args.adaptive_ratio
RES_MODE = args.res_mode
#####################################################################################################################

batch_size = 1  #only support batch_size = 1
dataset_name = args.dataset_name
resume = None
start_step = args.start_step
end_step = args.end_step
eval_step = args.eval_step
model_name = args.model_name
experiment_id = args.experiment_ID

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

rand_seed = seed
if rand_seed is not None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
torch.backends.cudnn.deterministic = True

# log frequency
disp_interval = 50
t = Timer()
t.tic()

pretrained_model = args.load_checkpoint
net_infer = CSRNet(add_mode=ADD_LOSS_MODE)
net_infer.cuda()
net = CSRNet(add_mode=ADD_LOSS_MODE, res_mode=RES_MODE)
net.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas = (beta1, beta2))
if osp.isfile(pretrained_model):
    print("=> loading checkpoint '{}'".format(pretrained_model))
    checkpoint = torch.load(pretrained_model)
    net_infer.load_state_dict(checkpoint['state_dict'])
    if not RES_MODE:
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loaded checkpoint '{}' (epoch {})".format(pretrained_model, checkpoint['epoch']))

else:
    print("=> no checkpoint found at '{}'".format(pretrained_model))

if RES_MODE:
    net_res = ResBranch() # to do
    net_res.cuda()
    optimizer_res = torch.optim.Adam(filter(lambda p: p.requires_grad, net_res.parameters()), lr=lr, betas = (beta1, beta2))

if EVAL_MODE == 'None':
    is_rank = False
elif EVAL_MODE == 'Rank':
    is_rank = True
elif EVAL_MODE == 'NoCompare':
    is_rank = None
else:
    raise AssertionError("Invalid Evaluation Mode!")

if dataset_name == 'JHU_Combine':
    data = JHU_combine(vis_mode=VIS_MODE,
                        add_err=ADD_LOSS_MODE,rank_err=RANK_LOSS_MODE,
                        split_ratio=split_ratio, split_num=split_num, down_sample=down_sample)
    data_compare = JHU_compare(down_sample=down_sample)             
elif dataset_name == 'ShanghaiTechA_AllCombine':
    data = ShanghaiTech_allcombine(vis_mode=VIS_MODE,
                        add_err=ADD_LOSS_MODE,rank_err=RANK_LOSS_MODE,
                        split_ratio=split_ratio, split_num=split_num, down_sample=down_sample)
    data_compare = ShanghaiTech_compare(down_sample=down_sample)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample)
elif dataset_name == 'ShanghaiTechA_Combine':
    data = ShanghaiTech_combine(vis_mode=VIS_MODE,
                        add_err=ADD_LOSS_MODE,rank_err=RANK_LOSS_MODE,
                        split_ratio=split_ratio, split_num=split_num, down_sample=down_sample)
    data_compare = ShanghaiTech_compare(down_sample=down_sample)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample)
elif dataset_name == 'ShanghaiTechA_Reg':
    data = ShanghaiTech_reg(down_sample=down_sample)
    data_compare = ShanghaiTech_compare(down_sample=down_sample)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample)
elif dataset_name == 'ShanghaiTechB_AllCombine':
    data = ShanghaiTech_allcombine(vis_mode=VIS_MODE,
                        add_err=ADD_LOSS_MODE,rank_err=RANK_LOSS_MODE,
                        split_ratio=split_ratio, split_num=split_num, down_sample=down_sample, is_A=False)
    data_compare = ShanghaiTech_compare(down_sample=down_sample, is_A=False)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample, is_A=False)
elif dataset_name == 'ShanghaiTechB_Combine':
    data = ShanghaiTech_combine(vis_mode=VIS_MODE,
                        add_err=ADD_LOSS_MODE,rank_err=RANK_LOSS_MODE,
                        split_ratio=split_ratio, split_num=split_num, down_sample=down_sample, is_A=False)
    data_compare = ShanghaiTech_compare(down_sample=down_sample, is_A=False)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample, is_A=False)
elif dataset_name == 'ShanghaiTechB_Reg':
    data = ShanghaiTech_reg(down_sample=down_sample, is_A=False)
    data_compare = ShanghaiTech_compare(down_sample=down_sample, is_A=False)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample, is_A=False)
else:
    raise AssertionError("Invalid Dataset Mode!")

data_loader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True)
data_loader_val = torch.utils.data.DataLoader(data_val,batch_size=1,shuffle=False)
data_loader_compare = torch.utils.data.DataLoader(data_compare,batch_size=1,shuffle=False)

REG_loss = torch.nn.L1Loss()

save_dir = '../results/'
mkdir_if_missing(save_dir)
output_dir = osp.join(save_dir, experiment_id+'_'+dataset_name+'_'+model_name)
mkdir_if_missing(output_dir)
result_txt = open(output_dir+'/result.txt', 'w')
best_MAE_MSE_txt = open(output_dir+'/best_MAE_MSE.txt', 'w')
best_mae_score = 9999.

for epoch in range(start_step, end_step+1):

    print('Start Epoch '+str(epoch)+' ...')
    step = 0 # how many samples are processed
    sum_loss = 0
    reg_loss = 0
    comp_loss = 0
    soft_comp_loss = 0
    net.train()

    for blob in tqdm(data_loader):
        step = step + 1
        im_data = blob['im'].cuda()
        num = blob['num'].cuda()
        im1_data = blob['im1'].cuda()
        im2_data = blob['im2'].cuda()
        label = blob['lb'].cuda()

        optimizer.zero_grad()
        if RES_MODE:
            optimizer_res.zero_grad()
        loss = 0.0

        out, p_dmap = net(im_data)
        regression_loss = REG_loss(out, num/scaling_rate)
        loss += lambda_reg * regression_loss
        reg_loss += regression_loss.item()


        if RES_MODE:
            out1, inner1 = net(im1_data)
            out2, inner2 = net(im2_data)
            out_ = net_res(inner1, inner2)
            out_0 = torch.zeros(out_.shape).cuda()
            comparator_loss = F.margin_ranking_loss(out1, out2, label, margin_comp)
            loss += lambda_comp * comparator_loss
            comp_loss += comparator_loss.item()
            margin = abs((net_infer(im1_data)[0] - net_infer(im2_data)[0]).detach().cpu().numpy()[0].item()) * adaptive_ratio # just for batch_size = 1
            soft_comparator_loss = F.margin_ranking_loss(out_, out_0, label, margin)
            loss += lambda_soft * soft_comparator_loss
            soft_comp_loss += soft_comparator_loss.item()
            # to do
        else:
            out1, _ = net(im1_data)
            out2, _ = net(im2_data)
            margin = abs((net_infer(im1_data)[0] - net_infer(im2_data)[0]).detach().cpu().numpy()[0].item()) * adaptive_ratio # just for batch_size = 1
            comparator_loss = F.margin_ranking_loss(out1, out2, label, margin)
            loss += lambda_comp * comparator_loss
            comp_loss += comparator_loss.item()

        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        if RES_MODE:
            optimizer_res.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = disp_interval / duration
            t.tic()

    print('epoch: ' + str(epoch) + '  ' + 'loss: ' + str(sum_loss/step) \
            + '  ' + 'reg loss: ' + str(reg_loss/step) \
            + '  ' + 'comp loss: ' + str(comp_loss/step) \
            + '  ' + 'soft comp loss:  ' + str(soft_comp_loss/step))

    if epoch % eval_step == 0:
        # calculate error on the validation dataset
        mae_score, mse_score = evaluate_model(net, data_loader_val, data_loader_compare, is_rank, scaling_rate)
        result_txt.write('epoch: ' + str(epoch) + '  ' + 'MAE: ' + str(mae_score) + ' MSE: ' + str(mse_score) + '\n')
        result_txt.flush()

        if mae_score < best_mae_score:
            best_mae_score = mae_score
            save_checkpoint({
                'epoch': epoch,
                'best_score': best_mae_score,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filepath=output_dir)
            best_MAE_MSE_txt.write('epoch: ' + str(epoch) + '  ' + 'MAE: ' + str(mae_score) + ' MSE: ' + str(mse_score) + '\n')
            best_MAE_MSE_txt.flush()
            print('epoch: ' + str(epoch) + '  ' + 'MAE: ' + str(mae_score) + ' MSE: ' + str(mse_score))

