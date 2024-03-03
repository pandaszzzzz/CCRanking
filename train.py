import os
import os.path as osp
import torch
import numpy as np
import sys
from tqdm import tqdm
from utils import Timer
from torch.autograd import Variable
from utils import save_checkpoint, save_epoch_checkpoint
from utils import mkdir_if_missing
from dataset import *
from csrnet import CSRNet
from evaluate import evaluate_model
import argparse

parser = argparse.ArgumentParser(description='Train crowd counting network')
parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float, help="initial learning rate")
parser.add_argument('--lr_D', '--learning-rate-D', default=0.001, type=float, help="initial learning rate")
parser.add_argument('--split_ratio', default=2., type=float, help="set split ratio as 2: difference in the number of people is at least twice as large")
parser.add_argument('--split_num', default=1000, type=int, help="set split num as 500: more than 500 is considered more")
parser.add_argument('--down_sample', default=8, type=int, help="set network downsampling ratio")
parser.add_argument('--compare_nums', default=50, type=int)

parser.add_argument('--vis_mode', default=False, action='store_true', help="use visualization mode, default False")
parser.add_argument('--add_loss_mode', default=False, action='store_true', help="use sum loss, default False")
parser.add_argument('--rank_loss_mode', default=False, action='store_true', help='use rank loss, default False')
parser.add_argument('--eval_mode', default='Rank', type=str, help="None | Rank | NoCompare")
parser.add_argument('--load_weights', default=False, action='store_true')
parser.add_argument('--save_all_weights', default=False, action='store_true')

parser.add_argument('--scaling_rate', default=100, type=float, help="result ratio")

parser.add_argument('--lambda_summ', default=1., type=float)
parser.add_argument('--lambda_rank', default=0.2, type=float)
parser.add_argument('--margin_comp', default=0.5, type=float, help="note if adaptive margin mode, it is initial margin")
parser.add_argument('--margin_rank', default=0.2, type=float)

parser.add_argument('--dataset_name', default='ShanghaiTechA')
parser.add_argument('--model_name', default='CSRNet', type=str)
parser.add_argument('--start_step', default=0, type=int)
parser.add_argument('--end_step', default=500, type=int, help="ShanghaiTechA step = 500 | ShanghaiTechA_JHU step = 70")
parser.add_argument('--eval_step', default=2, type=int)

parser.add_argument('--seed', default=64678, type=int)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--experiment-ID', default='0', type=str, help="ID of the experiment")
args = parser.parse_args()

lr = args.lr
lr_D = args.lr_D
split_ratio = args.split_ratio
split_num = args.split_num
down_sample = args.down_sample
compare_nums = args.compare_nums
seed = args.seed
beta1 = 0.9
beta2 = 0.999

################################################# EXPERIMENTAL MODE #################################################
VIS_MODE = args.vis_mode
ADD_LOSS_MODE = args.add_loss_mode
RANK_LOSS_MODE = args.rank_loss_mode
EVAL_MODE = args.eval_mode
SAVE_ALL_WEIGHTS = args.save_all_weights

scaling_rate = args.scaling_rate
margin_comp = args.margin_comp
margin_rank = args.margin_rank
lambda_summ = args.lambda_summ
lambda_rank = args.lambda_rank
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

net = CSRNet(add_mode=ADD_LOSS_MODE, load_weights=args.load_weights)
net.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas = (beta1, beta2))

if resume:
    resume_dir = osp.join(resume, experiment_id+'_'+dataset_name+'_'+model_name)
    pretrained_model = osp.join(resume_dir, 'model_best.pth.tar')
    disc_model = osp.join(resume_dir, 'disc_best.pth.tar')
    if osp.isfile(pretrained_model) and osp.isfile(disc_model):
        print("=> loading checkpoint '{}'".format(pretrained_model))
        checkpoint = torch.load(pretrained_model)
        start_step = checkpoint['epoch']
        best_score = checkpoint['best_score']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})".format(pretrained_model, checkpoint['epoch']))
    
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_model))

if EVAL_MODE == 'None':
    is_rank = False
elif EVAL_MODE == 'Rank':
    is_rank = True
elif EVAL_MODE == 'NoCompare':
    is_rank = None
else:
    raise AssertionError("Invalid Evaluation Mode!")


if dataset_name == 'ShanghaiTechA':
    data = ShanghaiTech(vis_mode=VIS_MODE,
                        add_err=ADD_LOSS_MODE,rank_err=RANK_LOSS_MODE,
                        split_ratio=split_ratio, split_num=split_num, down_sample=down_sample)
    data_compare = ShanghaiTech_compare(down_sample=down_sample, compare_nums=compare_nums)
    data_val = ShanghaiTech_eval(vis_mode=VIS_MODE,down_sample=down_sample)
else:
    raise AssertionError("Invalid Dataset Mode!")

data_loader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True)
data_loader_val = torch.utils.data.DataLoader(data_val,batch_size=1,shuffle=False)
data_loader_compare = torch.utils.data.DataLoader(data_compare,batch_size=1,shuffle=False)

COMP_loss = torch.nn.MarginRankingLoss(margin=margin_comp)
if RANK_LOSS_MODE:
    RANK_loss = torch.nn.MarginRankingLoss(margin=margin_rank)
if ADD_LOSS_MODE:
    SUMM_loss = torch.nn.L1Loss()

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
    comp_loss = 0
    summ_loss = 0
    rank_loss = 0
    net.train()

    for blob in tqdm(data_loader):
        step = step + 1
        im1_data = blob['im1'].cuda()
        im2_data = blob['im2'].cuda()
        if ADD_LOSS_MODE:
            im3_data = blob['im3'].cuda()
            im4_data = blob['im4'].cuda()
            im5_data = blob['im5'].cuda()
            im6_data = blob['im6'].cuda()
        if RANK_LOSS_MODE:
            im7_data = blob['im7'].cuda()

        label = blob['lb'].cuda()

        optimizer.zero_grad()

        loss = 0.0
        out1, p_dmap1 = net(im1_data)
        out2, p_dmap2 = net(im2_data)
        comparator_loss = COMP_loss(out1, out2, label)
        loss += comparator_loss
        comp_loss += comparator_loss.item()

        if ADD_LOSS_MODE:
            out3, _ = net(im3_data)
            out4, _ = net(im4_data)
            out5, _ = net(im5_data)
            out6, _ = net(im6_data)
            summer_loss = SUMM_loss(out1/torch.abs(out1), (out3+out4+out5+out6)/torch.abs(out1))
            loss += lambda_summ * summer_loss
            summ_loss += summer_loss.item()

        if RANK_LOSS_MODE:
            out7, _ = net(im7_data)
            ranker_loss = RANK_loss(out2, out7, torch.tensor([1.]).unsqueeze(0).cuda())
            loss += lambda_rank * ranker_loss
            rank_loss += ranker_loss.item()

        sum_loss += loss.item()
        loss.backward()
        optimizer.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = disp_interval / duration
            t.tic()

    print('epoch: ' + str(epoch) + '  ' + 'loss: ' + str(sum_loss/step) \
            + '  ' + 'comp loss: ' + str(comp_loss/step) \
            + '  ' + 'summ loss: ' + str(summ_loss/step) \
            + '  ' + 'rank loss: ' + str(rank_loss/step))

    if epoch % eval_step == 0:
        # calculate error on the validation dataset
        mae_score, mse_score = evaluate_model(net, data_loader_val, data_loader_compare, is_rank, scaling_rate)
        result_txt.write('epoch: ' + str(epoch) + '  ' + 'MAE: ' + str(mae_score) + ' MSE: ' + str(mse_score) + '\n')
        result_txt.flush()
        if SAVE_ALL_WEIGHTS:
            save_epoch_checkpoint({
                'epoch': epoch,
                'best_score': best_mae_score,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, epoch, filepath=output_dir)
        if mae_score < best_mae_score:
            best_mae_score = mae_score
            if not SAVE_ALL_WEIGHTS:
                save_checkpoint({
                    'epoch': epoch,
                    'best_score': best_mae_score,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, filepath=output_dir)
            best_MAE_MSE_txt.write('epoch: ' + str(epoch) + '  ' + 'MAE: ' + str(mae_score) + ' MSE: ' + str(mse_score) + '\n')
            best_MAE_MSE_txt.flush()
            print('epoch: ' + str(epoch) + '  ' + 'MAE: ' + str(mae_score) + ' MSE: ' + str(mse_score))
