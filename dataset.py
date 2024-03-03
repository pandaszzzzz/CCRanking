from __future__ import print_function, absolute_import
import sys
import os
import os.path as osp
import scipy.io as sio
from PIL import Image
import glob
import torch
import numpy as np
import cv2
import random
from utils import mkdir_if_missing
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import json

class BaseDataset(Dataset):
    def __init__(self):
        self.isTrain = True

    def _check_before_run(self, _check_dirs):
        """Check if all files are available before going deeper"""
        for _dir in _check_dirs:
            if not osp.exists(_dir):
                raise RuntimeError("'{}' is not available".format(_dir))

    def _vis_num_distribution(self): # for visualize data distribution
        nums = []
        for (_, num_) in self.samples_tuple_list:
            nums.append(num_)
        from matplotlib import pyplot
        pyplot.hist(nums, bins=200)
        pyplot.show()

    def _json_to_string(self, array): # for ShanghaiTech or UCF-QNRF
        """
        converts json to string specifically for shanghai tech dataset
        """
        if len(array)==0:
            return '[]'
        line = '['
        for i in range(len(array)):
            line += '{\"x\":'+str(array[i][0])+',\"y\":'+str(array[i][1])+'},' # [{x:1,y:1},{x:2,y:2}]
        return line[0:len(line)-1]+']'

    def _convert_mat_to_json(self, in_dir, out_dir): # for ShanghaiTech
        """
        converts every .mat file in in_dir to a .json equivalent in out_dir
        """
        print("converting mat to json from {}".format(in_dir))
        file_names = os.listdir(in_dir)
        for mat_file in file_names:
            mat_file_path = osp.join(in_dir, mat_file)
            file_extention = mat_file.split('.')[-1]
            file_id = mat_file[3:len(mat_file) - len(file_extention)]
            json_file_path = osp.join(out_dir, file_id + 'json')
            labels = sio.loadmat(mat_file_path)
            labels = labels['image_info'][0][0][0][0][0]
            labels = str(self._json_to_string(labels))
            with open(json_file_path, 'w') as outfile:
                outfile.write(labels)

    def _convert_ucf_mat_to_json(self, in_dir, out_dir): # for UCF-QNRF
        """
        converts every .mat file in in_dir to a .json equivalent in out_dir
        """
        print("converting mat to json from {}".format(in_dir))
        file_names = os.listdir(in_dir)
        for mat_file in file_names:
            mat_file_path = osp.join(in_dir, mat_file)
            file_extension = mat_file.split('.')[-1]
            file_id = mat_file[0:len(mat_file) - len(file_extension)-5]
            json_file_path = osp.join(out_dir, file_id + '.json')
            labels = sio.loadmat(mat_file_path)
            labels = labels['annPoints']
            labels = str(self._json_to_string(labels))
            with open(json_file_path, 'w') as outfile:
                outfile.write(labels)

    def _json_path_to_img_path(self, json_path): # for ShanghaiTech
        (dir, name) = osp.split(json_path)
        (r_dir, sub_dir) = osp.split(dir)
        if sub_dir == 'labels':
            sub_dir = 'images'
        elif sub_dir == 'train_lab':
            sub_dir = 'train_img'
        else:
            raise('Invalid JSON path!')
        img_path = osp.join(r_dir, sub_dir, name.split('.')[0]+'.jpg')
        return img_path

    def _img_path_to_img_tensor(self, img_path):
        img = plt.imread(img_path)/255# convert from [0,255] to [0,1]

        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        ds_rows, ds_cols = img.shape[0], img.shape[1]
        if self.isTrain:
            if random.randint(0,1)==1:
                img=img[:,::-1]

        while ds_cols * ds_rows > 700 * 800:
            ds_rows = int(ds_rows / 1.5)
            ds_cols = int(ds_cols / 1.5)

        # to downsample image and density-map to match deep-model.
        ds_rows=int(ds_rows//self.downsample)
        ds_cols=int(ds_cols//self.downsample)
        img = cv2.resize(img,(ds_cols*self.downsample,ds_rows*self.downsample))

        img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)

        img=torch.tensor(img,dtype=torch.float)
        img=transforms.functional.normalize(img,mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        return img

    def _img_path_to_4_img_tensor(self, img_path):
        img = plt.imread(img_path)/255# convert from [0,255] to [0,1]

        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        if self.isTrain:
            if random.randint(0,1)==1:
                img=img[:,::-1]

        imgs = []
        y, x = img.shape[0], img.shape[1]
        mid_y = random.randint(int(y/3.0), int(2.0*y/3.0))
        mid_x = random.randint(int(x/3.0), int(2.0*x/3.0))
        imgs.append(img[:mid_y,:mid_x])
        imgs.append(img[:mid_y,mid_x:])
        imgs.append(img[mid_y:,:mid_x])
        imgs.append(img[:mid_y,mid_x:])
        for i,img in enumerate(imgs):
            ds_rows, ds_cols = img.shape[0], img.shape[1]
            while ds_cols * ds_rows > 400 * 500:
                ds_rows = int(ds_rows / 1.5)
                ds_cols = int(ds_cols / 1.5)
            # to downsample image and density-map to match deep-model.
            ds_rows=int(ds_rows//self.downsample)
            ds_cols=int(ds_cols//self.downsample)
            img = cv2.resize(img,(ds_cols*self.downsample,ds_rows*self.downsample))

            img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)

            img=torch.tensor(img,dtype=torch.float)
            img=transforms.functional.normalize(img,mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            imgs[i] = img
        return imgs[0], imgs[1], imgs[2], imgs[3]

    def _img_path_to_crop_img_tensor(self, img_path):
        img = plt.imread(img_path)/255# convert from [0,255] to [0,1]

        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        if self.isTrain:
            if random.randint(0,1)==1:
                img=img[:,::-1]

        y, x = img.shape[0], img.shape[1]
        st_y = random.randint(0, int(y/4.0))
        st_x = random.randint(0, int(x/4.0))
        ed_y = st_y + int(3.0*y/4.0)
        ed_x = st_x + int(3.0*x/4.0)
        img = img[st_y:ed_y,st_x:ed_x]

        ds_rows, ds_cols = img.shape[0], img.shape[1]
        while ds_cols * ds_rows > 500 * 600:
            ds_rows = int(ds_rows / 1.5)
            ds_cols = int(ds_cols / 1.5)

        # to downsample image and density-map to match deep-model.
        ds_rows=int(ds_rows//self.downsample)
        ds_cols=int(ds_cols//self.downsample)
        img = cv2.resize(img,(ds_cols*self.downsample,ds_rows*self.downsample))

        img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)

        img=torch.tensor(img,dtype=torch.float)
        img=transforms.functional.normalize(img,mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        return img

    def _txt_path_to_img_path(self, txt_path): # for JHU
        (dir, name) = osp.split(txt_path)
        (r_dir, sub_dir) = osp.split(dir)
        if sub_dir == 'gt':
            sub_dir = 'images'
        else:
            raise('Invalid TXT path!')
        img_path = osp.join(r_dir, sub_dir, name.split('.')[0]+'.jpg')
        return img_path

    def _create_shanghai_or_UCF(self, shanghai=None):
        all_train_jsons = list(glob.glob(shanghai+'/*.json'))
        all_train_jsons.sort()
        for im_json in all_train_jsons:
            with open(im_json) as f:
                num_ = len(json.load(f))
                self.compare_tuple_list.append((self._json_path_to_img_path(im_json), num_))

    def _create_shanghai_or_UCF_labels(self, is_ucf=False):
        mkdir_if_missing(self.ori_dir_part_train_lab)
        #check if number os files is equal
        if len(os.listdir(self.ori_dir_part_train_mat)) != len(os.listdir(self.ori_dir_part_train_lab)):
            if is_ucf:
                self._convert_ucf_mat_to_json(self.ori_dir_part_train_mat, self.ori_dir_part_train_lab)
            else:
                self._convert_mat_to_json(self.ori_dir_part_train_mat, self.ori_dir_part_train_lab)

    def _create_shanghai_or_UCF_train(self):
        all_train_jsons = list(glob.glob(self.ori_dir_part_train_lab+'/*.json'))
        all_train_jsons.sort()
        for im_json in all_train_jsons:
            with open(im_json) as f:
                num_ = len(json.load(f))
                self.samples_tuple_list.append((im_json, num_))
    
    def _create_shanghai_or_UCF_test_labels(self, is_ucf=False):
        mkdir_if_missing(self.ori_dir_part_test_lab)
        #check if number os files is equal
        if len(os.listdir(self.ori_dir_part_test_mat)) != len(os.listdir(self.ori_dir_part_test_lab)):
            if is_ucf:
                self._convert_ucf_mat_to_json(self.ori_dir_part_test_mat, self.ori_dir_part_test_lab)
            else:    
                self._convert_mat_to_json(self.ori_dir_part_test_mat, self.ori_dir_part_test_lab)

    def _create_shanghai_or_UCF_test(self):
        all_test_jsons = list(glob.glob(self.ori_dir_part_test_lab+'/*.json'))
        all_test_jsons.sort()
        for im_json in all_test_jsons:
            with open(im_json) as f:
                num = len(json.load(f))
                self.samples_tuple_list.append((im_json, num))

    def _create_JHU_train(self):
        all_jhu_txt = list(glob.glob(self.ori_dir_jhu_train_txt+'/*.txt'))
        all_jhu_txt.sort()

        for im_txt in all_jhu_txt:
            num_ =  len(open(im_txt, 'r').readlines())
            if num_ > 3700 or (30 < num_ < 100 and num_ % 5 != 0) or (num_ < 200 and num_ >= 100 and (num_ % 5 != 0 and num_ % 10 != 0)) or num_ <= 30:
                continue
            self.samples_tuple_list.append((im_txt, num_))

    def _create_compare(self, shanghaiA=None, jhu=None, shanghaiB=None, ucf=None, compare_nums = 50):
        samples_search_list = []
        if shanghaiA:
            all_train_jsons = list(glob.glob(shanghaiA+'/*.json'))
            all_train_jsons.sort()
            for im_json in all_train_jsons:
                with open(im_json) as f:
                    num_ = len(json.load(f))
                    samples_search_list.append((im_json, num_))
        if shanghaiB:
            all_train_jsons = list(glob.glob(shanghaiB+'/*.json'))
            all_train_jsons.sort()
            for im_json in all_train_jsons:
                with open(im_json) as f:
                    num_ = len(json.load(f))
                    samples_search_list.append((im_json, num_))
        if jhu:
            all_jhu_txt = list(glob.glob(jhu+'/*.txt'))
            all_jhu_txt.sort()
            for im_txt in all_jhu_txt:
                num_ =  len(open(im_txt, 'r').readlines())
                samples_search_list.append((im_txt, num_))
        if ucf:
            all_train_jsons = list(glob.glob(ucf+'/*.json'))
            all_train_jsons.sort()
            for im_json in all_train_jsons:
                with open(im_json) as f:
                    num_ = len(json.load(f))
                    samples_search_list.append((im_json, num_))
        if jhu:
            if compare_nums == 30:
                search_space = [((8, 22), 1),
                            ((28, 42), 1),
                            ((45, 80), 2),
                            ((95, 130), 2),
                            ((145, 180), 2),
                            ((190, 260), 2),
                            ((290, 310), 1),
                            ((340, 410), 2),
                            ((440, 510), 3),
                            ((540, 610), 2),
                            ((640, 710), 2),
                            ((740, 810), 2),
                            ((840, 910), 2),
                            ((940, 1010), 1),
                            ((1090,1210), 1),
                            ((1290,1510), 1),
                            ((1590,1910), 1),
                            ((1990,2625), 1),
                            ((2825,3025), 1)]
            elif compare_nums == 50:
                search_space = [((8, 22), 1),
                            ((28, 42), 1),
                            ((45, 80), 3),
                            ((95, 130), 3),
                            ((145, 180), 3),
                            ((190, 260), 4),
                            ((290, 310), 1),
                            ((340, 410), 4),
                            ((440, 510), 4),
                            ((540, 610), 4),
                            ((640, 710), 3),
                            ((740, 810), 3),
                            ((840, 910), 3),
                            ((940, 1010), 2),
                            ((1090,1210), 2),
                            ((1290,1310), 1),
                            ((1390,1510), 1),
                            ((1590,1710), 1),
                            ((1810,1910), 1),
                            ((1990,2210), 1),
                            ((2375,2625), 1),
                            ((2825,3025), 1)]
            elif compare_nums == 100:
                search_space = [((8, 12), 1),
                            ((18, 22), 1),
                            ((28, 32), 1),
                            ((38, 42), 1),
                            ((45, 55), 3),
                            ((70, 80), 3),
                            ((95, 105), 3),
                            ((120, 130), 3),
                            ((145, 155), 3),
                            ((170, 180), 3),
                            ((190, 210), 4),
                            ((240, 260), 4),
                            ((290, 310), 4),
                            ((340, 360), 5),
                            ((390, 410), 5),
                            ((440, 460), 4),
                            ((490, 510), 4),
                            ((540, 560), 4),
                            ((590, 610), 4),
                            ((640, 660), 3),
                            ((690, 710), 3),
                            ((740, 760), 3),
                            ((790, 810), 3),
                            ((840, 860), 3),
                            ((890, 910), 3),
                            ((940, 960), 2),
                            ((990, 1010), 2),
                            ((1090,1110), 2),
                            ((1190,1210), 2),
                            ((1290,1310), 2),
                            ((1390,1410), 1),
                            ((1490,1510), 1),
                            ((1590,1610), 1),
                            ((1690,1710), 1),
                            ((1810,1820), 1),
                            ((1890,1910), 1),
                            ((1990,2010), 1),
                            ((2190,2210), 1),
                            ((2375,2425), 1),
                            ((2575,2625), 1),
                            ((2825,2875), 1),
                            ((2975,3025), 1)]
        elif shanghaiA:
            if compare_nums == 10:
                search_space = [((30, 100), 1),
                                 ((100, 150), 1),
                                 ((150, 200), 1),
                                 ((200, 300), 1),
                                 ((300, 400), 1),
                                 ((400, 600), 1),
                                 ((600, 800), 1),
                                 ((800, 1000), 1),
                                 ((1000, 2000), 1),
                                 ((2000, 3000), 1)]
            elif compare_nums == 30:
                search_space = [((30, 100), 3),
                                 ((100, 150), 3),
                                 ((150, 200), 3),
                                 ((200, 300), 4),
                                 ((300, 400), 4),
                                 ((400, 600), 3),
                                 ((600, 800), 3),
                                 ((800, 1000), 3),
                                 ((1000, 2000), 3),
                                 ((2000, 3000), 1)]
            elif compare_nums == 50:
                search_space = [((30, 60), 1),
                            ((60, 90), 1),
                            ((90, 120), 2),
                            ((120, 150), 2),
                            ((150, 180), 2),
                            ((180, 210), 2),
                            ((210, 240), 3),
                            ((240, 270), 3),
                            ((270, 300), 3),
                            ((300, 330), 2),
                            ((330, 380), 2),
                            ((380, 430), 2),
                            ((430, 480), 2),
                            ((480, 530), 2),
                            ((530, 580), 2),
                            ((580, 620), 1),
                            ((620, 660), 1),
                            ((660, 700), 1),
                            ((700, 740), 1),
                            ((740, 780), 1),
                            ((780, 820), 1),
                            ((820, 860), 1),
                            ((860, 900), 1),
                            ((900, 1000), 1),
                            ((1000, 1200), 1),
                            ((1200, 1400), 1),
                            ((1400, 1600), 1),
                            ((1600, 1800), 1),
                            ((1800, 2000), 1),
                            ((2000, 2500), 1),
                            ((2500, 3000), 1)]
            elif compare_nums == 80:
                search_space = [((30, 60), 2),
                            ((60, 90), 2),
                            ((90, 120), 3),
                            ((120, 150), 3),
                            ((150, 180), 4),
                            ((180, 210), 4),
                            ((210, 240), 4),
                            ((240, 270), 4),
                            ((270, 300), 4),
                            ((300, 330), 4),
                            ((330, 380), 4),
                            ((380, 430), 4),
                            ((430, 480), 4),
                            ((480, 530), 3),
                            ((530, 580), 3),
                            ((580, 620), 2),
                            ((620, 660), 2),
                            ((660, 700), 2),
                            ((700, 740), 2),
                            ((740, 780), 2),
                            ((780, 820), 2),
                            ((820, 860), 2),
                            ((860, 900), 2),
                            ((900, 1000), 2),
                            ((1000, 1200), 2),
                            ((1200, 1400), 2),
                            ((1400, 1600), 2),
                            ((1600, 1800), 1),
                            ((1800, 2000), 1),
                            ((2000, 2500), 1),
                            ((2500, 3000), 1)]
            elif compare_nums == 150:
                search_space = [((30, 100), 15),
                                 ((100, 150), 15),
                                 ((150, 200), 15),
                                 ((200, 300), 20),
                                 ((300, 400), 20),
                                 ((400, 600), 20),
                                 ((600, 800), 15),
                                 ((800, 1000), 15),
                                 ((1000, 2000), 10),
                                 ((2000, 3000), 5)]
            elif compare_nums == 300:
                search_space = [((0, 5000), 300)]
            else:
                raise('Invalid Compare Nums!')
        elif shanghaiB:
            search_space = [((20, 30), 2),
                        ((30, 40), 3),
                        ((40, 50), 3),
                        ((50, 60), 3),
                        ((60, 70), 3),
                        ((70, 80), 3),
                        ((80, 90), 3),
                        ((90, 100), 3),
                        ((100, 120), 5),
                        ((120, 140), 5),
                        ((140, 160), 5),
                        ((160, 180), 5),
                        ((180, 200), 5),
                        ((200, 250), 4),
                        ((250, 300), 4),
                        ((300, 350), 2),
                        ((350, 400), 1),
                        ((400, 500), 1),
                        ((500, 600), 1)]
        elif ucf:
            search_space = [((0, 100), 3),
                        ((100, 150), 4),
                        ((150, 200), 4),
                        ((150, 200), 4),
                        ((250, 300), 4),
                        ((250, 300), 4),
                        ((300, 400), 4),
                        ((400, 500), 4),
                        ((500, 600), 4),
                        ((600, 700), 4),
                        ((700, 800), 3),
                        ((800, 900), 3),
                        ((900, 1000), 3),
                        ((1000, 1200), 3),
                        ((1200, 1400), 3),
                        ((1400, 1600), 3),
                        ((1600, 1800), 3),
                        ((1800, 2000), 3),
                        ((2000, 3000), 4),
                        ((3000, 4000), 3),
                        ((4000, 5000), 2),
                        ((5000, 10000), 2),
                        ((10000, 20000), 1)]
        else:
            raise AssertionError("Invalid Compare Mode!")
        for search_range in search_space:
            cnt = 0
            for sample in samples_search_list:
                if sample[1] >= search_range[0][0] and sample[1] <= search_range[0][1]:
                    cnt = cnt + 1
                    if sample[0][-3:] == 'txt':
                        self.compare_tuple_list.append((self._txt_path_to_img_path(sample[0]), sample[1]))
                    elif sample[0][-4:] == 'json':
                        self.compare_tuple_list.append((self._json_path_to_img_path(sample[0]), sample[1]))

                    if cnt == search_range[1]:
                        break


class ShanghaiTech(BaseDataset):
    samples_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(ShanghaiTech, self).__init__()
        self.metadata = kwargs
        self.is_A = kwargs['is_A'] if 'is_A' in kwargs else True
        self.root = '../countingdata/ShanghaiTech/'
        self.ori_dir_part = osp.join(self.root, 'part_A') if self.is_A else osp.join(self.root, 'part_B')
        self.ori_dir_part_train = osp.join(self.ori_dir_part, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.ori_dir_part, self.ori_dir_part_train, self.ori_dir_part_train_img,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_mat]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self._create_shanghai_or_UCF_labels()
        self._create_shanghai_or_UCF_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'ShanghaiTechA' if self.is_A else 'ShanghaiTechB'
        self.num_samples = len(self.samples_tuple_list)

    def __getitem__(self, index):
        (json_1st, num_1st) = self.samples_tuple_list[index]
        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            sample_info = (self._json_path_to_img_path(json_1st),self._json_path_to_img_path(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            sample_info = (self._json_path_to_img_path(json_1st),self._json_path_to_img_path(json_2nd),False)
        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        
        return_dict = {}
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.rank_err:
            return_dict['im7'] = img7
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        return return_dict

    def __len__(self):
        return self.num_samples

class UCF(BaseDataset):

    samples_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(UCF, self).__init__()
        self.metadata = kwargs
        self.root = '../countingdata/UCF-QNRF/'
        self.ori_dir_part_train = osp.join(self.root, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.root, self.ori_dir_part_train, self.ori_dir_part_train_img,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_mat]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self._create_shanghai_or_UCF_labels(is_ucf=True)
        self._create_shanghai_or_UCF_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'UCF-QNRF'
        self.num_samples = len(self.samples_tuple_list)

    def __getitem__(self, index):
        (json_1st, num_1st) = self.samples_tuple_list[index]
        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            sample_info = (self._json_path_to_img_path(json_1st),self._json_path_to_img_path(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            sample_info = (self._json_path_to_img_path(json_1st),self._json_path_to_img_path(json_2nd),False)
        
        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        
        return_dict = {}
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.rank_err:
            return_dict['im7'] = img7
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        return return_dict

    def __len__(self):
        return self.num_samples


class JHU(BaseDataset):

    samples_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(JHU, self).__init__()
        self.metadata = kwargs
        self.root_jhu = '../countingdata/jhu_crowd_v2.0/'
        self.ori_dir_jhu_train = osp.join(self.root_jhu, 'train')
        self.ori_dir_jhu_train_txt = osp.join(self.ori_dir_jhu_train, 'gt')
        self.ori_dir_jhu_train_img = osp.join(self.ori_dir_jhu_train, 'images')
        self.ori_dir_jhu_train_lab = osp.join(self.ori_dir_jhu_train, 'density_maps')
        self.dirs = [self.ori_dir_jhu_train, self.ori_dir_jhu_train_txt, self.ori_dir_jhu_train_img]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self.pairs = kwargs['pairs'] if 'pairs' in kwargs else None
        self._create_JHU_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'JHU'
        self.num_samples = len(self.samples_tuple_list)

    def __getitem__(self, index):
        (json_1st, num_1st) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path
        
        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),False)
        
        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        
        return_dict = {}
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples


class JHU_compare(BaseDataset):

    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(JHU_compare, self).__init__()
        self.root_jhu = '../countingdata/jhu_crowd_v2.0/'
        self.ori_dir_jhu_train = osp.join(self.root_jhu, 'train')
        self.ori_dir_jhu_train_txt = osp.join(self.ori_dir_jhu_train, 'gt')
        self.ori_dir_jhu_train_img = osp.join(self.ori_dir_jhu_train, 'images')
        self.dirs = [self.root_jhu, self.ori_dir_jhu_train,
                     self.ori_dir_jhu_train_img, self.ori_dir_jhu_train_txt]
        self._check_before_run(self.dirs)
        self.metadata = kwargs
        self.isTrain = False
        self.downsample = kwargs['down_sample']  # set network downsampling ratio
        self.compare_nums = kwargs['compare_nums'] if 'compare_nums' in kwargs else 50
        self._create_compare(jhu=self.ori_dir_jhu_train_txt, compare_nums=self.compare_nums)
        self.num_samples = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[index]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        return {'im':img, 'num':num}
    
    def print(self):
        for sample in self.compare_tuple_list:
            print(sample[0], sample[1])

    def __len__(self):
        return self.num_samples


class JHU_num_combine(BaseDataset):
    images_tuple_list = []
    samples_tuple_list = []
    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(JHU_num_combine, self).__init__()
        self.metadata = kwargs
        self.root_jhu = '../countingdata/jhu_crowd_v2.0/'
        self.ori_dir_jhu_train = osp.join(self.root_jhu, 'train')
        self.ori_dir_jhu_train_txt = osp.join(self.ori_dir_jhu_train, 'gt')
        self.ori_dir_jhu_train_img = osp.join(self.ori_dir_jhu_train, 'images')
        self.ori_dir_jhu_train_lab = osp.join(self.ori_dir_jhu_train, 'density_maps')
        self.dirs = [self.ori_dir_jhu_train, self.ori_dir_jhu_train_txt, self.ori_dir_jhu_train_img]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self.compare_nums = kwargs['compare_nums'] if 'compare_nums' in kwargs else 50
        self._create_compare(jhu=self.ori_dir_jhu_train_txt, compare_nums=self.compare_nums)
        self.pairs = kwargs['pairs'] if 'pairs' in kwargs else None
        if self.pairs == 1000:
            self._create_JHU_1000_pairs()
        else:
            self._create_JHU_num_combine(pairs=self.pairs)
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'JHU_num_combine'
        self.num_samples = len(self.samples_tuple_list)
        self.num_compare = len(self.compare_tuple_list)
    
    def _create_JHU_1000_pairs(self):
        if os.path.exists(os.path.join(self.ori_dir_jhu_train, 'pairs_1000.txt')):
            f = open(os.path.join(self.ori_dir_jhu_train, 'pairs_1000.txt'), 'r')
            for line in f.readlines():
                p = line.split(',')
                self.samples_tuple_list.append((p[0], p[1]))
            f.close()
            return
        all_jhu_txt = list(glob.glob(self.ori_dir_jhu_train_txt + '/*.txt'))
        for im_txt in all_jhu_txt:
            num_ =  len(open(im_txt, 'r').readlines())
            if num_ == 0:
                continue
            self.images_tuple_list.append((im_txt, num_))
        print(len(self.images_tuple_list))
        self.num_images = len(self.images_tuple_list)
        
        elem_set = set()
        for i in range(1000):
            (json_1st, num_1st) = self.images_tuple_list[random.choice(range(self.num_images))]
            while json_1st in elem_set:
                (json_1st, num_1st) = self.images_tuple_list[random.choice(range(self.num_images))]
            (json_2nd, num_2nd) = self.images_tuple_list[random.choice(range(self.num_images))]
            while (json_2nd in elem_set) or (not (num_1st / num_2nd >= self.split_ratio or num_2nd / num_1st >= self.split_ratio)):
                (json_2nd, num_2nd) = self.images_tuple_list[random.choice(range(self.num_images))]
            elem_set.add(json_1st)
            elem_set.add(json_2nd)
            if num_1st > num_2nd:
                self.samples_tuple_list.append((json_1st, json_2nd))
            else:
                self.samples_tuple_list.append((json_2nd, json_1st))
        assert len(elem_set) == 2000
        
        f = open(os.path.join(self.ori_dir_jhu_train, 'pairs_1000.txt'), 'w')
        for sam in self.samples_tuple_list:
            str_ = sam[0] + ',' + sam[1] + '\n'
            f.writelines([str_])
        f.close()
        sys.exit(0)


    def _create_JHU_num_combine(self, pairs):
        if os.path.exists(os.path.join(self.ori_dir_jhu_train, 'pairs_'+str(pairs)+'.txt')):
            f = open(os.path.join(self.ori_dir_jhu_train, 'pairs_'+str(pairs)+'.txt'), 'r')
            for line in f.readlines():
                p = line.split(',')
                self.samples_tuple_list.append((p[0], p[1]))
            f.close()
            return
        all_jhu_txt = list(glob.glob(self.ori_dir_jhu_train_txt+ '/*.txt'))
        for im_txt in all_jhu_txt:
            num_ =  len(open(im_txt, 'r').readlines())
            if num_ == 0:
                continue
            self.images_tuple_list.append((im_txt, num_))
        print(len(self.images_tuple_list))
        self.num_images = len(self.images_tuple_list)

        elem_set = set()
        for i in range(pairs):
            (json_1st, num_1st) = self.images_tuple_list[random.choice(range(self.num_images))]
            (json_2nd, num_2nd) = self.images_tuple_list[random.choice(range(self.num_images))]
            while not (num_1st / num_2nd >= self.split_ratio or num_2nd / num_1st >= self.split_ratio):
                (json_2nd, num_2nd) = self.images_tuple_list[random.choice(range(self.num_images))]
            while (json_1st, json_2nd) in elem_set or (json_2nd, json_1st) in elem_set:
                (json_1st, num_1st) = self.images_tuple_list[random.choice(range(self.num_images))]
                (json_2nd, num_2nd) = self.images_tuple_list[random.choice(range(self.num_images))]
                while not (num_1st / num_2nd >= self.split_ratio or num_2nd / num_1st >= self.split_ratio):
                    (json_2nd, num_2nd) = self.images_tuple_list[random.choice(range(self.num_images))]
            if num_1st > num_2nd:
                self.samples_tuple_list.append((json_1st, json_2nd))
                elem_set.add((json_1st, json_2nd))
            else:
                self.samples_tuple_list.append((json_2nd, json_1st))
                elem_set.add((json_2nd, json_1st))
        print(len(elem_set))
        print(len(self.samples_tuple_list))
        assert len(elem_set) == pairs
        
        f = open(os.path.join(self.ori_dir_jhu_train, 'pairs_'+str(pairs)+'.txt'), 'w')
        for sam in self.samples_tuple_list:
            str_ = sam[0] + ',' + sam[1] + '\n'
            f.writelines([str_])
        f.close()
        sys.exit(0)


    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[random.choice(range(self.num_compare))]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])

        (json_1st, json_2nd) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path
        path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
        if random.random() < 0.5:
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            sample_info = (path_to_img_path_2nd(json_2nd),path_to_img_path_1st(json_1st),False)
        
        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] - num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples



class ShanghaiTech_compare(BaseDataset):

    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(ShanghaiTech_compare, self).__init__()
        self.is_A = kwargs['is_A'] if 'is_A' in kwargs else True
        self.root = '../countingdata/ShanghaiTech/'
        self.ori_dir_part = osp.join(self.root, 'part_A') if self.is_A else osp.join(self.root, 'part_B')
        self.ori_dir_part_train = osp.join(self.ori_dir_part, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')

        self.dirs = [self.root, self.ori_dir_part, self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img]
        self._check_before_run(self.dirs)
        self.metadata = kwargs
        self.isTrain = False
        self.downsample = kwargs['down_sample']  # set network downsampling ratio
        if self.is_A:
            self.compare_nums = kwargs['compare_nums'] if 'compare_nums' in kwargs else 50
            self._create_compare(shanghaiA = self.ori_dir_part_train_lab, compare_nums=self.compare_nums)
        else:
            self._create_compare(shanghaiB = self.ori_dir_part_train_lab)
        self.num_samples = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[index]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        return {'im':img, 'num':num}

    def print(self):
        for sample in self.compare_tuple_list:
            print(sample[0], sample[1])

    def __len__(self):
        return self.num_samples

class JHU_combine(BaseDataset):
    samples_tuple_list = []
    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(JHU_combine, self).__init__()
        self.metadata = kwargs
        self.root_jhu = '../countingdata/jhu_crowd_v2.0/'
        self.ori_dir_jhu_train = osp.join(self.root_jhu, 'all')
        self.ori_dir_jhu_train_txt = osp.join(self.ori_dir_jhu_train, 'gt')
        self.ori_dir_jhu_train_img = osp.join(self.ori_dir_jhu_train, 'images')
        self.ori_dir_jhu_train_lab = osp.join(self.ori_dir_jhu_train, 'density_maps')
        self.dirs = [self.ori_dir_jhu_train, self.ori_dir_jhu_train_txt, self.ori_dir_jhu_train_img]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self.compare_nums = kwargs['compare_nums'] if 'compare_nums' in kwargs else 50
        self._create_compare(jhu=self.ori_dir_jhu_train_txt, compare_nums=self.compare_nums)
        self._create_JHU_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'JHU_combine'
        self.num_samples = len(self.samples_tuple_list)
        self.num_compare = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[random.choice(range(self.num_compare))]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])

        (json_1st, num_1st) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path

        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),False)

        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] - num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples

class UCF_compare(BaseDataset):

    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(UCF_compare, self).__init__()
        self.root = '../countingdata/UCF-QNRF/'
        self.ori_dir_part_train = osp.join(self.root, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')

        self.dirs = [self.root, self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img]
        self._check_before_run(self.dirs)
        self.metadata = kwargs
        self.isTrain = False
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio
        self._create_compare(ucf = self.ori_dir_part_train_lab)
        self.num_samples = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[index]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        return {'im':img, 'num':num}

    def print(self):
        for sample in self.compare_tuple_list:
            print(sample[0], sample[1])

    def __len__(self):
        return self.num_samples

class ShanghaiTech_combine(BaseDataset):
    samples_tuple_list = []
    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(ShanghaiTech_combine, self).__init__()
        self.metadata = kwargs
        self.is_A = kwargs['is_A'] if 'is_A' in kwargs else True
        self.root = '../countingdata/ShanghaiTech/'
        self.ori_dir_part = osp.join(self.root, 'part_A') if self.is_A else osp.join(self.root, 'part_B')
        self.ori_dir_part_train = osp.join(self.ori_dir_part, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.ori_dir_part, self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img, self.ori_dir_part_train_lab]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self._create_shanghai_or_UCF_labels()
        if self.is_A:
            self.compare_nums = kwargs['compare_nums'] if 'compare_nums' in kwargs else 50
            self._create_compare(shanghaiA = self.ori_dir_part_train_lab, compare_nums=self.compare_nums)
        else:
            self._create_compare(shanghaiB = self.ori_dir_part_train_lab)
        self._create_shanghai_or_UCF_train()

        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'ShanghaiTechA' if self.is_A else 'ShanghaiTechB'
        self.num_samples = len(self.samples_tuple_list)
        self.num_compare = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[random.choice(range(self.num_compare))]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        (json_1st, num_1st) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path

        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),False)

        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples

class ShanghaiTech_allcombine(BaseDataset):
    samples_tuple_list = []
    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(ShanghaiTech_allcombine, self).__init__()
        self.metadata = kwargs
        self.is_A = kwargs['is_A'] if 'is_A' in kwargs else True
        self.root = '../countingdata/ShanghaiTech/'
        self.ori_dir_part = osp.join(self.root, 'part_A') if self.is_A else osp.join(self.root, 'part_B')
        self.ori_dir_part_train = osp.join(self.ori_dir_part, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.ori_dir_part, self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img, self.ori_dir_part_train_lab]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self._create_shanghai_or_UCF_labels()
        self._create_shanghai_or_UCF(self.ori_dir_part_train_lab)
        self._create_shanghai_or_UCF_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'ShanghaiTechA' if self.is_A else 'ShanghaiTechB'
        self.num_samples = len(self.samples_tuple_list)
        self.num_compare = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[random.choice(range(self.num_compare))]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        (json_1st, num_1st) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path

        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),False)

        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples

class ShanghaiTech_reg(BaseDataset):
    samples_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(ShanghaiTech_reg, self).__init__()
        self.metadata = kwargs
        self.is_A = kwargs['is_A'] if 'is_A' in kwargs else True
        self.root = '../countingdata/ShanghaiTech/'
        self.ori_dir_part = osp.join(self.root, 'part_A') if self.is_A else osp.join(self.root, 'part_B') 
        self.ori_dir_part_train = osp.join(self.ori_dir_part, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.ori_dir_part, self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img, self.ori_dir_part_train_lab]
        self._check_before_run(self.dirs)
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio
        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False

        self._create_shanghai_or_UCF_labels()
        self._create_shanghai_or_UCF_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'ShanghaiTechA_reg' if self.is_A else 'ShanghaiTechB_reg'
        self.num_samples = len(self.samples_tuple_list)

    def __getitem__(self, index):
        (im_json, num) = self.samples_tuple_list[index]
        img_path = self._json_path_to_img_path(im_json)
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return return_dict

    def __len__(self):
        return self.num_samples

class ShanghaiTech_eval(BaseDataset):

    samples_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(ShanghaiTech_eval, self).__init__()
        self.is_A = kwargs['is_A'] if 'is_A' in kwargs else True
        self.root = '../countingdata/ShanghaiTech/'
        self.ori_dir_part = osp.join(self.root, 'part_A') if self.is_A else osp.join(self.root, 'part_B')
        self.ori_dir_part_test = osp.join(self.ori_dir_part, 'test_data')
        self.ori_dir_part_test_mat = osp.join(self.ori_dir_part_test, 'ground_truth')
        self.ori_dir_part_test_img = osp.join(self.ori_dir_part_test, 'images')
        self.ori_dir_part_test_lab = osp.join(self.ori_dir_part_test, 'labels')
        self.dirs = [self.root, self.ori_dir_part, self.ori_dir_part_test_img,
                     self.ori_dir_part_test, self.ori_dir_part_test_mat, self.ori_dir_part_test_lab]
        self._check_before_run(self.dirs)
        self.metadata = kwargs
        self.isTrain = False
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio
        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self._create_shanghai_or_UCF_test_labels()
        self._create_shanghai_or_UCF_test()
        if self.vis_mode:
            self._vis_num_distribution()
        self.num_samples = len(self.samples_tuple_list)
        kwargs['name'] = 'ShanghaiTechA_eval' if self.is_A else 'ShanghaiTechB_eval'

    def __getitem__(self, index):
        (im_json, num) = self.samples_tuple_list[index]
        img_path = self._json_path_to_img_path(im_json)
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        return {'im':img, 'num':num, 'path':im_json}
    
    def __len__(self):
        return self.num_samples

class UCF_combine(BaseDataset):
    samples_tuple_list = []
    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(UCF_combine, self).__init__()
        self.metadata = kwargs
        self.root = '../countingdata/UCF-QNRF/'
        self.ori_dir_part_train = osp.join(self.root, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.root, self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self._create_shanghai_or_UCF_labels(is_ucf=True)
        self._create_compare(ucf = self.ori_dir_part_train_lab)
        self._create_shanghai_or_UCF_train()

        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'UCF-QNRF'
        self.num_samples = len(self.samples_tuple_list)
        self.num_compare = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[random.choice(range(self.num_compare))]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        (json_1st, num_1st) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path

        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),False)

        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples

class UCF_allcombine(BaseDataset):
    samples_tuple_list = []
    compare_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(UCF_allcombine, self).__init__()
        self.metadata = kwargs
        self.root = '../countingdata/UCF-QNRF/'
        self.ori_dir_part_train = osp.join(self.root, 'train_data')
        self.ori_dir_part_train_mat = osp.join(self.ori_dir_part_train, 'ground_truth')
        self.ori_dir_part_train_img = osp.join(self.ori_dir_part_train, 'images')
        self.ori_dir_part_train_lab = osp.join(self.ori_dir_part_train, 'labels')
        self.dirs = [self.ori_dir_part_train, self.ori_dir_part_train_mat,
                     self.ori_dir_part_train_lab, self.ori_dir_part_train_img]
        self._check_before_run(self.dirs)
        self.split_ratio =   kwargs['split_ratio'] # set split ratio as 2: difference in the number of people is at least twice as large.
        self.split_num =     kwargs['split_num']  # set split num as 500: more than 500 is considered "more"
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio

        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self.add_err =       kwargs['add_err'] if 'add_err' in kwargs else False
        self.rank_err =      kwargs['rank_err'] if 'rank_err' in kwargs else False

        self._create_shanghai_or_UCF_labels(is_ucf=True)
        self._create_shanghai_or_UCF(self.ori_dir_part_train_lab)
        self._create_shanghai_or_UCF_train()
        if self.vis_mode:
            self._vis_num_distribution()
        kwargs['name'] = 'UCF-QNRF'
        self.num_samples = len(self.samples_tuple_list)
        self.num_compare = len(self.compare_tuple_list)

    def __getitem__(self, index):
        (img_path, num) = self.compare_tuple_list[random.choice(range(self.num_compare))]
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        (json_1st, num_1st) = self.samples_tuple_list[index]
        path_to_img_path_1st = self._json_path_to_img_path if json_1st.split('.')[-1] == 'json' else self._txt_path_to_img_path

        if num_1st >= self.split_num:
            # find satisfied training pair
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st > num_2nd * self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),True)
        else:
            (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            while not (num_1st < num_2nd / self.split_ratio):
                (json_2nd, num_2nd) = self.samples_tuple_list[random.choice(range(self.num_samples))]
            path_to_img_path_2nd = self._json_path_to_img_path if json_2nd.split('.')[-1] == 'json' else self._txt_path_to_img_path
            sample_info = (path_to_img_path_1st(json_1st),path_to_img_path_2nd(json_2nd),False)

        img1 = self._img_path_to_img_tensor(sample_info[0])
        img2 = self._img_path_to_img_tensor(sample_info[1])
        label = torch.tensor([1.]) if sample_info[2] else torch.tensor([-1.])
        if self.add_err:
            img3, img4, img5, img6 = self._img_path_to_4_img_tensor(sample_info[0])
        if self.rank_err:
            img7 = self._img_path_to_crop_img_tensor(sample_info[1])

        return_dict = {}
        return_dict['im'] = img
        return_dict['num'] = num
        return_dict['im1'] = img1
        return_dict['im2'] = img2
        return_dict['lb'] = label
        if self.add_err:
            return_dict['im3'] = img3
            return_dict['im4'] = img4
            return_dict['im5'] = img5
            return_dict['im6'] = img6
        if self.vis_mode:
            return_dict['path1'] = sample_info[0]
            return_dict['path2'] = sample_info[1]
            return_dict['num1'] = num_1st
            return_dict['num2'] = num_2nd
        if self.rank_err:
            return_dict['im7'] = img7
        return return_dict

    def __len__(self):
        return self.num_samples

class UCF_eval(BaseDataset):

    samples_tuple_list = []
    metadata = dict()

    def __init__(self, **kwargs):
        super(UCF_eval, self).__init__()
        self.root = '../countingdata/UCF-QNRF/'
        self.ori_dir_part_test = osp.join(self.root, 'test_data')
        self.ori_dir_part_test_mat = osp.join(self.ori_dir_part_test, 'ground_truth')
        self.ori_dir_part_test_img = osp.join(self.ori_dir_part_test, 'images')
        self.ori_dir_part_test_lab = osp.join(self.ori_dir_part_test, 'labels')
        self.dirs = [self.root, self.ori_dir_part_test_img,
                     self.ori_dir_part_test, self.ori_dir_part_test_mat, self.ori_dir_part_test_lab]
        self._check_before_run(self.dirs)
        self.metadata = kwargs
        self.isTrain = False
        self.downsample =    kwargs['down_sample']  # set network downsampling ratio
        self.vis_mode =      kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        self._create_shanghai_or_UCF_test_labels(is_ucf=True)
        self._create_shanghai_or_UCF_test()
        if self.vis_mode:
            self._vis_num_distribution()
        self.num_samples = len(self.samples_tuple_list)
        kwargs['name'] = 'UCF_eval'

    def __getitem__(self, index):
        (im_json, num) = self.samples_tuple_list[index]
        img_path = self._json_path_to_img_path(im_json)
        img = self._img_path_to_img_tensor(img_path)
        num = torch.tensor([num])
        return {'im':img, 'num':num, 'path':im_json}
    
    def __len__(self):
        return self.num_samples

if __name__ == '__main__':
    data = JHU_compare(down_sample=8)
    print(len(data))
    # data.print()
    data_val = ShanghaiTech_eval(down_sample=8)
    print(len(data_val))
    train_jhu = JHU(split_ratio=2, split_num=1000, down_sample=8)
    print(len(train_jhu))
    train = ShanghaiTech(split_ratio=2, split_num=1000, down_sample=8)
    print(len(train))
