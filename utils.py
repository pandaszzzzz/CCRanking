import os.path as osp
import os
from shutil import copyfile
import sys
import errno
import time
import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def copy_to_directory(files_list, output_dir):
    for file in files_list:
        file_name = file.split('/')[-1]
        out_file_path = osp.join(output_dir, file_name)
        copyfile(file, out_file_path)

def save_checkpoint(state, filepath=''):
    torch.save(state, filepath+'/model_best.pth.tar')

def save_epoch_checkpoint(state, epoch, filepath=''):
    torch.save(state, filepath+ '/'+str(epoch) + '_model_best.pth.tar')

def save_D_checkpoint(state, filepath=''):
    torch.save(state, filepath+'/disc_best.pth.tar')
    
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def intersec(first, second):
    overlap = False
    overlap = overlap or (first[0] <= second[0] and second[0] <= first[2] and first[1] <= second[1] and second[1] <= first[3])
    overlap = overlap or (first[0] <= second[2] and second[2] <= first[2] and first[1] <= second[1] and second[1] <= first[3])
    overlap = overlap or (first[0] <= second[0] and second[0] <= first[2] and first[1] <= second[3] and second[3] <= first[3])
    overlap = overlap or (first[0] <= second[2] and second[2] <= first[2] and first[1] <= second[3] and second[3] <= first[3])
    return overlap

def cnt_overlaps(boxes):
    boxes_overlap = []
    id_overlap = []
    for ind_first, first in enumerate(boxes):
        cnt = 0
        overlap = []
        for ind_second, second in enumerate(boxes):
            if ind_first != ind_second and intersec(first, second):
                cnt += 1
                overlap.append(ind_second)
        boxes_overlap.append(cnt)
        id_overlap.append(overlap)
    return boxes_overlap, id_overlap


class Timer(object):
    def __init__(self):
        self.tot_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.tot_time += self.diff
        self.calls += 1
        self.average_time = self.tot_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
    