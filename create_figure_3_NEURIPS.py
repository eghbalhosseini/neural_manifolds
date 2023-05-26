import numpy as np
import sys
import os
import glob
import getpass
import argparse
import pickle
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
import collections
def makehash():
    return collections.defaultdict(makehash)
import fnmatch
import torch
from tqdm import tqdm
from utils import save_dir, analyze_dir, results_dir,train_pool
import scipy.spatial.distance as dist
from tqdm import tqdm
import re
def moving_average(x,w):
    w_min=1#int(np.floor(w/2))
    return pd.Series(x).rolling(w, min_periods=w_min).mean()
av_window=10

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

save_dir = '/nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/extracted/'
data_dir = '/nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/data/'
analyze_dir = '/nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/analyze/'
results_dir = '/nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/results/'


if __name__ == '__main__':
    model_identifier = 'NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed'
    train_identifier = 'epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06'
    analyze_identifier = 'mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'

    training_files = []
    for file in os.listdir(os.path.join(save_dir, model_identifier, train_identifier)):
        if fnmatch.fnmatch(file, '*.pth'):
            training_files.append(os.path.join(save_dir, model_identifier, train_identifier, file))

    grad_pkl_files = []
    for file in os.listdir(os.path.join(save_dir, analyze_identifier, model_identifier, train_identifier)):
        if fnmatch.fnmatch(file, '*gradient_data_v3.pkl'):
            grad_pkl_files.append(os.path.join(save_dir, analyze_identifier, model_identifier, train_identifier, file))
    s = [re.findall('/\d+', x) for x in grad_pkl_files]
    s = [item for sublist in s for item in sublist]
    dummy_id = [(x.split('/')) for x in s]
    file_id = [int(x.split('/')[1]) for x in s]
    sorted_files = [grad_pkl_files[x] for x in np.argsort(file_id)]
    grad_pkl_files = sorted_files

    hier_accu = []
    test_predictions = []
    test_probabilites = []
    test_grad_dict = []
    for idx, files in tqdm(enumerate(training_files)):
        test = torch.load(files)
        hier_accu.append([test['epoch'], test['batchidx'], test['hier_test_acc']])
        test_predictions.append([test['target_test'], test['pred_test']])
        test_probabilites.append(test['pred_test_prob'])
        test_grad_dict.append(test['grad_dict'])

    dummy = []
    epochs = []
    batches = []
    for item in hier_accu:
        epochs.append(item[0])
        batches.append(item[1])
        dummy.append(item[0] * 10000 + item[1])
    fixed_idx = np.argsort(dummy)

    epoch_factor = 1e3 * np.ceil(np.max(batches) / 1e3)
    epoch_batch = epoch_factor * np.asarray(epochs) + np.asarray(batches)
    x_bar = np.arange(len(epoch_batch)) / len(epoch_batch)
    epoch_bar = np.argwhere(np.diff(np.asarray(epochs))) / (len(epoch_batch))

    params = train_pool[model_identifier]()
    params.load_dataset()