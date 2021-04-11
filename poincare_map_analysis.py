import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import sys
# poincaremaps
sys.path.insert(1,'/om/user/ehoseini/PoincareMaps')
from poincare_maps import *
from main import *

import torch
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try :
    torch.set_deterministic(True)
except:
    pass
torch.set_printoptions(precision=10)

import numpy as np

import os
import argparse
from utils.model_utils import save_dict

import fnmatch
from utils import save_dir, analyze_dir, result_dir,train_pool
from utils.analysis_utils import analyze_pool
import re
import itertools

ROOTDIR = "/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/"

parser = argparse.ArgumentParser(description='run poincare map analysis')
parser.add_argument('model_id', type=str, default="NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed")
parser.add_argument('train_id', type=str, default="epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06")
parser.add_argument('analyze_id', type=str, default="mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5")
parser.add_argument('layer_name', type=str, default="layer_1_Linear")
parser.add_argument('hier_idx',type=int,default=1)
args = parser.parse_args()

if __name__ == '__main__':


    model_identifier = args.model_id
    train_identifier = args.train_id
    analyze_id = args.analyze_id
    layer = args.layer_name
    hier_idx = args.hier_id
    num_subsamples = 100
    k = 100
    knn_identifier = f"knn_k={k}_subsamples={num_subsamples}"
    pkl_name = f"layer={layer}_hier={hier_idx}_with_data.pkl"
    SAVEDIR = os.path.join(ROOTDIR, "analyze", knn_identifier, model_identifier, train_identifier)
    data = pickle.load(open(os.path.join(SAVEDIR + "/" + pkl_name), "rb"))
    # extract features
    feat = torch.tensor(data['data']).to(device)
    labels = torch.tensor(np.ravel(np.stack(data['targets']))).to(device)

    # reduce the dimensionality of features
    low_dim_num = 10
    pca_type = 'fixed'
    u, s, v = torch.svd(feat)
    # keep 85% variance explained ,
    idx_85 = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2) <= .95
    cols = list(torch.where(idx_85)[0].cpu().numpy())
    if pca_type == 'fixed':
        feat_pca = torch.matmul(feat, v[:, :low_dim_num])
    elif pca_type == 'equal_var':
        feat_pca = torch.matmul(feat, v[:, cols])
    var_explained = torch.cumsum(torch.cat((torch.tensor([0], device=feat.device, dtype=feat.dtype), s ** 2)),
                                 dim=0) / torch.sum(s ** 2)


    samples = np.linspace(0, labels.shape[0] - 1, 30000).astype('int')
    feat_subsample = feat_pca[samples, :]
    label_subsample = labels[samples]

    poincare_coord, _ = compute_poincare_maps(feat_subsample.cpu(), label_subsample.cpu(),
                                              f"{SAVEDIR}/poincare_maps/p_map_data_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_sparse",
                                              mode='features', k_neighbours=15,
                                              distlocal='minkowski', sigma=1.0, gamma=2.0,
                                              color_dict=None, epochs=1000,
                                              batchsize=-1, lr=0.1, earlystop=0.0001, cuda=1)
    model = PoincareMaps(poincare_coord)
    model.plot('ori', labels=label_subsample.cpu(),
               file_name=f"{SAVEDIR}/poincare_maps/p_map_plot_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_sparse",
               title_name='Poincaré map', coldict=None, labels_order=None, zoom=None, bbox=(1.1, 0.8))
    model.iroot = poincare_root(1, label_subsample.cpu(), feat_subsample.cpu())
    # we could also just explicitly say with respect to each point we want the rotation (via index of this point),
    # e.g. model.iroot = 0

    model.rotate()
    # now we can easily plot the rotation
    model.plot('rot', labels=label_subsample.cpu(),
               file_name=f"{SAVEDIR}/poincare_maps/p_map_rotated_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_sparse",
               title_name='Poincaré map',
               coldict=None,
               d1=5, d2=5, fs=9, ms=10, alpha=0.8,
               labels_order=None,
               print_labels=True,
               zoom=None, bbox=(1.1, 0.8))

    # early phase
    feat_subsample = feat_pca[0:40000, :]
    label_subsample = labels[0:40000]
    poincare_coord, _ = compute_poincare_maps(feat_subsample.cpu(), label_subsample.cpu(),
                                              f"{SAVEDIR}/poincare_maps/p_map_data_{layer}_hier_{hier_idx}_mode_early",
                                              mode='features', k_neighbours=15,
                                              distlocal='minkowski', sigma=1.0, gamma=2.0,
                                              color_dict=None, epochs=1000,
                                              batchsize=-1, lr=0.1, earlystop=0.0001, cuda=1)

    model = PoincareMaps(poincare_coord)
    model.plot('ori', labels=label_subsample.cpu(),
               file_name=f"{SAVEDIR}/poincare_maps/p_map_plot_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_early",
               title_name='Poincaré map', coldict=None, labels_order=None, zoom=None, bbox=(1.1, 0.8))
    model.iroot = poincare_root(1, label_subsample.cpu(), feat_subsample.cpu())
    # we could also just explicitly say with respect to each point we want the rotation (via index of this point),
    # e.g. model.iroot = 0

    model.rotate()
    # now we can easily plot the rotation
    model.plot('rot', labels=label_subsample.cpu(),
               file_name=f"{SAVEDIR}/poincare_maps/p_map_rotated_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_early",
               title_name='Poincaré map',
               coldict=None,
               d1=5, d2=5, fs=9, ms=10, alpha=0.8,
               labels_order=None,
               print_labels=True,
               zoom=None, bbox=(1.1, 0.8))

    # poincare map of only 1 class
    samples = torch.where(labels == 1)
    feat_subsample = feat_pca[samples[0], :]
    label_subsample = labels[samples[0]]
    poincare_coord, _ = compute_poincare_maps(feat_subsample.cpu(), label_subsample.cpu(),
                                              f"{SAVEDIR}/poincare_maps/p_map_data_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_group_1",
                                              mode='features', k_neighbours=15,
                                              distlocal='minkowski', sigma=1.0, gamma=2.0,
                                              color_dict=None, epochs=1000,
                                              batchsize=-1, lr=0.1, earlystop=0.0001, cuda=1)
    model = PoincareMaps(poincare_coord)
    model.plot('ori', labels=label_subsample.cpu(),
               file_name=f"{SAVEDIR}/poincare_maps/p_map_plot_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_group_1",
               title_name='Poincaré map', coldict=None, labels_order=None, zoom=None, bbox=(1.1, 0.8))
    model.iroot = poincare_root(1, label_subsample.cpu(), feat_subsample.cpu())
    # we could also just explicitly say with respect to each point we want the rotation (via index of this point),
    # e.g. model.iroot = 0

    model.rotate()
    # now we can easily plot the rotation
    model.plot('rot', labels=label_subsample.cpu(),
               file_name=f"{SAVEDIR}/poincare_maps/p_map_rotated_{layer}_hier_{hier_idx}_pca_{pca_type}_mode_group_1",
               title_name='Poincaré map',
               coldict=None,
               d1=5, d2=5, fs=9, ms=10, alpha=0.8,
               labels_order=None,
               print_labels=True,
               zoom=None, bbox=(1.1, 0.8))


    print('done')






