import numpy as np
import torch
import pickle
import sys
import os
from mftma.manifold_analysis_correlation import manifold_analysis_corr

import getpass
import argparse
from neural_manifold_utils import  save_dict
from datetime import datetime
print('__cuda available ',torch.cuda.is_available())
print('__Python VERSION:', sys.version)
print('__Number CUDA Devices:', torch.cuda.device_count())
try :
    print('__Device name:', torch.cuda.get_device_name(0))
except:
    print('no gpu to run')

user = getpass.getuser()
print(user)
if user == 'eghbalhosseini':
    save_dir = '/Users/eghbalhosseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir = '/Users/eghbalhosseini/MyData/neural_manifolds/synthetic_datasets/'

elif user == 'ehoseini':
    save_dir = '/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir = '/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'
parser = argparse.ArgumentParser(description='neural manifold test network')
parser.add_argument('train_dir', type=str, default="train_VGG16_synthdata_tree_nclass_50_n_exm_1000",help='')
parser.add_argument('epoch_id',type=int,default=1)
parser.add_argument('layer_num',type=int,default=1)
args=parser.parse_args()


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datafile = os.path.join(save_dir,args.train_dir, 'train_epoch_' + str(args.epoch_id))
    epoch_dat = pickle.load(open(datafile, 'rb'))
    activations_cell = epoch_dat['activations_cell']
    # contstruct a result dir and remove the big file
    mfmta_data_ = {'mftma_results': [],
                   'train_spec': epoch_dat['train_spec'],
                   'train_accuracy': epoch_dat['train_accuracy'],
                   'train_success': epoch_dat['train_success'],
                   'epoch': epoch_dat['epoch'],
                   'layer_num':args.layer_num
                   }
    del epoch_dat
    hier_layer_names = [list(activations.keys()) for activations in activations_cell]
    layer_id = [x[args.layer_num] for x in hier_layer_names]
    layer_activ_cell = [{layer_id[idx]: x[layer_id[idx]]} for idx, x in enumerate(activations_cell)]
    del activations_cell
    # do projection:
    # project the epoch data first
    np.random.seed(0)
    X = [np.random.randn(500, 50) for i in range(100)]  # Replace this with data to analyze
    kappa = 0
    n_t = 200
    capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(X, kappa, n_t)
    avg_capacity = 1 / np.mean(1 / capacity_all)
    avg_radius = np.mean(radius_all)
    avg_dimension = np.mean(dimension_all)
    # for hier_id, activ_hier in enumerate(layer_activ_cell):
    #     layer_names = list(activ_hier.keys())
    #     for layer, data, in activ_hier.items():
    #         X = [d.reshape(d.shape[0], -1).T for d in data]
    #         # Get the number of features in the flattened data
    #         N = X[0].shape[0]
    #         # If N is greater than 5000, do the random projection to 5000 features
    #         if N > 5000:
    #             print("Projecting {}".format(layer))
    #             M = np.random.randn(5000, N)
    #             M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
    #             X = [np.matmul(M, d) for d in X]
    #         activ_hier[layer] = X
    #     layer_activ_cell[hier_id] = activ_hier
    # run mftma on all layers and hierarchies
    # mftmas_cell = []
    # for hier_id, activ_hier in enumerate(layer_activ_cell):
    #     data_ = {'capacities': [],
    #              'radii': [],
    #              'dimensions': [],
    #              'correlations': [],
    #              'layer': [],
    #              'n_hier_class': [],
    #              'hierarchy': hier_id}
    #     capacities = []
    #     radii = []
    #     dimensions = []
    #     correlations = []
    #     data_['layer'] = layer_id[hier_id]
    #     data_['n_hier_class'] = len(activ_hier[layer_id[hier_id]])
    #     for k, X, in activ_hier.items():
    #         print(k)
    #         # Analyze each layer's activations
    #         try:
    #             a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)
    #
    #             # Compute the mean values
    #             a = 1 / np.mean(1 / a)
    #             r = np.mean(r)
    #             d = np.mean(d)
    #             print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))
    #         except:
    #             a = np.nan
    #             r = np.nan
    #             d = np.nan
    #             r0 = np.nan
    #         # Store for later
    #         capacities.append(a)
    #         radii.append(r)
    #         dimensions.append(d)
    #         correlations.append(r0)
    #     # combine the results
    #     data_['capacities'] = capacities
    #     data_['radii'] = radii
    #     data_['dimensions'] = dimensions
    #     data_['correlations'] = correlations
    #     mftmas_cell.append(data_)

    # combine the results and save them
    # mfmta_data_['mftma_results']=mftmas_cell
    # result_file = os.path.join(save_dir, args.train_dir, 'mftma_epoch_' + str(args.epoch_id)+'_layer_'+str(args.layer_num))
    # save_dict(mfmta_data_, result_file)
    print('Done!')
