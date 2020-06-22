import numpy as np
import torch
import sys
import os
from torchvision import models
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze
import getpass
import argparse
from neural_manifold_utils import CFAR100_fake_dataset_mftma , save_dict
from datetime import datetime
print('__cuda available ',torch.cuda.is_available())
print('__Python VERSION:', sys.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())

user=getpass.getuser()
print(user)
if user=='eghbalhosseini':
    save_dir='/Users/eghbalhosseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/Users/eghbalhosseini/MyData/neural_manifolds/synthetic_datasets/'
elif user=='ehoseini':
    save_dir='/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'

parser = argparse.ArgumentParser(description='neural manifold test network')
parser.add_argument('datafile', type=str, default="synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat",help='')
args=parser.parse_args()


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CFAR100_fake_dataset_mftma(data_dir=os.path.join(data_dir, args.datafile))
    # extract samples :
    sampled_classes = train_dataset.n_class
    examples_per_class = 100
    data = make_manifold_data(train_dataset, sampled_classes, examples_per_class, seed=0)
    data = [d.to(device) for d in data]
    # create the model
    model_save_path=save_dir+'VGG16_synthdata_'+train_dataset.structure+'_nclass_'+str(int(train_dataset.n_class))+'_n_exm_'+str(int(train_dataset.exm_per_class))
    model = models.vgg16(num_classes=train_dataset.n_class)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model = model.eval()
    activations = extractor(model, data, layer_types=['Conv2d', 'Linear'])
    list(activations.keys())
    for layer, data, in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]
        activations[layer] = X

    capacities = []
    radii = []
    dimensions = []
    correlations = []

    for k, X, in activations.items():
        # Analyze each layer's activations
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)

        # Compute the mean values
        a = 1 / np.mean(1 / a)
        r = np.mean(r)
        d = np.mean(d)
        print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))

        # Store for later
        capacities.append(a)
        radii.append(r)
        dimensions.append(d)
        correlations.append(r0)
    names = list(activations.keys())
    names = [n.split('_')[1] + ' ' + n.split('_')[2] for n in names]
    # save the results:
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    results_file = os.path.join(save_dir,'mftma_'+model_save_path+'_'+current_time)

    data_ = {'capacities': capacities,
             'radii': radii,
             'dimensions': dimensions,
             'correlations': correlations,
             'names': names,
             'analyze_exm_per_class': examples_per_class,
             'analyze_n_class': sampled_classes
             }

    result_save_path = save_dir + 'mftma_VGG16_synthdata_' + train_dataset.structure + '_nclass_' + str(
        int(train_dataset.n_class)) + '_n_exm_' + str(int(train_dataset.exm_per_class)) + '_' + current_time

    save_dict(data_, result_save_path)
