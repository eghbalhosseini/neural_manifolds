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
parser.add_argument('n_class',type=int,default=50)
parser.add_argument('exm_per_class',type=int,default=100)
args=parser.parse_args()


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CFAR100_fake_dataset_mftma(data_dir=os.path.join(data_dir, args.datafile))
    # extract samples :
    sampled_classes = args.n_class
    examples_per_class = args.exm_per_class
    data = make_manifold_data(train_dataset, sampled_classes, examples_per_class, seed=0)
    data = [d.to(device) for d in data]
    # create the model
    model_save_path=save_dir+'VGG16_synthdata_'+train_dataset.structure+'_nclass_'+str(int(train_dataset.n_class))+'_n_exm_'+str(int(train_dataset.exm_per_class))
    model = models.vgg16(num_classes=train_dataset.n_class)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model = model.eval()
    activations = extractor(model, data, layer_types=['Conv2d', 'Linear'])
    result_save_path = save_dir + 'activation_VGG16_synthdata_' + train_dataset.structure + '_nclass_' + str(sampled_classes) + '_exm_per_class_' + str(examples_per_class)

    data_ = {'activations': activations,
             'n_class':sampled_classes,
             'exm_per_class':examples_per_class,
             'network_dir':model_save_path,
             'dataset_dir':os.path.join(data_dir, args.datafile)}

    save_dict(data_, result_save_path)
    print(result_save_path)