# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:08:09 2020

@author: greta
"""

from neural_manifold_utils import sub_data, NN, CNN
import neural_manifold_utils
from torchvision import models
import itertools
import copy

import os
import getpass
user = getpass.getuser()
print('User is: \n', user)

if user=='eghbalhosseini':
    save_dir='/Users/eghbalhosseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/Users/eghbalhosseini/MyData/neural_manifolds/synthetic_datasets/'
elif user=='ehoseini':
    save_dir='/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'
elif user == 'gretatu':
    save_dir = '/om/user/gretatu/neural_manifolds/network_training_on_synthetic/'
    data_dir = '/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'



class params:
    def __init__(self,datafile="synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat",
                 model='CNN_open',
                 train_type='train',identifier=None):
        ##### DATA ####
        self.datafile=datafile
        self.identifier=identifier
        print('running')
        self.dataset= sub_data(data_path=os.path.join(data_dir, self.datafile))
        #
        #datafile = "synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat"
        #dataset = sub_data(data_path=os.path.join(data_dir, datafile))
        exm_per_class = 100 # examples per class
        resize = True # reshape data into a 2D array # TODO make adaptable
        #### MODEL ####
        self.models=getattr(neural_manifold_utils,model)
        model = CNN # models.vgg16(num_classes=dataset.n_class) # or CNN etc
    
        #### TRAINING ####
        self.train_type=train_type

    #model = CNN_open  # models.vgg16(num_classes=dataset.n_class) # or CNN etc
    #datafile = "synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat"
    #train_type = 'train_test'
    #### TRAINING ####
    train_type = 'train_test'

    batch_size_train = 64
    batch_size_test = 64
    epochs = 50
    momentum = 0.5
    lr = 0.01
    # gamma = 0.7 not used
    log_interval = 75 # when to save, extract, and test the data
    test_split = .2
    shuffle_dataset = True
    random_seed = 1
    save_epochs = False # save individual mat files for each chosen epoch # GET RID OF?


data_config={
'partition/nclass=100/beta0.1/sigma1.5':'synth_partition_nobj_100000_nclass_100_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat',
'partition/nclass=50/beta0.01/sigma1.5':'synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat',
#'partition/nclass=64/beta0.00/sigma0.83':'synth_partition_nobj_64000_nclass_64_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat',
#'partition/nclass=64/beta0.02/sigma0.83':'synth_partition_nobj_64000_nclass_64_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat',
#'partition/nclass=64/beta0.02/sigma2.5':'synth_partition_nobj_64000_nclass_64_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat'
}
train_configuration=[]
for dataset , model, train_type in itertools.product(list(data_config.keys()),['CNN_open'],['train']):
    identifier=f"[{model}]-[{dataset}]-[{train_type}]"
    train_configuration.append(dict(identifier=identifier,dataset=data_config[dataset],model=model,train_type=train_type))

train_pool={}
# create the pool
for config in train_configuration:
    configuration=copy.deepcopy(config)
    identifier=configuration['identifier']
    def train_instantionation(identfier=identifier,configure=frozenset(configuration.items())):
        configure = dict(configure)
        model=configure['model']
        datafile=configure['dataset']
        train_type=configure['train_type']
        train_param=params(model=model,datafile=datafile,train_type=train_type,identifier=identifier)
        return train_param

    train_pool[identifier]=train_instantionation



