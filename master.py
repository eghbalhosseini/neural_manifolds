# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:08:09 2020

@author: greta
"""
import neural_manifold_utils
from neural_manifold_utils import sub_data, NN_open, CNN_open, LazyLoad
from torchvision import models
import itertools

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
                 train_type='train'):
        ##### DATA ####
        self.datafile=datafile
        self.dataset= sub_data(data_path=os.path.join(data_dir, self.datafile))
        #
        datafile = "synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat"
        dataset = sub_data(data_path=os.path.join(data_dir, datafile))
        exm_per_class = 100 # examples per class
        resize = True # reshape data into a 2D array # TODO make adaptable
        #### MODEL ####
        self.models=getattr(neural_manifold_utils,model)
        model = CNN_open # models.vgg16(num_classes=dataset.n_class) # or CNN etc
    
        #### TRAINING ####
        self.train_type=train_type

    model = CNN_open  # models.vgg16(num_classes=dataset.n_class) # or CNN etc
    datafile = "synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat"
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

train_configuration=[]
for dataset , model, train_type in itertools.product(
        ['synth_partition_nobj_100000_nclass_100_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat',
         'synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat',
         'synth_partition_nobj_64000_nclass_64_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat',
         'synth_partition_nobj_64000_nclass_64_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat',
         'synth_partition_nobj_64000_nclass_64_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat',
         'synth_partition_nobj_96000_nclass_96_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat'
         'synth_partition_nobj_96000_nclass_96_nfeat_3072_beta_0.00_sigma_2.50_norm_1.mat',
         'synth_partition_nobj_96000_nclass_96_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat',
         'synth_partition_nobj_96000_nclass_96_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat']
        ,['CNN_open','VGG16'],['train','train_test']):
    train_configuration.append(dict(dataset=dataset,model=model,train_type=train_type))


# create the pool
#for config in train_configuration:



