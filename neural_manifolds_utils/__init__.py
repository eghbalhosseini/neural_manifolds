# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:08:09 2020

@author: greta
"""
from neural_manifolds_utils.neural_manifold_utils import sub_data, NN
from neural_manifolds_utils import neural_manifold_utils
import itertools
import copy
import os
import getpass
user = getpass.getuser()
import re
from importlib import import_module


if user=='eghbalhosseini':
    save_dir='/Users/eghbalhosseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/Users/eghbalhosseini/MyData/neural_manifolds/synthetic_datasets/'
elif user=='ehoseini':
    save_dir='/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'
elif user == 'gretatu':
    save_dir = '/om/user/gretatu/neural_manifolds/network_training_on_synthetic/'
    data_dir = '/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'

def load_train(train_name):
    return train_pool[train_name]()

class params:
    def __init__(self,datafile=None,model=None,train_type='train',identifier=None,beta=0,sigma=0,nclass=0,nobj=0,shape=(1,1,1),structure=None):
        ##### DATA ####
        self.datafile=datafile
        self.identifier=identifier
        self.beta=beta
        self.sigma=sigma
        self.nclass=nclass
        self.nobj=nobj
        self.shape=shape
        self.structure=structure
        self.dataset= sub_data(data_path=os.path.join(data_dir, self.datafile))
        self.model=model
        self.train_type=train_type
    #training_spec
    resize = True  # reshape data into a 2D array # TODO make adaptable
    exm_per_class = 100  # examples per class
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

data_config=[{'data_file':'synth_partition_nobj_100000_nclass_100_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat','shape':(3,32,32)},
             {'data_file':'synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat','shape':(3,32,32)} ]


train_configuration=[]
for dataset , model, train_type in itertools.product(data_config,['NN'],['train_test']):
    s = re.findall('nclass_\d+', dataset['data_file'])[0]
    nclass = int(s.split('_')[1])
    s = re.findall('synth_\w+_nobj', dataset['data_file'])[0]
    structure = (s.split('_')[1])
    s = re.findall('beta_\d+\.\d+', dataset['data_file'])[0]
    beta=float(s.split('_')[1])
    s = re.findall('sigma_\d+\.\d+', dataset['data_file'])[0]
    sigma = float(s.split('_')[1])
    s = re.findall('nobj_\d+', dataset['data_file'])[0]
    nobj = int(s.split('_')[1])
    s = re.findall('nfeat_\d+', dataset['data_file'])[0]
    nfeat = int(s.split('_')[1])
    identifier=f"[{model}]-[{structure}/nclass={nclass}/nobj={nobj}/beta={beta}/sigma={sigma}/nfeat={nfeat}]-[{train_type}]"
    train_configuration.append(dict(identifier=identifier,dataset=dataset['data_file'],shape=dataset['shape'],
                                    nclass=nclass,nobj=nobj,sigma=sigma,beta=beta,model=model,train_type=train_type,structure=structure))

train_pool={}
# create the pool
for config in train_configuration:
    configuration=copy.deepcopy(config)
    identifier=configuration['identifier']

    def train_instantionation(identfier=identifier,configure=frozenset(configuration.items())):
        configure = dict(configure)
        module = import_module('neural_manifolds_utils.neural_manifold_utils')
        model=getattr(module,configure['model'])
        train_param=params(model=model,
                           datafile=configure['dataset'],
                           train_type=configure['train_type'],
                           identifier=identifier,
                           shape=configure['shape'],
                           beta=configure['beta'],
                           sigma=configure['sigma'],
                           nclass=configure['nclass'],
                           nobj=configure['nobj'],
                           structure=configure['structure'])
        return train_param

    train_pool[identifier]=train_instantionation




# TODO , think about how to make test examples seperate from training examples

analyze_method=['mftma','super_class','dating']
exm_per_class=[50,100]

analyze_configuration=[]
for analyze_meth , exm in itertools.product(analyze_method,exm_per_class):

    identifier=f"[{analyze_meth}]-[exm_per_class={exm}]"
    analyze_configuration.append(dict(identifier=identifier,exm_per_class=exm,analysis_method=analyze_meth))


analyze_pool={}
# create the pool




