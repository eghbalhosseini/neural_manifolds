# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:08:09 2020

@author: greta
"""
save_dir='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/'
#data_dir='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/data/'
data_dir='/Users/eghbalhosseini/Desktop/'
analyze_dir='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/'
result_dir='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/result/'

from utils.model_utils import sub_data, NN, leaf_traverse, add_layer_names
from utils import model_utils
import itertools
import copy
import os
import re
from importlib import import_module
import numpy as np

def load_train(train_name):
    return train_pool[train_name]()


#############TRAINING ###########################
class params:
    def __init__(self,datafile=None,model=None,train_type='train_test',identifier=None,beta=0.0,sigma=0.0,nclass=0,nobj=0,nhier=0,
                 shape=(1,1,1),structure=None,nfeat=0,stop_criteria='test_performance'):
        ##### DATA ####
        self.datafile=datafile
        self.identifier=identifier
        self.beta=beta
        self.sigma=sigma
        self.nclass=nclass
        self.nobj=nobj
        self.nhier=nhier
        self.shape=shape
        self.structure=structure
        # TODO make this a method and only save the path here
        self.dataset_path=os.path.join(data_dir, self.datafile)
        #self.dataset=sub_data(data_path=os.path.join(data_dir, self.datafile))
        #
        self.model = model
        self.train_type = train_type
        self.nfeat = nfeat
        self.stop_criteria=stop_criteria

    def load_dataset(self):
        self.dataset = sub_data(data_path=self.dataset_path)
    def get_layer_names(self):
        childrens=[]
        leaf_traverse(self.model(),childrens)
        add_layer_names(childrens)
        layer_names=['layer_0_Input']
        [layer_names.append(x.layer_name) for x in childrens]
        self.layer_names=layer_names
        return layer_names
    #####  Training specs #####
    #exm_per_class = 100  # examples per class
    batch_size_train = 32
    batch_size_test = 32
    epochs = 4
    momentum = 0.5
    lr = 0.001
    log_interval = 30 # when to save, extract, and test the data
    test_split = .2
    shuffle_dataset = True
    random_seed = 1
data_config=[]
# Creating tags for training paradigm
data_structure=[dict(struct='partition',nclass=64,n_hier=1,shape=(1,936)),
                dict(struct='tree',nclass=64,n_hier=6,shape=(1,936))]
for idx, structure in enumerate(data_structure):
    for beta in ['0.0160']: # ['0.0000','0.0160','0.0330','0.0500']
        for sigma in ['0.8330']: # ['0.0000','0.8330','1.6670','2.5000']
            nfeat=np.prod(structure['shape'])
            data_file=f"synth_{structure['struct']}_nobj_{structure['nclass']*1000}_nclass_{structure['nclass']}_nhier_{structure['n_hier']}_nfeat_{nfeat}_beta_{beta}_sigma_{sigma}_norm_1.mat"
            data_config.append(dict(data_file=data_file,shape=structure['shape']))

train_configuration = []

for dataset , model, train_type, stop_type in itertools.product(data_config,['NN','linear_NN'],['train_test'],['fixed','test_performance']):
    s = re.findall('nclass_\d+', dataset['data_file'])[0]
    nclass = int(s.split('_')[1])
    s = re.findall('nhier_\d+', dataset['data_file'])[0]
    nhier = int(s.split('_')[1])
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
    train_identifier=f"[{model}]-[{structure}/nclass={nclass}/nobj={nobj}/nhier={nhier}/beta={beta}/sigma={sigma}/nfeat={nfeat}]-[{train_type}]-[{stop_type}]"
    train_identifier=train_identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    train_configuration.append(dict(identifier=train_identifier,dataset=dataset['data_file'],shape=dataset['shape'],
                                    nclass=nclass,nobj=nobj,nhier=nhier,sigma=sigma,beta=beta,model=model,train_type=train_type,
                                    structure=structure,nfeat=nfeat,stop_criteria=stop_type))
train_pool={}
# create the pool
for config in train_configuration:
    configuration=copy.deepcopy(config)
    train_identifier=configuration['identifier']
    def train_instantiation(configure=frozenset(configuration.items())):
        configure = dict(configure)
        module = import_module('utils.model_utils')
        model=getattr(module,configure['model'])
        train_param=params(model=model,
                           datafile=configure['dataset'],
                           train_type=configure['train_type'],
                           identifier=configure['identifier'],
                           shape=configure['shape'],
                           beta=configure['beta'],
                           sigma=configure['sigma'],
                           nclass=configure['nclass'],
                           nobj=configure['nobj'],
                           nhier=configure['nhier'],
                           structure=configure['structure'],
                           nfeat=configure['nfeat'],
                           stop_criteria=configure['stop_criteria'])
        return train_param

    train_pool[train_identifier] = train_instantiation




