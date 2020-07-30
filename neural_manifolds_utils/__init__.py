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
    def __init__(self,datafile=None,model=None,train_type='train',identifier=None,beta=0.0,sigma=0.0,nclass=0,nobj=0,
                 shape=(1,1,1),structure=None,nfeat=0,stop_criteria='test_performence'):
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
        self.nfeat=nfeat
        self.stop_criteria=stop_criteria
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
for dataset , model, train_type, stop_type in itertools.product(data_config,['NN'],['train_test'],['fixed','test_performance']):
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
    train_identifier=f"[{model}]-[{structure}/nclass={nclass}/nobj={nobj}/beta={beta}/sigma={sigma}/nfeat={nfeat}]-[{train_type}]-[{stop_type}]"
    train_configuration.append(dict(identifier=train_identifier,dataset=dataset['data_file'],shape=dataset['shape'],
                                    nclass=nclass,nobj=nobj,sigma=sigma,beta=beta,model=model,train_type=train_type,
                                    structure=structure,nfeat=nfeat,stop_criteria=stop_type))

train_pool={}
# create the pool
for config in train_configuration:
    configuration=copy.deepcopy(config)
    train_identifier=configuration['identifier']
    #TODO model to device doesnt work
    def train_instantionation(identfier=train_identifier,configure=frozenset(configuration.items())):
        configure = dict(configure)
        module = import_module('neural_manifolds_utils.neural_manifold_utils')
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
                           structure=configure['structure'],
                           nfeat=configure['nfeat'],
                           stop_criteria=configure['stop_criteria'])
        return train_param

    train_pool[train_identifier]=train_instantionation

# Define extraction pool
extract_pool={}

extract_method=['mftma_extract']
exm_per_class=[20,50,100]
projection_flag=[False,True]

extract_configuration=[]
for extract_meth , exm, p_flag in itertools.product(extract_method,exm_per_class,projection_flag):
    identifier=f"[{extract_meth}]-[exm_per_class={exm}]-[Proj={p_flag}]"
    extract_configuration.append(dict(identifier=identifier,exm_per_class=exm,extract_method=extract_meth,project=p_flag))

[extract_pool.update({x['identifier']:x}) for x in extract_configuration]

# define the analysis pool
analyze_pool={}
analyze_method=['mftma']
n_ts=[300]
kappas=[0]
n_reps=[1]
analyze_configuration=[]
for analy_meth , n_t, kappa,n_rep in itertools.product(analyze_method,n_ts,kappas,n_reps):
    identifier=f"[{analy_meth}]-[n_t={n_t}]-[kappa={kappa}]-[n_rep={n_rep}]"
    analyze_configuration.append(dict(identifier=identifier,n_t=n_t,analyze_method=analy_meth,kappa=kappa,n_rep=n_rep))


[analyze_pool.update({x['identifier']:x}) for x in analyze_configuration]



