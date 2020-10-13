# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:08:09 2020

@author: greta
"""
save_dir='/om/group/evlab/Greta_Eghbal_manifolds/extracted/'
data_dir='/om/group/evlab/Greta_Eghbal_manifolds/data/'
analyze_dir='/om/group/evlab/Greta_Eghbal_manifolds/analyze/'

from utils.model_utils import sub_data, NN, leaf_traverse, add_layer_names
from utils import model_utils
from utils.analysis_utils import mftmaAnalysis, knnAnalysis
import itertools
import copy
import os
import re
from importlib import import_module

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
    batch_size_train = 64
    batch_size_test = 64
    epochs = 3
    momentum = 0.5
    lr = 0.01
    log_interval = 15 # when to save, extract, and test the data
    test_split = .2
    shuffle_dataset = True
    random_seed = 1

# Creating tags for training paradigm
data_config = [{'data_file':'synth_partition_nobj_100000_nclass_100_nhier_1_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_partition_nobj_50000_nclass_50_nhier_1_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_partition_nobj_64000_nclass_64_nhier_1_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_partition_nobj_64000_nclass_64_nhier_1_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_partition_nobj_64000_nclass_64_nhier_1_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_partition_nobj_96000_nclass_96_nhier_1_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_partition_nobj_96000_nclass_96_nhier_1_nfeat_3072_beta_0.00_sigma_2.50_norm_1.mat','shape': (1, 3072)},
                {'data_file':'synth_partition_nobj_96000_nclass_96_nhier_1_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat','shape': (1, 3072)},
                {'data_file':'synth_partition_nobj_96000_nclass_96_nhier_1_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat','shape': (1, 3072)},
                {'data_file':'synth_tree_nobj_50000_nclass_50_nhier_3_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat','shape':(1,3072)},
                {'data_file':'synth_tree_nobj_100000_nclass_100_nhier_4_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat','shape':(1,3072)},
               {'data_file': 'synth_tree_nobj_64000_nclass_64_nhier_6_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_64000_nclass_64_nhier_6_nfeat_3072_beta_0.00_sigma_2.50_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_64000_nclass_64_nhier_6_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_64000_nclass_64_nhier_6_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_96000_nclass_96_nhier_6_nfeat_3072_beta_0.00_sigma_0.83_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_96000_nclass_96_nhier_6_nfeat_3072_beta_0.00_sigma_2.50_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_96000_nclass_96_nhier_6_nfeat_3072_beta_0.02_sigma_0.83_norm_1.mat','shape': (1, 3072)},
               {'data_file': 'synth_tree_nobj_96000_nclass_96_nhier_6_nfeat_3072_beta_0.02_sigma_2.50_norm_1.mat','shape': (1, 3072)}]



train_configuration = []

for dataset , model, train_type, stop_type in itertools.product(data_config,['NN'],['train_test'],['fixed','test_performance']):
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


#############ANALYSIS ###########################
# Creating tags for analysis paradigms
analyze_pool={}

analyze_method=['mftma']
n_ts=[300]
kappas=[0]
n_reps=[1]
n_projs=[5000]

exm_per_class=[20,50,100]
projection_flag=[False,True]
randomize=[True,False]

analyze_configuration=[]
for method , n_t,n_rep, kappa, exm ,proj_flag,rand_flag,n_proj in itertools.product(analyze_method,n_ts,n_reps,kappas,exm_per_class,projection_flag,randomize,n_projs):
    identifier=f"[{method}]-[exm_per_class={exm}]-[proj={proj_flag}]-[rand={rand_flag}]-[kappa={kappa}]-[n_t={n_t}]-[n_rep={n_rep}]"
    identifier=identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    analyze_configuration.append(dict(identifier=identifier,exm_per_class=exm,method=method,project=proj_flag,randomize=rand_flag,kappa=kappa,n_t=n_t,n_rep=n_rep,n_project=n_proj))


# create the pool
for config in analyze_configuration:
    configuration=copy.deepcopy(config)
    analyze_identifier=configuration['identifier']
    def analyze_instantiation(identfier=analyze_identifier,configure=frozenset(configuration.items())):
        configure = dict(configure)
        analyze_param=mftmaAnalysis(analyze_method=configure['method'],
                           exm_per_class=configure['exm_per_class'],
                           identifier=configure['identifier'],
                           n_t=configure['n_t'],
                           kappa=configure['kappa'],
                           n_rep=configure['n_rep'],
                           randomize=configure['randomize'],
                           project=configure['project'],
                          n_project=configure['n_project'])
        return analyze_param

    analyze_pool[analyze_identifier] = analyze_instantiation


analyze_configuration=[]
analyze_method=['knn']
ks=[100]
nums_subsamples=[100,200,500]
dist_metric=['euclidean','cosine']

for method , k,dist_m,num_subsamples in itertools.product(analyze_method,ks,dist_metric,nums_subsamples):
    identifier=f"[{method}]-[k={k}]-[dist_metric={dist_m}]-[num_subsamples={num_subsamples}]"
    identifier=identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    analyze_configuration.append(dict(identifier=identifier,k=k,dist_metric=dist_m,num_subsamples=num_subsamples))

# create the pool
for config in analyze_configuration:
    configuration=copy.deepcopy(config)
    analyze_identifier=configuration['identifier']
    def analyze_instantiation(identfier=analyze_identifier,configure=frozenset(configuration.items())):
        configure = dict(configure)
        analyze_param=knnAnalysis(identifier=configure['identifier'],
                           k=configure['k'],
                           num_subsamples=configure['num_subsamples'],
                                  distance_metric=configure['dist_metric'])
        return analyze_param

    analyze_pool[analyze_identifier] = analyze_instantiation




