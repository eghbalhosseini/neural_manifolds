import numpy as np
import pickle
import sys
import os
from mftma.manifold_analysis_correlation import manifold_analysis_corr

def run_mftma(layer_data,kappa=0,n_t=300,n_reps=1):
    mfmta_data_ = {'mftma_results': []}
    # run mftma on all layers and hierarchies
    mftmas_cell = []
    for hier_id, activ_hier in enumerate(layer_data):
        data_ = {'capacities': [],'radii': [],'dimensions': [],'correlations': [],'layer': [],'n_hier_class': [],'hierarchy': hier_id}
        capacities = []
        radii = []
        dimensions = []
        correlations = []
        for k, X, in activ_hier.items():
            data_['layer'] = k
            data_['n_hier_class'] = len(X)
            # Analyze each layer's activations
            try:
                a, r, d, r0, K = manifold_analysis_corr(X, kappa, n_t)
            # Compute the mean values
                a = 1 / np.mean(1 / a)
                r = np.mean(r)
                d = np.mean(d)
                print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))
            except :
                a = np.nan
                r = np.nan
                d = np.nan
                r0 = np.nan
            # Store for later
            capacities.append(a)
            radii.append(r)
            dimensions.append(d)
            correlations.append(r0)
        # combine the results
            data_['capacities'] = capacities
            data_['radii'] = radii
            data_['dimensions'] = dimensions
            data_['correlations'] = correlations
            mftmas_cell.append(data_)
    return mftmas_cell

class mftmaAnalysis:
    def __init__(self,analyze_method=None,exm_per_class=None,identifier=None,n_t=None,kappa=None,n_rep=None,randomize=None,project=None,n_project=5000,save_mat=False):
        ##### DATA ####
        self.analyze_method=analyze_method
        self.exm_per_class=exm_per_class
        self.randomize=randomize
        self.project=project
        self.n_project=n_project
        self.identifier=identifier
        self.n_t=n_t
        self.kappa=kappa
        self.n_rep=n_rep
        self.save_mat=save_mat





class knnAnalysis:
    def __init__(self,identifier=None,save_fig=True,distance_metric=None,k=100,num_subsamples=100):
        ##### DATA ####
        self.identifier=identifier
        self.k=k
        self.distance_metric=distance_metric
        self.num_subsamples=num_subsamples
        self.save_fig=save_fig


    #####  Training specs #####
    #batch_size_train = 64
    #batch_size_test = 64
    #epochs = 3
    #log_interval = 15 # when to save, extract, and test the data
    #test_split = .2
    #shuffle_dataset = True
    #random_seed = 1