import numpy as np
import pickle
import sys
import os
from mftma.manifold_analysis_correlation import manifold_analysis_corr

def run_mftma(data,layer_idx=0,kappa=0,n_t=300,n_reps=1):
    mfmta_data_ = {'mftma_results': []}
    layer_n = [list(x.keys()) for x in data]
    layer_id = [x[layer_idx] for x in layer_n]
    layer_activ_cell = [{layer_id[idx]: x[layer_id[idx]]} for idx, x in enumerate(data)]
    # run mftma on all layers and hierarchies
    mftmas_cell = []
    for hier_id, activ_hier in enumerate(layer_activ_cell):

        data_ = {'capacities': [],'radii': [],'dimensions': [],'correlations': [],'layer': [],'n_hier_class': [],'hierarchy': hier_id}
        capacities = []
        radii = []
        dimensions = []
        correlations = []
        data_['layer'] = layer_id[hier_id]
        data_['n_hier_class'] = len(activ_hier[layer_id[hier_id]])
        for k, X, in activ_hier.items():
            # Analyze each layer's activations

            a, r, d, r0, K = manifold_analysis_corr(X, kappa, n_t, n_reps)
            # Compute the mean values
            a = 1 / np.mean(1 / a)
            r = np.mean(r)
            d = np.mean(d)
            print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))

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
    #combine the results and save them
    #mfmta_data_['mftma_results']=mftmas_cell
