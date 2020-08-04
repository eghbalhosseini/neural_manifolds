from utils.analysis_utils import run_mftma
import torch
import pickle
import os
if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    #
    model_identifier='[NN]-[partition/nclass=100/nobj=100000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[fixed]'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file='/Users/eghbalhosseini/Desktop/projection_test.pkl'

    with open(data_file) as x:
        data = pickle.load(open(data_file, 'rb'))
    projection_data_=data['projection_results']
    # run mftma
    results=run_mftma(projection_data_, layer_idx=2,kappa=0, n_t=300, n_reps=1)
    # projection step :
    pass