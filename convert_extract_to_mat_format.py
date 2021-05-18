
from utils import save_dir,analyze_dir, train_pool,save_dict
from utils.analysis_utils import analyze_pool
import pickle
import os
import argparse
import numpy as np
import fnmatch
import re
import scipy.io as sio
parser = argparse.ArgumentParser(description='run covar and save results')
parser.add_argument('model_id', type=str, default='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str, default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
parser.add_argument('train_id', type=str, default='epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06')
parser.add_argument('overwrite',type=str,default='false')
args = parser.parse_args()

if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version: of the data
    model_identifier = args.model_id
    analyze_identifier = args.analyze_id
    train_dir_identifier = args.train_id
    overwrite = args.overwrite
    # for testing the code
    #save_dir='/Users/eghbalhosseini/Desktop'
    #model_identifier = "NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed"
    #analyze_identifier = "mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5"
    #train_dir_identifier = "epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06"
    #distance_metric='cosine'

    #overwrite = "true"

    #
    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    # find pth files
    extracted_files = []
    for file in os.listdir(os.path.join(save_dir, analyze_identifier_for_saving, model_identifier_for_saving,
                                        train_dir_identifier)):
        if fnmatch.fnmatch(file, '*_extracted_v3.pkl'):
            extracted_files.append(os.path.join(save_dir, analyze_identifier_for_saving, model_identifier_for_saving,
                                            train_dir_identifier, file))
    s = [re.findall('/\d+', x) for x in extracted_files]
    s = [item for sublist in s for item in sublist]
    file_id = [int(x.split('/')[1]) for x in s]
    sorted_files = [extracted_files[x] for x in np.argsort(file_id)]

    for x in sorted_files:
        y = x.replace('.pkl', '.mat')
        data = pickle.load(open(x, 'rb'))
        sio.savemat(y, {'activation': data})
    print('done')

