from utils.analysis_utils import run_mftma

from utils import save_dir, data_dir,analyze_dir, train_pool,save_dict
from utils.analysis_utils import analyze_pool
import torch
import pickle
import os
import argparse
import numpy as np
import scipy.spatial.distance as dist
parser = argparse.ArgumentParser(description='run covar and save results')
parser.add_argument('file_id', type=str, default='')
parser.add_argument('model_id', type=str, default='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str, default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
parser.add_argument('overwrite',type=str,default='false')
args = parser.parse_args()

if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version: of the data
    file_id = args.file_id
    model_identifier = args.model_id
    analyze_identifier = args.analyze_id
    overwrite = args.overwrite
    #
    file_parts = file_id.split('/')
    data_dir = '/'.join(file_parts[:-1])

    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    analyze_params = analyze_pool[analyze_identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    results_dir = data_dir.replace(save_dir, os.path.join(analyze_dir+'/'))
    #
    # check if path exists
    if not os.path.exists(os.path.join(analyze_dir,analyze_identifier)):
        try:
            os.mkdir(os.path.join(analyze_dir,analyze_identifier))
        except:
            print(f'looks like the folder {os.path.join(analyze_dir,analyze_identifier)} already exists \n')
    if not os.path.exists(os.path.join(analyze_dir,analyze_identifier,model_identifier)):
        try:
            os.mkdir(os.path.join(analyze_dir,analyze_identifier,model_identifier))
        except:
            print(f'looks like the folder {os.mkdir(os.path.join(analyze_dir,analyze_identifier,model_identifier))} already exists \n')
    if not os.path.exists(results_dir):
        try:
            os.mkdir(results_dir)
        except:
            print(f'looks like the folder {results_dir} already exists \n')


    file_parts=file_id.split('/')
    extracted_data = pickle.load(open(file_id, 'rb'))
    projection_data_ = extracted_data['projection_results']
    # create outputfile
    covar_file = os.path.join(results_dir,file_parts[-1])
    covar_file = covar_file.replace("_extracted.pkl", '_center_covar.pkl')
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check whether file exist
    do_analysis=True
    if 'false' in overwrite:
         if os.path.exists(covar_file):
            do_analysis=False
         else:
             do_analysis=True
    #
    if do_analysis:
    # run mftma

        covar_results = []
        for hier_id, activ_hier in enumerate(projection_data_):
            data_ = {'layer': [], 'n_hier_class': [], 'hierarchy': hier_id}
            center_cov_all=[]
            for k, X, in activ_hier.items():
                data_['layer'] = k
                data_['n_hier_class'] = len(X)
                
                centers = [np.mean(X[i], axis=1) for i in range(len(X))]
                centers = np.stack(centers, axis=1)  # Centers is of shape (N, m) for m manifolds
                center_cov=dist.squareform(dist.pdist(np.transpose(centers)))
                #center_cov=np.cov(np.transpose(centers))
                center_cov_all.append(center_cov)
            data_['center_cov']=center_cov_all
            covar_results.append(data_)


        # save results:
        covar_file=os.path.join(results_dir,file_parts[-1])
        covar_file = covar_file.replace("_extracted.pkl", '_center_covar.pkl')
        #print(mftma_file)
        #
        d_master = {'covar_results': covar_results,
                    'analyze_identifier': analyze_identifier,
                    'model_identifier': model_identifier,
                    'layer_name': extracted_data['layer_name'],
                    'train_acc': extracted_data['train_acc'],
                    'test_acc': extracted_data['test_acc'],
                    'epoch': extracted_data['epoch'],
                    'files_generated': covar_file}
        save_dict(d_master, covar_file)
        print('done!')
    else:
        print('file already exists, abort')
        pass

