from utils.analysis_utils import run_mftma
import pickle
save_dir = '/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/'
data_dir = '/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/data/'
analyze_dir = '/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/'

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

from utils import train_pool
from utils.analysis_utils import analyze_pool
import torch

import os
import argparse
parser = argparse.ArgumentParser(description='run mftma and save results')
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
    #
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
    mftma_file = os.path.join(results_dir,file_parts[-1])
    mftma_file = mftma_file.replace("_extracted_v3.pkl", '_mftma_analysis_v3.pkl')
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check whether file exist
    do_analysis=True
    if 'false' in overwrite:
         if os.path.exists(mftma_file):
            do_analysis=False
         else:
             do_analysis=True
    #
    if do_analysis:
    # run mftma
        mftma_results = run_mftma(projection_data_, kappa=analyze_params.kappa, n_t=analyze_params.n_t, n_reps=analyze_params.n_rep)
        # save results:
        mftma_file=os.path.join(results_dir,file_parts[-1])
        mftma_file = mftma_file.replace("_extracted_v3.pkl", '_mftma_analysis_v3.pkl')
        #print(mftma_file)
    #
        d_master = {'mftma_results': mftma_results,
                 'analyze_identifier': analyze_identifier,
                 'model_identifier': model_identifier,
                 'layer_name': extracted_data['layer_name'],
                 'train_acc': extracted_data['train_acc'],
                 'test_acc': extracted_data['test_acc'],
                 'epoch': extracted_data['epoch'],
                 'files_generated': mftma_file}
        save_dict(d_master, mftma_file)
    else:
         print('file already exists, abort')
         pass

