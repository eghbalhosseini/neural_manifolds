from utils.analysis_utils import run_mftma
from utils.model_utils import save_dict
from utils import save_dir, data_dir,analyze_dir, train_pool
from utils.analysis_utils import analyze_pool
import torch
import pickle
import os
import argparse
import re
parser = argparse.ArgumentParser(description='run mftma and save results')
#parser.add_argument('task_id', type=int, default=1)
parser.add_argument('file_id', type=str, default='')
parser.add_argument('model_id', type=str, default='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str, default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
parser.add_argument('overwrite',type=str,default='True')
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

    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    # check if path exists
    if not os.path.exists(os.path.join(analyze_dir,analyze_identifier)):
        os.mkdir(os.path.join(analyze_dir,analyze_identifier))

    file_parts=file_id.split('/')
    extracted_data = pickle.load(open(file_id, 'rb'))
    projection_data_ = extracted_data['projection_results']
    # create outputfile
    mftma_file=os.path.join(analyze_dir,analyze_identifier,file_parts[-1])
    mftma_file = mftma_file.replace("_extracted.pkl", '_mftma_analysis.pkl')
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check whether file exist
    do_analysis=True
    if 'False' in overwrite:
         if os.path.exists(mftma_file):
            do_analysis=False
         else:
             do_analysis=True
    #
    if do_analysis:
    # run mftma
        mftma_results = run_mftma(projection_data_, kappa=analyze_params.kappa, n_t=analyze_params.n_t, n_reps=analyze_params.n_rep)
        # save results:
        mftma_file=os.path.join(analyze_dir,analyze_identifier,file_parts[-1])
        mftma_file = mftma_file.replace("_extracted.pkl", '_mftma_analysis.pkl')
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
    #     if not os.path.exists(os.path.join(save_dir, model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_mftma_analysis.csv')):
    #         mftma_analysis_files_txt = open(os.path.join(save_dir,model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_mftma_analysis.csv'), 'w')
    #         mftma_analysis_files_txt.write(mftma_file+'\n')
    #     else:
    #         mftma_analysis_files_txt = open(os.path.join(save_dir, model_identifier_for_saving,'master_' + model_identifier_for_saving + '_mftma_analysis.csv'), 'a')
    #         mftma_analysis_files_txt.write(mftma_file + '\n')
    #     print('done')
    # else:
    #     print('file already exists, abort')
    #     pass

