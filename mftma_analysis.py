from utils.analysis_utils import run_mftma
from utils.model_utils import save_dict
from utils import save_dir, data_dir, analyze_pool, train_pool
import torch
import pickle
import os
import argparse
parser = argparse.ArgumentParser(description='run mftma and save results')
parser.add_argument('task_id', type=int, default=1)
parser.add_argument('model_id', type=str, default='[NN]-[partition/nclass=;50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str, default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
args = parser.parse_args()

if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version: of the data
    #
    task_id = args.task_id
    model_identifier = args.model_id
    analyze_identifier = args.analyze_id

    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    generated_files_txt = open(os.path.join(save_dir,model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_extracted.csv'), 'r')
    extracted_files = generated_files_txt.read().splitlines()
    extracted_file = extracted_files[task_id]
    print(extracted_file)
    extracted_data = pickle.load(open(extracted_file, 'rb'))
    projection_data_ = extracted_data['projection_results']
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # run mftma
    mftma_results = run_mftma(projection_data_, kappa=analyze_params.kappa, n_t=analyze_params.n_t, n_reps=analyze_params.n_rep)
    # save results:
    mftma_file = extracted_file.replace("_extracted.pkl", '_mftma_analysis.pkl')

    d_master = {'mftma_results': mftma_results,
                'analyze_identifier': analyze_identifier,
                'model_identifier': model_identifier,
                'layer_name': extracted_data['layer_name'],
                'train_acc': extracted_data['train_acc'],
                'test_acc': extracted_data['test_acc'],
                'epoch': extracted_data['epoch'],
                'files_generated': mftma_file}
    save_dict(d_master, mftma_file)

    if not os.path.exists(os.path.join(save_dir, model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_mftma_analysis.csv')):
        mftma_analysis_files_txt = open(os.path.join(save_dir,model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_mftma_analysis.csv'), 'w')
        mftma_analysis_files_txt.write(mftma_file+'\n')
    else:
        mftma_analysis_files_txt = open(os.path.join(save_dir, model_identifier_for_saving,'master_' + model_identifier_for_saving + '_mftma_analysis.csv'), 'a')
        mftma_analysis_files_txt.write(mftma_file + '\n')
    print('done')
