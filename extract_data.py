from utils.extractor_utils import mftma_extractor
from utils.model_utils import NN, save_dict
from utils.extractor_utils import make_manifold_data
from utils import save_dir, data_dir, analyze_pool, train_pool
import pickle
import torch
import argparse
import os
import scipy.io as sio

parser = argparse.ArgumentParser(description='extract and save activations')
parser.add_argument('task_id', type=int,default=1)
parser.add_argument('model_id', type=str,default='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str,default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
args = parser.parse_args()

if __name__ == '__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    # STEP 1. get the variables
    task_id = args.task_id
    model_identifier = args.model_id
    analyze_identifier = args.analyze_id
    #
    # STEP 2. load model and analysis parameters
    #
    params = train_pool[model_identifier]()
    layer_names=params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    pickle_file = os.path.join(save_dir,model_identifier_for_saving, 'master_'+model_identifier_for_saving+'.pkl')
    #
    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    generated_files_txt = open(os.path.join(save_dir, model_identifier_for_saving, 'master_' + model_identifier_for_saving + '.csv'), 'r')
    weight_files = generated_files_txt.read().splitlines()
    weight_file = weight_files[args.task_id]
    weight_data = pickle.load(open(weight_file, 'rb'))

    # STEP 3. load the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(pickle_file, 'rb'))


    # STEP 4. create the dataset for testing
    data_loader = data['test_loader']
    num_hierarchy = len(data_loader.dataset.hierarchical_target)
    sample_idx = data_loader.sampler.indices
    hier_classes = [x.astype(int) for x in data_loader.dataset.hierarchical_target]
    hier_n_class = [int(max(x) + 1) for x in hier_classes]
    hier_dataset = []
    exm_per_class = analyze_params.exm_per_class
    for idx, x in enumerate(hier_classes):
        dat_hier = []
        [dat_hier.append((data_loader.dataset[i][0], x[i])) for i in sample_idx]
        hier_dataset.append(dat_hier)
    # TODO make_manifold_data should output labels too:x
    # TODO save train and test accuracy in the output of extraction

    hier_sample_mtmfa = [make_manifold_data(x, hier_n_class[idx],
                                            examples_per_class=analyze_params.exm_per_class,seed=0,
                                            randomize=analyze_params.randomize) for idx, x in enumerate(hier_dataset)]

    hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]
    # STEP 5. load the model and weights
    model = data['model_structure']
    model = model.to(device)

    model.load_state_dict(torch.load(weight_file)['state_dict'])
    weight_data=torch.load(weight_file)
    model = model.eval()
    # STEP 6. create projection dataset
    projection_data_ = {'projection_results': []}
    extract = mftma_extractor()
    activations_cell = [extract.extractor(model, x) for x in hier_sample_mtmfa]
    projection_cell = [extract.project(x, max_dim=analyze_params.n_project) for x in activations_cell]
    for x in projection_cell:
        assert(len(layer_names) == len(x))
    # reorder files based on the layer
    projection_file_list = []
    # TODO make the number of hierarchies a part of projection results.
    for name in layer_names:
        layer_proj_cell = [{name: x[name]} for x in projection_cell]
        # STEP 7. save the file
        projection_file = weight_file.replace(".pth", '')
        projection_file = projection_file + '_' + name + '_extracted.pkl'
        projection_file = projection_file.replace(os.path.join(save_dir, model_identifier_for_saving) + '/',
                                          os.path.join(save_dir, model_identifier_for_saving) + '/' + str(task_id).zfill(4) + '_')

        d_master = {'projection_results': layer_proj_cell,
                    'analyze_identifier': analyze_identifier,
                    'model_identifier': model_identifier,
                    'layer_name': name,
                    'files_generated': projection_file,
                    'train_acc':weight_data['train_acc'],
                    'test_acc':weight_data['test_acc'],
                    'epoch':weight_data['epoch'],
                    'batchidx':weight_data['batchidx']
                    }
        save_dict(d_master, projection_file)
        mat_file_name = projection_file.replace(".pkl", '.mat')
        sio.savemat(mat_file_name, {'activation': d_master})
        projection_file_list.append(projection_file+'\n')
    # write to text file
    if not os.path.exists(os.path.join(save_dir, model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_extracted.txt')):
        extracted_files_txt = open(os.path.join(save_dir, model_identifier_for_saving, 'master_' + model_identifier_for_saving + '_extracted.csv'), 'w')
        extracted_files_txt.writelines(projection_file_list)
    else:
        extracted_files_txt = open(os.path.join(save_dir, model_identifier_for_saving,'master_' + model_identifier_for_saving + '_extracted.csv'),'a')
        extracted_files_txt.writelines(projection_file_list)
    print('done!')
