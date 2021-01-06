from utils.extractor_utils import mftma_extractor
from utils.model_utils import NN, save_dict
from utils.extractor_utils import make_manifold_data
from utils import save_dir, data_dir, train_pool
from utils.analysis_utils import analyze_pool
import pickle
import torch
import argparse
import os
import scipy.io as sio
import numpy as np
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='extract and save activations')
parser.add_argument('file_id', type=str,default=' ')
parser.add_argument('task_id', type=int,default=0)
parser.add_argument('model_id', type=str,default='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed')
parser.add_argument('analyze_id', type=str,default='mftma-exm_per_class=100-proj=False-rand=False-kappa=0-n_t=300-n_rep=1')
parser.add_argument('overwrite',type=str,default='false')

args = parser.parse_args()

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

if __name__ == '__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    # STEP 1. get the variables
    file_id = args.file_id
    task_id = args.task_id
    model_identifier = args.model_id
    analyze_identifier = args.analyze_id
    overwrite = args.overwrite
    #
    file_parts = file_id.split('/')
    train_dir=file_parts[-2]
    data_dir=os.path.join(save_dir,analyze_identifier,model_identifier,train_dir)
    pth_dir=os.path.join(save_dir,model_identifier,train_dir)
    #data_dir='/'.join(file_parts[:-1])
    # STEP 2. load model and analysis parameters
    #
    params = train_pool[model_identifier]()
    layer_names=params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    #pickle_file = os.path.join(pth_dir, 'master_'+model_identifier+'.pkl')
    #data = pickle.load(open(pickle_file, 'rb'))

    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    # check if path exists
    #if not os.path.exists(os.path.join(save_dir,model_identifier_for_saving)):
    #    os.mkdir(os.path.join(save_dir,model_identifier_for_saving))


    layer_extraction=[True for k in layer_names]
    distance_extraction=True
    projection_done_file_list=[]
    do_extraction=True
    print(f"overwriting {overwrite}")
    # in case of not overwriting figure out what needs to be analyzed
    if 'false' in overwrite:
    ## extraction of data for knn and projection

        for idx, name in enumerate(layer_names):
            projection_file = os.path.join(data_dir, file_parts[-1])
            projection_file = projection_file.replace(".pth", '')
            projection_file = projection_file + '_' + name + '_extracted.pkl'
            projection_file = projection_file.replace(os.path.join(data_dir) + '/',
                                                      os.path.join(data_dir) + '/' + str(
                                                        task_id).zfill(4) + '_')
            print(f"looking for {projection_file}")
            if os.path.exists(projection_file):
                # file exist already , write it and set layer_analysis to false
                print(f"{projection_file} already exists")
                layer_extraction[idx]=False
                projection_done_file_list.append(projection_file)
                # check if file exist in master
        # write the files if they dont exist in extraction
        projection_done_file_list=[x+'\n' for x in projection_done_file_list]


        ## check if distance data exsits:
        distance_file = os.path.join(data_dir, file_parts[-1])
        distance_file = distance_file.replace(".pth", '')
        distance_file = distance_file + '_distance_data.pkl'
        distance_file = distance_file.replace(os.path.join(data_dir) + '/',
                                          os.path.join(data_dir) + '/' + str(
                                              task_id).zfill(4) + '_')
        if os.path.exists(distance_file):
            distance_extraction=False

    do_extraction_mftma_knn=False
    if True in layer_extraction:
        do_extraction_mftma_knn=True
        print('extracting data for mftma and knn')
    else:
        do_extraction_mftma_knn=False
        print('not extracting data for mftma and knn')
    do_extraction= do_extraction_mftma_knn or distance_extraction

    if do_extraction:
        weight_data = torch.load(open(file_id, 'rb'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = data['model_structure']
        model = model.to(device)
        model.load_state_dict(torch.load(file_id)['state_dict'])
        weight_data = torch.load(file_id)
        model = model.eval()
        extract = mftma_extractor()

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
        # doing extraction for mftma and knn
        if do_extraction_mftma_knn:
        # this is for projection part
            hier_sample_mtmfa = [make_manifold_data(x, hier_n_class[idx],
                                            examples_per_class=analyze_params.exm_per_class,seed=0,
                                            randomize=analyze_params.randomize) for idx, x in enumerate(hier_dataset)]

            hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]
            projection_data_ = {'projection_results': []}
            activations_cell = [extract.extractor(model, x) for x in hier_sample_mtmfa]
            projection_cell = [extract.project(x, max_dim=analyze_params.n_project) for x in activations_cell]
            for x in projection_cell:
                assert(len(layer_names) == len(x))
        # reorder files based on the layer
            projection_file_list = []
            for name in layer_names:
                layer_proj_cell = [{name: x[name]} for x in projection_cell]
            # STEP 7. save the file
                projection_file=os.path.join(data_dir,file_parts[-1])
                projection_file = projection_file.replace(".pth", '')
                projection_file = projection_file + '_' + name + '_extracted.pkl'

                projection_file = projection_file.replace(os.path.join(data_dir) + '/',
                                          os.path.join(data_dir) + '/' + str(task_id).zfill(4) + '_')
                print(projection_file)
                d_master = {'projection_results': layer_proj_cell,
                        'analyze_identifier': analyze_identifier,
                        'model_identifier': model_identifier,
                        'layer_name': name,
                        'files_generated': projection_file,
                        'train_acc': weight_data['train_acc'],
                        'test_acc': weight_data['test_acc'],
                        'epoch': weight_data['epoch'],
                        'batchidx': weight_data['batchidx']
                        }
                save_dict(d_master, projection_file)
                mat_file_name = projection_file.replace(".pkl", '.mat')
                sio.savemat(mat_file_name, {'activation': d_master})

        # doing extraction for distance:
        if distance_extraction:
            data_mod = dict()
            for i, x in enumerate(data['distance_pair_index'].keys()):
                data_mod[len(data['distance_pair_index']) - 1 - i] = data['distance_pair_index'][x]
            full_dataset = data['test_loader'].dataset.data
            distance_pairs_in_data = dict()  # Fetch data from the chosen distance pair indices
            for hier_idx, value in data_mod.items():
                index_pairs = value['index_pairs']  # the chosen pairs
                index_pairs2 = [np.transpose(np.stack(x)).tolist() for x in index_pairs]  # separate out [a1,b1] pairs to two arrays of [a1,...,an] and [b1,...,bn]
                distance_pairs_in_data[hier_idx] = [[torch.from_numpy(full_dataset[x, :]).float() for x in y] for y in
                                                    index_pairs2]  # get the data representation
            distance_pair_activations = [[[extract.extractor(model, [x.to(device)]) for x in y] for y in v] for k, v in
                                         distance_pairs_in_data.items()]

            data_reshaped = dict()
            for name in layer_names:
                hierarchy_dict = dict()
                for idx, hierarchy in enumerate(distance_pair_activations):
                    category_list = []
                    for category_pairs in hierarchy:
                        category_list.append([x[name][0] for x in category_pairs])

                    hierarchy_dict[idx] = dict(pairs=category_list,
                                               distance=[np.diag(cdist(x[0], x[1])) for x in category_list])
                data_reshaped[name] = hierarchy_dict

            d_distance = {'results': distance_pair_activations,
                        'distance_data': data_reshaped,
                        'distance_pairs_in_data': distance_pairs_in_data,
                        'train_acc': weight_data['train_acc'],
                        'test_acc': weight_data['test_acc'],
                        'epoch': weight_data['epoch'],
                        'batchidx': weight_data['batchidx']}
            distance_file = os.path.join(data_dir, file_parts[-1])
            distance_file = distance_file.replace(".pth", '')
            distance_file = distance_file + '_distance_data.pkl'
            distance_file = distance_file.replace(os.path.join(data_dir) + '/',
                                                  os.path.join(data_dir) + '/' + str(task_id).zfill(
                                                      4) + '_')
            save_dict(d_distance, distance_file)
