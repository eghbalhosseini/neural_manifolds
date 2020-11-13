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

# parser = argparse.ArgumentParser(description='extract and save activations')
# parser.add_argument('file_id', type=str,default=' ')
# parser.add_argument('task_id', type=int,default=0)
# parser.add_argument('model_id', type=str,default='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed')
# parser.add_argument('analyze_id', type=str,default='mftma-exm_per_class=100-proj=False-rand=False-kappa=0-n_t=300-n_rep=1')
# parser.add_argument('overwrite',type=str,default='false')
#
# args = parser.parse_args()

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

if __name__ == '__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    # STEP 1. get the variables
    # file_id = args.file_id
    # task_id = args.task_id
    # model_identifier = args.model_id
    # analyze_identifier = args.analyze_id
    # overwrite = args.overwrite



    file_id = '/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed/NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed-epoch=1-batchidx=570.pth'
    task_id = 1
    model_identifier = 'NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed'
    analyze_identifier = 'mftma-exm_per_class=100-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
    overwrite = 'true'

    #
    # STEP 2. load model and analysis parameters
    #
    params = train_pool[model_identifier]()
    layer_names=params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    pickle_file = os.path.join(save_dir,model_identifier_for_saving, 'master_'+model_identifier+'.pkl')
    data = pickle.load(open(pickle_file, 'rb'))
    #
    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    #
    # check if path exists
    if not os.path.exists(os.path.join(save_dir,model_identifier_for_saving)):
        os.mkdir(os.path.join(save_dir,model_identifier_for_saving))

    file_parts = file_id.split('/')
    layer_extraction=[True for k in layer_names]
    projection_done_file_list=[]
    print(f"overwriting {overwrite}")
    if 'false' in overwrite:
        for idx, name in enumerate(layer_names):
            projection_file = os.path.join(save_dir, model_identifier_for_saving, file_parts[-1])
            projection_file = projection_file.replace(".pth", '')
            projection_file = projection_file + '_' + name + '_extracted.pkl'
            projection_file = projection_file.replace(os.path.join(save_dir, model_identifier) + '/',
                                                      os.path.join(save_dir, model_identifier) + '/' + str(
                                                          task_id).zfill(4) + '_')

            print(f"looking for {projection_file}")
            if os.path.exists(projection_file):
                # file exist already , write it and set layer_analysis to false
                print(f"{projection_file} already exists")
                if not os.path.exists(os.path.join(save_dir, model_identifier, 'master_' + model_identifier + '_extracted.csv')):
                    extracted_files_txt = open(os.path.join(save_dir, model_identifier, 'master_' + model_identifier + '_extracted.csv'), 'w',os.O_NONBLOCK)
                    extracted_files_txt.writelines(projection_file+'\n')
                    extracted_files_txt.flush()
                else:
                    extracted_files_txt = open(os.path.join(save_dir, model_identifier, 'master_' + model_identifier + '_extracted.csv'), 'a+',os.O_NONBLOCK)
                    already_written = extracted_files_txt.readlines()
                    print(f" {len(already_written)} are already written")
                    temp = intersection(already_written, [projection_file])
                    for k in temp:
                        [projection_file].remove(k)
                    projection_done_list = [x + '\n' for x in [projection_file]]
                    print(f"adding {len(projection_done_list)} remaining files to extracted.csv")
                    extracted_files_txt.writelines(projection_done_list)
                    extracted_files_txt.flush()

                layer_extraction[idx]=False
                projection_done_file_list.append(projection_file)
                # check if file exist in master
        # write the files if they dont exist in extraction
        projection_done_file_list=[x+'\n' for x in projection_done_file_list]
        if not os.path.exists(os.path.join(save_dir, model_identifier, 'master_' + model_identifier + '_extracted.csv')):
            extracted_files_txt = open(os.path.join(save_dir, model_identifier, 'master_' + model_identifier + '_extracted.csv'), 'w',os.O_NONBLOCK)

            extracted_files_txt.writelines(projection_done_file_list)
            extracted_files_txt.flush()
            print(f"adding {len(projection_done_file_list)} new files to extracted.csv")
        else:
            extracted_files_txt = open(os.path.join(save_dir, model_identifier, 'master_' + model_identifier + '_extracted.csv'), 'a+',os.O_NONBLOCK)
            already_written=extracted_files_txt.readlines()
            print(f" {len(already_written)} are already written")
            temp=intersection(already_written,projection_done_file_list)
            for k in temp:
                projection_done_file_list.remove(k)
            print(f"adding {len(projection_done_file_list)} remaining files to extracted.csv")
            extracted_files_txt.writelines(projection_done_file_list)
            extracted_files_txt.flush()
    # based on layers decide to run the files
    print(layer_extraction)
    do_extraction=False
    if True in layer_extraction:
        do_extraction=True
        print('extracting data')
    else:
        do_extraction=False
        print('not extracting data')

    if do_extraction:
        # weight_data = pickle.load(open(file_id, 'rb'))
        weight_data = torch.load(open(file_id, 'rb'))
        # STEP 3. load the dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # TODO include hierarchical target label
        # hier_sample_mtmfa = [make_manifold_data(x, hier_n_class[idx],
        #                                     examples_per_class=analyze_params.exm_per_class,seed=0,
        #                                     randomize=analyze_params.randomize) for idx, x in enumerate(hier_dataset)]
        #
        # hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]
        # STEP 5. load the model and weights
        model = data['model_structure']
        model = model.to(device)

        model.load_state_dict(torch.load(file_id)['state_dict'])
        weight_data=torch.load(file_id)
        model = model.eval()

        # STEP 5.5. extract hierarchical distances
        data_mod=dict()
        for i, x in enumerate(data['distance_pair_index'].keys()):
            data_mod[len(data['distance_pair_index'])-1-i]=data['distance_pair_index'][x]

        full_dataset = data['test_loader'].dataset.data
        distance_pairs_in_data = dict() # Fetch data from the chosen distance pair indices

        for hier_idx, value in data_mod.items():
            index_pairs = value['index_pairs'] # the chosen pairs
            index_pairs2 = [np.transpose(np.stack(x)).tolist() for x in index_pairs] # separate out [a1,b1] pairs to two arrays of [a1,...,an] and [b1,...,bn]
            distance_pairs_in_data[hier_idx] = [[torch.from_numpy(full_dataset[x,:]) for x in y] for y in index_pairs2] # get the data representation
        extract = mftma_extractor()
        distance_pair_activations=[[[extract.extractor(model,x.to(device)) for x in y] for y in v] for k, v in distance_pairs_in_data.items()]
        # STEP 6. create projection dataset
        projection_data_ = {'projection_results': []}
        d_master = {'results': distance_pair_activations,
                        'distance_pairs_in_data': distance_pairs_in_data
                        }
        projection_file = os.path.join(save_dir, model_identifier_for_saving, file_parts[-1])
        projection_file = projection_file.replace(".pth", '')
        projection_file = projection_file + '_distance_data.pkl'

        save_dict(d_master, projection_file)
        print('done!')
