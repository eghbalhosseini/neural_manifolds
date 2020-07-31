from neural_manifolds_utils.extractor_utils import mftma_extractor
from neural_manifolds_utils.neural_manifold_utils import NN, save_dict
from neural_manifolds_utils.extractor_utils import make_manifold_data
from neural_manifolds_utils import save_dir, data_dir, analyze_pool, train_pool
import pickle
import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='extract and save activations')
parser.add_argument('save_dir', type=str,
                    default='/Users/eghbalhosseini/Desktop/network_training_on_synthetic/')

parser.add_argument('pickle_file', type=str,
                    default="master_NN-partition_nclass=50_nobj=50000_beta=0.01_sigma=1.5_nfeat=3072-train_test-test_performance.pkl")
parser.add_argument('weight_file', type=str,
                    default='NN-partition_nclass=50_nobj=50000_beta=0.01_sigma=1.5_nfeat=3072-train_test-test_performance-epoch=1-batchidx=75.pth')



parser.add_argument('example_per_class', type=int, default=20)


args = parser.parse_args()

if __name__ == '__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    #
    #STEP 1. get the variables
    save_identifier = args.save_dir
    data_identifier = args.pickle_file
    weight_identifier = args.weight_file
    exm_per_class = args.example_per_class
    data_file = os.path.join(save_identifier, data_identifier)
    weight_file = os.path.join(save_identifier, weight_identifier)

    #STEP 2. load the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(data_file, 'rb'))
    weight_data = pickle.load(open(weight_file, 'rb'))

    # STEP 3. create the dataset for testing
    data_loader = data['test_loader']
    num_hierarchy = len(data_loader.dataset.hierarchical_target)
    sample_idx = data_loader.sampler.indices
    hier_classes = [x.astype(int) for x in data_loader.dataset.hierarchical_target]
    hier_n_class = [int(max(x) + 1) for x in hier_classes]
    hier_dataset = []
    for idx, x in enumerate(hier_classes):
        dat_hier = []
        [dat_hier.append((data_loader.dataset[i][0], x[i])) for i in sample_idx]
        hier_dataset.append(dat_hier)
    hier_sample_mtmfa = [make_manifold_data(x, hier_n_class[idx], exm_per_class, seed=0, randomize=False) for idx, x in enumerate(hier_dataset)]
    hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]

    # STEP 4. load the model and weights
    model=data['model_structure']
    model = model.to(device)
    model.load_state_dict(torch.load(weight_file)['state_dict'])

    # STEP 5. create projection dataset
    projection_data_ = {'projection_results': []}
    extract = mftma_extractor()
    activations_cell = [extract.extractor(model, x) for x in hier_sample_mtmfa]
    projection_cell = [extract.project(x, max_dim=200) for x in activations_cell]

    # STEP 6. save the file
    projection_file = weight_file.replace(".pth", '')
    projection_file = projection_file + '_extracted.pkl'
    projection_data_['projection_results'] = projection_cell
    save_dict(projection_data_, projection_file)
    # projection step :
