from neural_manifolds_utils.extractor_utils import mftma_extractor
from neural_manifolds_utils.neural_manifold_utils import create_manifold_data
from neural_manifolds_utils import save_dir, data_dir, analyze_pool

import pickle
import copy
if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    #
    data_file='/Users/eghbalhosseini/Desktop/test_loader.pkl'
    with open(data_file) as x:
        data = pickle.load(open(data_file, 'rb'))
    data_loader=res
    num_hierarchy = len(data_loader.dataset.hierarchical_target)
    sample_idx=data_loader.sampler.indices
    hier_classes = [x.astype(int) for x in data_loader.dataset.hierarchical_target]
    hier_n_class = [int(max(x) + 1) for x in hier_classes]
    hier_dataset = []
    exm_per_class=100
    for idx, x in enumerate(hier_classes):
        dat_mfmta = []
        dat_mtmfa = copy.deepcopy(data_loader.dataset)
        dat_mtmfa.targets = hier_classes[idx]
        hier_dataset.append(copy.deepcopy(dat_mtmfa))

    hier_sample_mtmfa = [create_manifold_data(x, hier_n_class[idx], exm_per_class, seed=0) for idx, x in
                         enumerate(hier_dataset)]
    hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]

    pass
