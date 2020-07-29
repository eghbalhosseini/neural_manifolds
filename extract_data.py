from neural_manifolds_utils.extractor_utils import mftma_extractor
from neural_manifolds_utils.extractor_utils import create_manifold_data
from neural_manifolds_utils import save_dir, data_dir, analyze_pool
import pickle
import copy
import torch

if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version of the data
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file='/Users/eghbalhosseini/Desktop/test_loader.pkl'
    with open(data_file) as x:
        data = pickle.load(open(data_file, 'rb'))
        res = dict(zip(['1'], data))
    data_loader=res['1']
    num_hierarchy = len(data_loader.dataset.hierarchical_target)
    sample_idx=data_loader.sampler.indices
    hier_classes = [x.astype(int) for x in data_loader.dataset.hierarchical_target]
    hier_n_class = [int(max(x) + 1) for x in hier_classes]
    hier_dataset = []
    exm_per_class=20
    # create samples of the data
    for idx, x in enumerate(hier_classes):
        dat_hier=[]
        [dat_hier.append((data_loader.dataset[i][0],x[i])) for i in sample_idx]
        hier_dataset.append(dat_hier)

    hier_sample_mtmfa = [create_manifold_data(x, hier_n_class[idx], exm_per_class, seed=0,randomize=False) for idx, x in
                         enumerate(hier_dataset)]
    hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]

    pass
