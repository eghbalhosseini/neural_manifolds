import numpy as np
import sys
import os
import argparse
from utils.model_utils import save_dict
import pickle
import fnmatch
from tqdm import tqdm
from utils import save_dir, analyze_dir, result_dir,train_pool
from utils.analysis_utils import analyze_pool
from tqdm import tqdm
import re
import itertools

if __name__ == '__main__':
    model_identifier = 'NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed'
    train_identifier = 'epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06'
    analyze_identifier = 'mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'

    training_files = []
    for file in os.listdir(os.path.join(save_dir, model_identifier, train_identifier)):
        if fnmatch.fnmatch(file, '*.pth'):
            training_files.append(os.path.join(save_dir, model_identifier, train_identifier, file))

    grad_pkl_files = []
    for file in os.listdir(os.path.join(save_dir, analyze_identifier, model_identifier, train_identifier)):
        if fnmatch.fnmatch(file, '*gradient_data.pkl'):
            grad_pkl_files.append(os.path.join(save_dir, analyze_identifier, model_identifier, train_identifier, file))
    s = [re.findall('/\d+', x) for x in grad_pkl_files]
    s = [item for sublist in s for item in sublist]
    dummy_id = [(x.split('/')) for x in s]
    file_id = [int(x.split('/')[1]) for x in s]
    sorted_files = [grad_pkl_files[x] for x in np.argsort(file_id)]
    grad_pkl_files = sorted_files

    params = train_pool[model_identifier]()
    params.load_dataset()
    layer_names = params.get_layer_names()[1:]
    transfo_mat = params.dataset.transformation_mats
    analyze_params = analyze_pool[analyze_identifier]()
    tiled_transfo_mat = [np.tile(x, (1, analyze_params.exm_per_class)).reshape(-1, x.shape[1]) for x in transfo_mat]

    # do analysis for layers
    layer_gradient_dict=dict()
    for layer in layer_names:
        print(f"analyzing {layer} \n")
        layer_branch_data=dict()
        all_grad_data = []
        # method 1 , only analyze leaf node.
        for idx, file in tqdm(enumerate(grad_pkl_files)):
            g = pickle.load(open(file, 'rb'))
            e = np.asarray(g['results'][0][layer]).squeeze()
            all_grad_data.append(e)
        e_f_list = [np.reshape(x, [-1, x.shape[2]]) for x in all_grad_data]

        for hier_id,transfo in enumerate(tiled_transfo_mat):
            print(f"analyzing hierarchy {hier_id} \n")
            # get grad data:
            # create combination pairs
            example_per_class=int(np.unique(transfo.sum(axis=0)))
            a = list(range(example_per_class))
            combs = list(itertools.combinations_with_replacement(a, r=2))
            combs_1 = list(itertools.combinations(a, r=2))
            combs_1 = [(x[1], x[0]) for x in combs_1]
            all_combs = [combs, combs_1]
            all_combs = [item for sublist in all_combs for item in sublist]
            if len(all_combs)>example_per_class**2:
                select_combs = [all_combs[x] for x in np.random.choice(np.arange(len(all_combs)), size=2500)]
            else:
                select_combs=all_combs
            # use the combination to compute the differences in vectors
            all_branch_diffs = []
            for _, time_point in tqdm(enumerate(e_f_list)):
                branch_diffs = []
                for branch in transfo.transpose():  # for each single "over" branch, iterate through the categories
                    P12 = time_point[np.where(branch == 1), :].squeeze()  # extract each category
                    p12_diff = [np.diff(P12[x, :], axis=0) for x in
                                select_combs]  # find all pairwise differences between all samples in cat 1 and cat 2 (the leaf branches)
                    branch_diffs.append(
                        np.mean(np.stack(p12_diff).squeeze(), axis=0))  # take mean over all the pairwise diffs
                all_branch_diffs.append(branch_diffs)
            # compute the norms for differences
            branch_norm = [[np.linalg.norm(x) for x in branch_diffs] for branch_diffs in all_branch_diffs]
            branch_norm_mat = np.stack(branch_norm)
            layer_branch_data[f"hier_{hier_id}"]=branch_norm_mat
        layer_gradient_dict[layer]=layer_branch_data
        # save layer data independently
        layer_gradient_file = os.path.join(save_dir, analyze_identifier, model_identifier, train_identifier,
                                     f'{model_identifier}_{layer}_gradient_pooled.pkl')
        d_layer = {'analyze_identifier': analyze_identifier,
                    'model_identifier': model_identifier,
                    'train_identifier': train_identifier,
                    'layer':layer,
                    'gradient_results': layer_branch_data}
        save_dict(d_layer, layer_gradient_file)
        print('saved ' + layer_gradient_file)

    # save the results
    gradient_file = os.path.join(save_dir, analyze_identifier, model_identifier, train_identifier, f'{model_identifier}_gradient_pooled.pkl')
    d_master = {'analyze_identifier': analyze_identifier,
                'model_identifier': model_identifier,
                'train_identifier':train_identifier,
                'gradient_results':layer_gradient_dict}
    save_dict(d_master, gradient_file)
    print('saved ' + gradient_file)
    print('done')






