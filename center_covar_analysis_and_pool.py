
from utils import save_dir,analyze_dir, train_pool,save_dict
from utils.analysis_utils import analyze_pool
import pickle
import os
import argparse
import numpy as np
import scipy.spatial.distance as dist
import fnmatch
import re
parser = argparse.ArgumentParser(description='run covar and save results')
parser.add_argument('model_id', type=str, default='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str, default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
parser.add_argument('train_id', type=str, default='epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06')
parser.add_argument('distance_metric', type=str, default='cosine')
parser.add_argument('overwrite',type=str,default='false')
args = parser.parse_args()

if __name__=='__main__':
    # get identifier,
    # get model
    # get sub_data
    # create hierarchical version: of the data
    model_identifier = args.model_id
    analyze_identifier = args.analyze_id
    train_dir_identifier = args.train_id
    distance_metric = args.distance_metric
    overwrite = args.overwrite
    # for testing the code
    #save_dir='/Users/eghbalhosseini/Desktop'
    #model_identifier = "NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed"
    #analyze_identifier = "mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5"
    #train_dir_identifier = "epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06"
    #distance_metric='cosine'

    #overwrite = "true"

    #
    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    # find pth files
    extracted_files = []
    for file in os.listdir(os.path.join(save_dir, analyze_identifier_for_saving, model_identifier_for_saving,
                                        train_dir_identifier)):
        if fnmatch.fnmatch(file, '*_extracted_v2.pkl'):
            extracted_files.append(os.path.join(save_dir, analyze_identifier_for_saving, model_identifier_for_saving,
                                            train_dir_identifier, file))
    s = [re.findall('/\d+', x) for x in extracted_files]
    s = [item for sublist in s for item in sublist]
    file_id = [int(x.split('/')[1]) for x in s]
    sorted_files = [extracted_files[x] for x in np.argsort(file_id)]

    covar_pooled = dict()
    for idx, layer in enumerate(layer_names):
        s = np.asarray([int(not not re.findall(layer, x)) for x in sorted_files])
        layer_files = [sorted_files[int(x)] for x in np.argwhere(s)]
        x_idx = np.argwhere(s)
        layer_results = []
        for id_file, file in enumerate(layer_files):
            extracted_data = pickle.load(open(file, 'rb'))
            assert (extracted_data['layer_name'] == layer)
            s = re.findall('-batchidx=\d+', file)
            batchidx = [int(x.split('=')[1]) for x in s][0]
            assert (extracted_data['batchidx'] == batchidx)
            s = re.findall('-epoch=\d+', file)
            epochidx = [int(x.split('=')[1]) for x in s][0]
            assert (extracted_data['epoch'] == epochidx)
            projection_data_ = extracted_data['projection_results']
            covar_results=[]
            print(f"analyzing  {id_file}: {file}")
            for hier_id, activ_hier in enumerate(projection_data_):

                data_ = {'layer': [], 'n_hier_class': [], 'hierarchy': hier_id}
                center_cov_all = []
                for k, X, in activ_hier.items():
                    data_['layer'] = k
                    data_['n_hier_class'] = len(X)

                    centers = [np.mean(X[i], axis=1) for i in range(len(X))]
                    centers = np.stack(centers, axis=1)  # Centers is of shape (N, m) for m manifolds
                    center_cov = dist.squareform(dist.pdist(np.transpose(centers),metric=distance_metric))
                    # center_cov=np.cov(np.transpose(centers))
                    center_cov_all.append(center_cov)
                data_['center_cov'] = center_cov_all
                covar_results.append(data_)
            layer_results.append(dict(center_covar=covar_results, epoch=epochidx, batch=batchidx,
                                      seq=id_file, train_acc=extracted_data['train_acc'], test_acc=extracted_data['test_acc'], file=file))


        covar_pooled[layer] = layer_results
    pool_file = os.path.join(analyze_dir, analyze_identifier_for_saving, model_identifier_for_saving,
                             train_dir_identifier, f'{model_identifier_for_saving}_{distance_metric}_center_covar_pooled_v2.pkl')
    d_master = {'analyze_identifier': analyze_identifier,
                'model_identifier': model_identifier,
                'train_identifier': train_dir_identifier,
                'distance_metric': distance_metric,
                'center_covar_results': covar_pooled,
                'file_generated': pool_file}
    save_dict(d_master, pool_file)
    print('saved ' + pool_file)
    print('done')

