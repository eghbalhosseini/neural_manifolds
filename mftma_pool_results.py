from utils.model_utils import save_dict
from utils import save_dir, data_dir, analyze_pool, train_pool
import pickle
import os
import glob
import re
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='run mftma and save results')
parser.add_argument('model_id', type=str,default='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]')
parser.add_argument('analyze_id', type=str,default='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]')
args = parser.parse_args()


if __name__ == '__main__':

    model_identifier = args.model_id
    analyze_identifier = args.analyze_id

    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    analyze_params = analyze_pool[analyze_identifier]()
    analyze_identifier_for_saving = analyze_params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    # find layers
    extracted_files = glob.glob(os.path.join(save_dir,model_identifier_for_saving, model_identifier_for_saving + '*_mftma_analysis*'))

    # find epochs
    s=[re.findall('-batchidx=\d+', x)[0] for x in extracted_files]
    batchidx = [int(x.split('=')[1]) for x in s]
    # find batch_id
    epoch_factor=10**np.ceil(np.log10(np.max(batchidx)))
    s = [re.findall('-epoch=\d+', x)[0] for x in extracted_files]
    epochidx = [int(x.split('=')[1]) for x in s]
    epoch_batch=[batchidx[idx]+x*epoch_factor for idx, x in enumerate(epochidx)]
    sort_id = np.argsort(epoch_batch)
    sorted_files = [extracted_files[int(x)] for x in list(sort_id)]
    # do layerwise saving
    for idx, layer in enumerate(layer_names):
        s = np.asarray([int(not not re.findall(layer, x)) for x in sorted_files])
        x_idx=np.argwhere(s)
        layer_files=[sorted_files[int(x)] for x in list(x_idx)]
        s = [re.findall('-epoch=\d+', x)[0] for x in layer_files]
        epochidx = [int(x.split('=')[1]) for x in s]
        s = [re.findall('-batchidx=\d+', x)[0] for x in layer_files]
        batchidx = [int(x.split('=')[1]) for x in s]
        epoch_factor = 10 ** np.ceil(np.log10(np.max(batchidx)))
        epoch_batch = [batchidx[idx] + x * epoch_factor for idx, x in enumerate(epochidx)]
        assert((np.asarray(epoch_batch)==np.sort(epoch_batch)).all())
        layer_results=[]
        for id_file, file in enumerate(layer_files):
            data_=pickle.load(open(file, 'rb'))
            assert(data_['layer_name']==layer)
            layer_results.append(dict(mftma=data_['mftma_results'], epoch=epochidx[id_file], batch=batchidx[id_file],
                 seq=id_file))

        pool_file = os.path.join(save_dir,model_identifier_for_saving, f'{model_identifier_for_saving}_{layer}_mftma_pooled.pkl')
        d_master = {'layer_results': layer_results,
                    'analyze_identifier': analyze_identifier,
                    'model_identifier': model_identifier,
                    'layer_name': layer,
                    'files_generated': pool_file}
        save_dict(d_master, pool_file)
        print('saved '+pool_file)
    print('done')







