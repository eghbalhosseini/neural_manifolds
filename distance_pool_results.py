from utils.model_utils import save_dict
from utils import save_dir, data_dir, train_pool, analyze_dir
from utils.analysis_utils import analyze_pool
import pickle
import os
import re
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='pool results of distance analysis')
parser.add_argument('model_id', type=str, default='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed')
parser.add_argument('analyze_id', type=str, default='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1')
args = parser.parse_args()

save_dir='/Users/eghbalhosseini/Desktop/'
if __name__ == '__main__':
    #model_identifier = args.model_id
    #analyze_identifier = args.analyze_id
    #params = train_pool[model_identifier]()
    model_identifier='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed'
    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    # find layers
    extraction_files_csv = open(os.path.join(save_dir, model_identifier_for_saving, 'master_' + model_identifier_for_saving + '.csv'),'r')
    analysis_files = extraction_files_csv.read().splitlines()
    analysis_files=[x.replace('/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/',save_dir) for x in analysis_files]
    s = [re.findall('/\d+', x)[0] for x in analysis_files]
    file_id = [int(x.split('/')[1]) for x in s]
    sorted_files = [analysis_files[x] for x in np.argsort(file_id)]
    # do layerwise saving
    mftma_pooled = dict()
    for idx, layer in enumerate(layer_names):
        s = np.asarray([int(not not re.findall(layer, x)) for x in sorted_files])
        layer_files=[sorted_files[int(x)] for x in np.argwhere(s)]
        x_idx=np.argwhere(s)
        layer_results=[]
        for id_file, file in enumerate(layer_files):
            data_=pickle.load(open(file, 'rb'))
            assert(data_['layer_name']==layer)
            s =re.findall('-batchidx=\d+', file)
            batchidx = [int(x.split('=')[1]) for x in s][0]
            s = re.findall('-epoch=\d+', file)
            epochidx = [int(x.split('=')[1]) for x in s][0]

            layer_results.append(dict(mftma=data_['mftma_results'], epoch=epochidx, batch=batchidx,
                 seq=id_file,train_acc=data_['train_acc'],test_acc=data_['test_acc'] , file=file))
        mftma_pooled[layer]=layer_results
    pool_file = os.path.join(analyze_dir,analyze_identifier_for_saving,model_identifier_for_saving, f'{model_identifier_for_saving}_mftma_pooled.pkl')
    d_master = {'analyze_identifier': analyze_identifier,
                'model_identifier': model_identifier,
                'mftma_results': mftma_pooled,
                'file_generated': pool_file}
    save_dict(d_master, pool_file)
    print('saved '+pool_file)
    print('done')







