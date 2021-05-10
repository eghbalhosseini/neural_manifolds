
from utils import save_dir, data_dir, train_pool, analyze_dir,save_dict
from utils.analysis_utils import analyze_pool
import pickle
import os
import re
import numpy as np
import argparse
import collections
import fnmatch
def makehash():
    return collections.defaultdict(makehash)

parser = argparse.ArgumentParser(description='pool results of distance analysis')
parser.add_argument('model_id', type=str, default='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed')
parser.add_argument('analyze_id', type=str, default='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1')
parser.add_argument('train_id', type=str, default='')

args = parser.parse_args()


#save_dir='/Users/eghbalhosseini/Desktop/'
if __name__ == '__main__':
    model_identifier = args.model_id
    #model_identifier = 'NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed'
    analyze_identifier = args.analyze_id
    train_dir_identifier=args.train_id
    params = train_pool[model_identifier]()
    layer_names = params.get_layer_names()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))

    # manually walk through the files
    distance_files=[]
    for file in os.listdir(os.path.join(save_dir,analyze_identifier,model_identifier_for_saving,train_dir_identifier)):
        if fnmatch.fnmatch(file, '*distance_data_v2.pkl'):
            distance_files.append(os.path.join(save_dir,analyze_identifier,model_identifier_for_saving,train_dir_identifier,file))
    s = [re.findall('/\d+', x) for x in distance_files]
    s = [item for sublist in s for item in sublist]
    file_id = [int(x.split('/')[1]) for x in s]
    sorted_files = [distance_files[x] for x in np.argsort(file_id)]

    #TODO: there is an issue with file saving in writing of files : need to write a routine and fix this.
    # do layerwise saving
    distance_pooled = makehash()
    for id_file, file in enumerate(sorted_files):
        file=file.replace('\x00','')
        if os.path.exists(file):
            data_=pickle.load(open(file, 'rb'))
            s =re.findall('-batchidx=\d+', file)
            batchidx = [int(x.split('=')[1]) for x in s][0]
            s = re.findall('-epoch=\d+', file)
            epochidx = [int(x.split('=')[1]) for x in s][0]
            # create the dimensions and coordinates
            distance_data=data_['distance_data']
            print(file)
            for layer_name, hierarchies in distance_data.items():
                # values are dictionary of different hierarchies;
                 temp=distance_pooled[layer_name]
                 for hier_id, hier_val in hierarchies.items():
                     pair_distance_list=hier_val['distance']
                     temp2=dict(identifier=f'{layer_name}-hier= {hier_id}',epoch=epochidx,batchidx=batchidx,distance=np.stack(pair_distance_list))
                     distance_pooled[layer_name][hier_id][id_file]=temp2
        else:
            print(f"{file} is missing")
    d_master = {'model_identifier': model_identifier,
                'distance_results': distance_pooled,
                'file_generated': sorted_files}


    pool_file = os.path.join(save_dir,analyze_identifier, model_identifier_for_saving,train_dir_identifier, f'{model_identifier_for_saving}_distance_pooled_v2.pkl')
    save_dict(d_master, pool_file)
    print('saved '+pool_file)
    print('done')







