
import pickle
import os
import glob
import re
import numpy as np
import argparse
import scipy.io
import mat73
from glob import glob

os.chdir('/Users/gt/Documents/GitHub/neural_manifolds')
model_identifier = 'NN-tree_nclass=64_nobj=64000_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed'
feature='testAcc'


def getKNN(model_identifier, feature):
    d_model = {}

    # cd into analyze identifier folder

    # find files corresponding to the correct model identifier
    # model_files = glob('NN-tree_nclass=64_nobj=64000_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed*')

    model_files = glob(model_identifier + '*')
    sorted_model_files = np.sort(model_files)
    # extract per layer


    for layerFile in sorted_model_files:
        print(layerFile) # check that order corresponds
        startIdx = sorted_model_files[0].index('layer')
        endIdx = sorted_model_files[0].index('_numSub')

        layerName = sorted_model_files[0][startIdx:endIdx]

        mat = mat73.loadmat(layerFile)

        superStruct = mat['super_struct']

        for hier in range(1, len(superStruct)):
            hierFile = superStruct['hier_' + str(hier)]

            assert(hier == int(hierFile['hier_level']))

            featureOfInterest = hierFile[feature]

            d_model[layerName] = featureOfInterest

    return d_model







# parser = argparse.ArgumentParser(description='run mftma and save results')
# parser.add_argument('model_id', type=str, default='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed')
# parser.add_argument('analyze_id', type=str, default='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1')
# args = parser.parse_args()


