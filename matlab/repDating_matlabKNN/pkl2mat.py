import numpy as np
import scipy.io
import pickle

g=pickle.load(open('/Users/gt/Documents/GitHub/neural_manifolds/matlab/repDating_matlabKNN/layer=layer_1_Linear_hier=1_v3.pkl','rb'))

scipy.io.savemat('/Users/gt/Documents/GitHub/neural_manifolds/matlab/repDating_matlabKNN/layer=layer_1_Linear_hier=1_v3.mat', mdict={'data': g})