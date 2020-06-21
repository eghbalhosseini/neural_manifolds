import torch
import numpy as np
from torch.utils.data import Dataset
import mat73
import scipy
import scipy.io as sio
from scipy.io import loadmat

class CFAR100_fake_dataset(Dataset):
    def __init__(self, data_dir=None):
        self.data_dir=data_dir
        self.dat , self.target=self.load_data()
        self.n_samples=self.dat.shape[0]
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        item=self.dat[idx]
        targ=np.squeeze(self.target[idx])
        return (torch.tensor(item,dtype=torch.float), targ)
    def load_data(self):
        try:
            annot=loadmat(self.data_dir)
            ops_struct=annot['ops_out']
            vals=ops_struct[0,0]
        except:
            data_dict = mat73.loadmat(self.data_dir)
            vals=data_dict['ops_out']
        dat=vals['data']
        self.vals=vals
        self.adj=vals['Adjacency']
        dat_new=dat[:,range(3*32*32)]
        dat_new=np.reshape(dat_new,(-1,3,32,32))
        target=np.double(np.transpose(vals['class_id'])-1.0)
        return dat_new, target