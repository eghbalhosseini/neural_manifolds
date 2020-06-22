import torch
import numpy as np
from torch.utils.data import Dataset
import mat73
from scipy.io import loadmat
import pickle

class CFAR100_fake_dataset_mftma(Dataset):
    def __init__(self, data_dir=None):
        self.data_dir=data_dir
        self.data = []
        self.targets = []
        self.dat , self.target=self.load_data()
        self.n_samples=self.dat.shape[0]
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        #item=np.expand_dims(self.dat[idx],axis=0)
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
        self.beta=float(vals['beta'])
        self.sigma = float(vals['sigma'])
        self.data_latent=vals['data_latent']
        self.exm_per_class=int(vals['exm_per_class'])
        self.n_class=int(vals['n_class'])
        self.n_feat=int(vals['n_feat'])
        self.n_latent=int(vals['n_latent'])
        self.is_norm=bool(vals['norm'])
        self.structure = str(vals['structure'])
        dat_new=dat[:,range(3*32*32)]
        dat_new=np.reshape(dat_new,(-1,3,32,32))
        target=np.double(np.transpose(vals['class_id'])-1.0)
        #target=list(vals.class_id.astype(int))
        # add extra component defining the graph and dataset.
        return dat_new, target
    
class CFAR100_fake_dataset(Dataset):
    def __init__(self, data_dir=None):
        self.data_dir=data_dir
        self.data = []
        self.targets = []
        self.dat , self.target=self.load_data()
        self.n_samples=self.dat.shape[0]
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        #item=np.expand_dims(self.dat[idx],axis=0)
        item=self.dat[idx]
        targ=np.squeeze(self.target[idx])
        torch.tensor(item,dtype=torch.float), torch.tensor(targ,dtype=torch.long)
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
        self.beta=float(vals['beta'])
        self.sigma = float(vals['sigma'])
        self.data_latent=vals['data_latent']
        self.exm_per_class=int(vals['exm_per_class'])
        self.n_class=int(vals['n_class'])
        self.n_feat=int(vals['n_feat'])
        self.n_latent=int(vals['n_latent'])
        self.is_norm=bool(vals['norm'])
        self.structure = str(vals['structure'])
        dat_new=dat[:,range(3*32*32)]
        dat_new=np.reshape(dat_new,(-1,3,32,32))
        target=np.double(np.transpose(vals['class_id'])-1.0)
        #target=list(vals.class_id.astype(int))
        # add extra component defining the graph and dataset.
        return dat_new, target

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % 1000 == 0) & (batch_idx!=0):
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('Train Epoch: [{}/{} ({:.0f}%)]\Loss: {:.6f}, Train Accuracy: ({:.0f}%)'.format(
                 batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                 100. * correct / len(target)))


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.nn.functional.log_softmax(output, dim=1)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acu = 100. * correct / len(test_loader.dataset)
    return test_acu

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
