import torch
import numpy as np
from torch.utils.data import Dataset
import mat73
from scipy.io import loadmat
import pickle
from collections import defaultdict
import torch
import numpy as np

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
        self.dat , self.target=self.load_data()
        self.n_samples=self.dat.shape[0]
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        #item=np.expand_dims(self.dat[idx],axis=0)
        item=self.dat[idx]
        targ=np.squeeze(self.target[idx])
        return torch.tensor(item,dtype=torch.float), torch.tensor(targ,dtype=torch.long)
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
        # add extra component defining the graph and dataset.
        return dat_new, target

def resize_tensor(data,shape=(-1,3,32,32)):
    dat_new = np.reshape(data, shape)
    return dat_new

class sub_data(Dataset):
    def __init__(self, data_path,resize=True,transform=None):
        self.data_path=data_path
        mat = mat73.loadmat(data_path)
        ops = mat['ops_out']
        self.data = ops['data']  # obs! if loading using sio, do: ops['data'][0][0]
        self.targets = ops['class_id'].squeeze()
        self.targets = self.targets - 1
        #assert (np.sqrt(datatest.shape[1]) == type(int))
        self.transform = transform
        self.resize = resize
        self.structure = ops.structure
        self.n_class = int(ops.n_class)
        self.exm_per_class = int(ops.exm_per_class)
        self.beta = int(ops.beta)
        self.sigma = int(ops.sigma)
        self.data_latent=ops.data_latent
        self.n_feat=int(ops.n_feat)
        self.n_latent=int(ops.n_latent)
        self.is_norm=bool(ops.norm)
        self.hierarchical_target=[x-1 for x in ops.hierarchical_class_ids]
        if self.resize:
            self.resize_tensor()
    def resize_tensor(self, shape=(-1, 3, 32, 32)):
        self.data = np.reshape(self.data, shape)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        single_data = self.data[idx]
        single_target = self.targets[idx]
        target_tensor = torch.from_numpy(np.array(single_target))
        target = target_tensor.long()
        if self.transform is not None:
            single_data = self.transform(single_data)
        data_tensor = torch.from_numpy(single_data)
        data_tensor = data_tensor.type(torch.FloatTensor)
        return data_tensor, target




def create_manifold_data(dataset, sampled_classes, examples_per_class, max_class=None, seed=0):
    '''
    Samples manifold data for use in later analysis

    Args:
        dataset: PyTorch style dataset, or iterable that contains (input, label) pairs
        sampled_classes: Number of classes to sample from (must be less than or equal to
            the number of classes in dataset)
        examples_per_class: Number of examples per class to draw (there should be at least
            this many examples per class in the dataset)
        max_class (optional): Maximum class to sample from. Defaults to sampled_classes if unspecified
        seed (optional): Random seed used for drawing samples

    Returns:
        data: Iterable containing manifold input data
    '''
    if max_class is None:
        max_class = sampled_classes
    assert sampled_classes <= max_class, 'Not enough classes in the dataset'
    assert examples_per_class * max_class <= len(dataset), 'Not enough examples per class in dataset'

    # Set the seed
    np.random.seed(0)
    # Storage for samples
    sampled_data = defaultdict(list)
    # Sample the labels
    sampled_labels = np.random.choice(list(range(max_class)), size=sampled_classes, replace=False)
    # Shuffle the order to iterate through the dataset
    idx = [i for i in range(len(dataset))]
    np.random.shuffle(idx)
    # Iterate through the dataset until enough samples are drawn
    for i in idx:
        sample, label = dataset[i]
        label = int(label)
        if label in sampled_labels and len(sampled_data[label]) < examples_per_class:
            sampled_data[label].append(sample)
        # Check if enough samples have been drawn
        complete = True
        for s in sampled_labels:
            if len(sampled_data[s]) < examples_per_class:
                complete = False
        if complete:
            break
    # Check that enough samples have been found
    assert complete, 'Could not find enough examples for the sampled classes'
    # Combine the samples into batches
    data = []
    for s, d in sampled_data.items():
        data.append(torch.stack(d))
    return data

def train(epoch,model, device, train_loader,test_loader, optimizer,train_spec):
    model.train()
    test_accuracies = []
    train_accuracies = []
    log_interval=train_spec['log_interval']
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % log_interval == 0) & (batch_idx!=0):
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy_train = (100. * correct / len(target))
            train_accuracies.append(accuracy_train)
            print('Train Epoch: [{}/{} ({:.0f}%)]\Loss: {:.6f}, Train Accuracy: ({:.0f}%)'.format(
                 batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                 100. * correct / len(target)))
    epoch_dat = {
        "fc": [],
        "target": [],
        "batch": [],
        "epoch": epoch,
        "test_acc": test_accuracies,
        "train_acc": train_accuracies}

    # if is_cuda:
    epoch_dat['test_acc'] = np.stack(epoch_dat['test_acc'])
    epoch_dat['train_acc'] = np.stack(epoch_dat['train_acc'])
    epoch_dat['fc'] = np.concatenate(epoch_dat['fc'], axis=0)
    epoch_dat['target'] = np.concatenate(epoch_dat['target'])
    epoch_dat['batch'] = np.concatenate(epoch_dat['batch'])

    return epoch_dat

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
