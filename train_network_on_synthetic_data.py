from __future__ import print_function
import argparse
import torch
import copy
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from neural_manifold_utils import train, test, save_dict, sub_data, create_manifold_data
from torch.utils.data.sampler import SubsetRandomSampler
import os, sys
import socket
from datetime import datetime
import getpass
import numpy as np

from mftma.utils.activation_extractor import extractor
print('__cuda available ',torch.cuda.is_available())
print('__Python VERSION:', sys.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
try :
    print('__Device name:', torch.cuda.get_device_name(0))
except:
    print('no gpu to run')

user=getpass.getuser()
print(user)
if user=='eghbalhosseini':
    save_dir='/Users/eghbalhosseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/Users/eghbalhosseini/MyData/neural_manifolds/synthetic_datasets/'
elif user=='ehoseini':
    save_dir='/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'
elif user == 'gretatu':
    save_dir = '/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir = '/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'

if not os.path.exists(save_dir):
        os.makedirs(save_dir)
parser = argparse.ArgumentParser(description='neural manifold train network')
parser.add_argument('datafile', type=str, default="synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat",help='')
args=parser.parse_args()

if __name__=='__main__':
    # load dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size_train = 64
    batch_size_test = 64
    resize = True
    epochs = 50
    exm_per_class = 100
    save_epochs = False
    momentum = 0.5
    lr = 0.01
    gamma = 0.7
    log_interval = 75
    test_split = .2
    shuffle_dataset = True
    random_seed = 1
    #
    dataset = sub_data(data_path=os.path.join(data_dir, args.datafile))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # create train test splits
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, sampler=test_sampler)
    # create manifold analysis dataset:
    num_hierarchy = len(dataset.hierarchical_target)
    hier_classes = [x.astype(int) for x in dataset.hierarchical_target]
    hier_n_class = [int(max(x) + 1) for x in hier_classes]
    hier_dataset = []
    for idx, x in enumerate(hier_classes):
        dat_mfmta = []
        dat_mtmfa = copy.deepcopy(dataset)
        dat_mtmfa.targets = hier_classes[idx]
        hier_dataset.append(copy.deepcopy(dat_mtmfa))

    hier_sample_mtmfa = [create_manifold_data(x, hier_n_class[idx], exm_per_class, seed=0) for idx, x in enumerate(hier_dataset)]
    hier_sample_mtmfa = [[d.to(device) for d in data] for data in hier_sample_mtmfa]

    # define train specs and model
    train_spec = {'train_batch_size': batch_size_train,
                  'test_batch_size': batch_size_test,
                  'num_epochs': epochs,
                  'structure': dataset.structure,
                  'n_class': dataset.n_class,
                  'exm_per_class': dataset.exm_per_class,
                  'beta': dataset.beta,
                  'sigma': dataset.sigma,
                  'norm': dataset.is_norm,
                  'log_interval': log_interval,
                  'is_cuda': torch.cuda.is_available()
                  }

    model = models.vgg16(num_classes=dataset.n_class)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_dir, 'runs', current_time + '_' + socket.gethostname())
    model_dir = save_dir + 'train_VGG16_synthdata_' + dataset.structure + '_nclass_' + str(
        int(dataset.n_class)) + '_n_exm_' + str(int(dataset.exm_per_class))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # training :

    for epoch in range(1, epochs + 1):
        data_ = {'activations_cell': [],
                 'train_spec': train_spec,
                 'train_accuracy': [],
                 'test_accuracy': [],
                 'train_success': False,
                 'epoch': epoch}
        train_test_data = []
        epoch_dat = train(epoch, model, device, train_loader, test_loader, optimizer, train_spec)
        test_accuracy = test(model, device, test_loader, epoch)
        if test_accuracy > 70:
            train_success = True
        else:
            train_success = False
        # extract activation
        model = model.eval()
        activations_cell = [extractor(model, x, layer_types=['Conv2d', 'Linear']) for x in hier_sample_mtmfa]
        data_['train_accuracy']=epoch_dat['train_acc']
        data_['test_accuracy']=test_accuracy
        data_['activations_cell'] = activations_cell
        data_['train_success'] = train_success
        epoch_save_path = os.path.join(model_dir, 'train_epoch_' + str(epoch))
        save_dict(data_, epoch_save_path)
        if train_success:
            print('successful training')
            break




