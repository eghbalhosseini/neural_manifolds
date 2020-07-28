from __future__ import print_function
import torch
import copy

from mftma.utils.activation_extractor import extractor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from neural_manifolds_utils.neural_manifold_utils import train, test, train_test, save_dict, create_manifold_data, NN
from torch.utils.data.sampler import SubsetRandomSampler
import os, sys
import socket
from datetime import datetime
import getpass
import numpy as np
from neural_manifolds_utils import train_pool
# from mftma.utils.activation_extractor import extractor

print('__cuda available ',torch.cuda.is_available())
print('__Python VERSION:', sys.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
try:
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
    save_dir = '/om/user/gretatu/neural_manifolds/network_training_on_synthetic/'
    data_dir = '/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'

if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
#parser = argparse.ArgumentParser(description='neural manifold train network')
#parser.add_argument('datafile', type=str, default="synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat",help='')
#args=parser.parse_args()

if __name__=='__main__':
    model_identifer='[CNN]-[partition/nclass=100/nobj=100000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]'
    params=train_pool[model_identifer]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### DATA ####
    dataset = params.dataset
    exm_per_class = params.exm_per_class
    resize = params.resize
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(params.test_split * dataset_size))
    if params.shuffle_dataset:
        np.random.seed(params.random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # create train test splits
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size_train, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size_test, sampler=test_sampler)
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

    # Define train specs and model
    train_spec = {'train_batch_size': params.batch_size_train,
                  'test_batch_size': params.batch_size_test,
                  'num_epochs': params.epochs,
                  'structure': dataset.structure,
                  'n_class': dataset.n_class,
                  'exm_per_class': dataset.exm_per_class,
                  'beta': dataset.beta,
                  'sigma': dataset.sigma,
                  'norm': dataset.is_norm,
                  'log_interval': params.log_interval,
                  'is_cuda': torch.cuda.is_available()
                  }


    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)
    
    
    #### LOGGING ####
    # Tensorboard
    access_rights = 0o755
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_dir, 'runs', current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)
    # writer.add_graph(model)
    writer.add_hparams(hparam_dict=train_spec,metric_dict={})

    model_dir = save_dir + 'train_synthdata_' + dataset.structure + '_nclass_' + str(
        int(dataset.n_class)) + '_n_exm_' + str(int(dataset.exm_per_class))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    #### TRAINING ####
    for epoch in range(1, params.epochs + 1):
        data_ = {'activations_cell': [],
                 'train_spec': train_spec,
                 'train_accuracy': [],
                 'test_accuracy': [],
                 'train_success': False,
                 'epoch': epoch}
        
        train_test_data = []
        
        #### DEFINE TRAIN FUNCTION ####
        if params.train_type == 'train_test':
            epoch_dat = train_test(epoch, model, device, train_loader, test_loader, optimizer, train_spec, writer)
        if params.train_type == 'train':
            epoch_dat = train(epoch, model, device, train_loader, test_loader, optimizer, train_spec)
        else:
            raise ValueError('Training method not recognized')
        
        test_accuracy = test(model, device, test_loader, epoch)
        if test_accuracy > 70:
            train_success = True
        else:
            train_success = False
            
        # extract activation
        model = model.eval()
        activations_cell = [extractor(model, x) for x in hier_sample_mtmfa]
        data_['train_accuracy'] = epoch_dat['train_acc']
        data_['test_accuracy'] = test_accuracy
        data_['activations_cell'] = activations_cell
        data_['train_success'] = train_success
        num = str(epoch)
        enum = num.zfill(3)
        epoch_save_path = os.path.join(model_dir, 'train_epoch_' + str(enum))
        save_dict(data_, epoch_save_path)
        if train_success:
            print('successful training')
            break




