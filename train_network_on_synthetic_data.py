from __future__ import print_function
import torch
import copy

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from neural_manifolds_utils.neural_manifold_utils import train, test, train_test, save_dict, NN, show_cov
from torch.utils.data.sampler import SubsetRandomSampler
import os, sys
import socket
from datetime import datetime
import getpass
import numpy as np
from neural_manifolds_utils import train_pool

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
    model_identifer = '[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]' # TODO args
    params = train_pool[model_identifer]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### DATA ####
    dataset = params.dataset
    exm_per_class = params.exm_per_class

    # If plotting a subsampled cov matrix:
    # show_cov(dataset)

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

    # Define train specs and model
    train_spec = {'model_identifier': params.identifier,
                  'train_batch_size': params.batch_size_train,
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

    model = params.model(num_classes=params.nclass, num_fc1=params.shape[1])
    # model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)

    # Save the very initial weights
    model_untrained = model.state_dict()

    #### LOGGING ####
    # Tensorboard
    access_rights = 0o755
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_dir, 'runs', current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)
    # writer.add_graph(model)
    writer.add_hparams(hparam_dict=train_spec, metric_dict={})

    #### TRAINING ####
    for epoch in range(1, params.epochs + 1):

        #### DEFINE TRAIN FUNCTION ####
        if params.train_type == 'train_test':
            test_acc = train_test(epoch, model, device, train_loader, test_loader, optimizer, train_spec, writer)
        if params.train_type == 'train':
            epoch_dat = train(epoch, model, device, train_loader, test_loader, optimizer, train_spec)

        # Define when to stop training:
        stop = False

        if params.stop_criteria == 'test_performance':
            if test_acc:
                    if test_acc > 70:
                        train_success = True
            else:
                train_success = False

            # Stop if test accuracy reached, or at last epoch
            if train_success or epoch == params.epochs:
                stop = True
                if train_success:
                    print('Successful training - test accuracy > 70%')
                if epoch == params.epochs:
                    print('Model did not reach test accuracy > 70% - reaching end of epochs')

        #if params.stop_criteria ==

            # Save the test set used
            # Generate list of batch idx files
            num_batches = int(len(train_loader) / params.batch_size_train)
            num_batches_lst = []
            for i in range(0, num_batches):
                if (i % params.log_interval == 0) & (i != 0):
                    num_batches_lst.append(i)

            files = []
            for e in range(1, epoch + 1):
                for b in num_batches_lst:
                    files.append(params.identifier + '-[epoch=' + str(e) + ']' + '-[batchidx=' + str(b) + ']' + '.pth')

            d = {'test_loader': test_loader,
                 'model_untrained': model_untrained,
                 'files_generated': files}

            save_dict(d, 'master-'+params.identifier+'.pkl')

            break




