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
import re

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
    model_identifer = '[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]' # TODO args
    params = train_pool[model_identifer]()
    model_identifier_for_saving = params.identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
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
    train_spec = {'model_identifier': model_identifier_for_saving,
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
    for epoch in range(1, params.epochs + 1): # e.g. if epochs = 50, then running from 1 to 50

        #### DEFINE TRAIN FUNCTION ####
        if params.train_type == 'train_test':
            test_accuracies = train_test(epoch, model, device, train_loader, test_loader, optimizer, train_spec, writer)
        if params.train_type == 'train':
            epoch_dat = train(epoch, model, device, train_loader, test_loader, optimizer, train_spec)

        # Define when to stop training:
        train_success = False
        if params.stop_criteria == 'test_performance':
            num_last_test_accs = 3
            if num_last_test_accs < len(test_accuracies):
                test_acc_mean = np.mean(test_accuracies[-num_last_test_accs:])

            if num_last_test_accs >= len(test_accuracies):
                test_acc_mean = np.mean(test_accuracies) # mean over all values

            print('Mean test acc: ', test_acc_mean)
            if test_acc_mean > 50:
                train_success = True

        # Stop if test accuracy reached, or at last epoch
        if train_success or epoch == params.epochs:
            if train_success:
                print('Successful training - test accuracy > 50%')
            if epoch == params.epochs:
                print('Reaching end of epochs')

            # Save the test set used (either at test accuracy performance, or reaching end of epochs)
            # Generate list of batch idx files
            num_batches = int(len(train_loader) / params.batch_size_train)
            num_batches_lst = []
            for i in range(1, num_batches):
                num_batches_lst.append(i*params.log_interval)

            files = []
            generated_files_txt = open(save_dir + 'master_' + model_identifier_for_saving + '.txt', 'w')
            for e in range(1, epoch + 1):
                for b in num_batches_lst:
                    pth_file = model_identifier_for_saving + '-epoch=' + str(e) + '-batchidx=' + str(b) + '.pth'
                    files.append(pth_file)

                    # Write to txt file
                    generated_files_txt.writelines(pth_file + '\n')

            generated_files_txt.close()

            d_master = {'test_loader': test_loader,
                 'model_untrained': model_untrained,
                 'files_generated': files}

            save_dict(d_master, save_dir + 'master_'+model_identifier_for_saving+'.pkl')

            break # Break statement in case the end was not reached (test accuracy termination)





