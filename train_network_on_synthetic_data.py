from __future__ import print_function
import torch
import copy

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from utils.model_utils import train, train_test, save_dict, NN, show_cov
from torch.utils.data.sampler import SubsetRandomSampler
import os, sys
import socket
from datetime import datetime
import getpass
import numpy as np
from utils import save_dir, train_pool
import re
import argparse

print('__cuda available ', torch.cuda.is_available())
print('__Python VERSION:', sys.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
try:
    print('__Device name:', torch.cuda.get_device_name(0))
except:
    print('no gpu to run')

user = getpass.getuser()
print(user)

parser = argparse.ArgumentParser(description='neural manifold train network')
parser.add_argument('model_identifier', type=str, default="synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat", help='')
args = parser.parse_args()

if __name__ == '__main__':
    model_identifier = args.model_identifier
    params = train_pool[model_identifier]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### DATA ####
    params.load_dataset()
    dataset = params.dataset
    exm_per_class = dataset.exm_per_class

    # If plotting a subsampled cov matrix:
    # show_cov(dataset)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(params.test_split * dataset_size))
    if params.shuffle_dataset:
        np.random.seed(params.random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # create sample pairs for doing similarity between hierarchies (pine/oak ...)
    num_pair = params.n_pairs
    L=params.dataset.hierarchical_target
    L=L[::-1] # backward
    L_unique = [list(set(x)) for x in L ]
    ASSEMBLY = dict()
    for idx, level in enumerate(L_unique[:-1]):
        pair_assembly = dict(n_hier=len(level), index_pairs=[])
        for l in level:
            indexes = np.squeeze(np.argwhere(np.asarray(L[idx]) == l))
            valid_pairs = []
            while len(valid_pairs) < num_pair:
                pairs = [np.random.choice(indexes, 2, replace=True) for x in range(300)]
                print(len(valid_pairs))
                [valid_pairs.append(x) for x in pairs if L[idx + 1][x[0]] != L[idx + 1][x[1]]]
            pair_assembly['index_pairs'].append(valid_pairs[:num_pair])
        ASSEMBLY[idx] = pair_assembly



    # create train test splits
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size_train, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size_test, sampler=test_sampler)

    # Define train specs and model
    train_spec = {'model_identifier': model_identifier,
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
    model = model.to(device)
    model_initialized = copy.deepcopy(model)

    # Save the very initial weights
    model_untrained = model.state_dict()

    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)
    optimizer_initialized = copy.deepcopy(optimizer)

    #### LOGGING ####
    # Tensorboard
    access_rights = 0o755
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_dir, model_identifier, 'runs', current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparam_dict=train_spec, metric_dict={})

    #### TRAINING ####
    for epoch in range(1, params.epochs + 1):  # e.g. if epochs = 50, then running from 1 to 50

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
                test_acc_mean = np.mean(test_accuracies)  # mean over all values

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
            num_batches = int(len(train_loader))
            num_batches_lst = []
            for batch_idx in range(1, num_batches):
                if (batch_idx % params.log_interval == 0) & (batch_idx != 0):
                    num_batches_lst.append(batch_idx)

            files = []
            generated_files_csv = open(save_dir + '/' + model_identifier + '/master_' + model_identifier + '.csv', 'w')
            for e in range(1, epoch + 1):
                for b in num_batches_lst:
                    #pth_file = save_dir + '/' + model_identifier + '/' + model_identifier + '-epoch=' + str(e) + '-batchidx=' + str(b) + '.pth'
                    pth_file = os.path.join(save_dir, model_identifier,
                                 model_identifier + '-epoch=' + str(e) + '-batchidx=' + str(b) + '.pth')
                    files.append(pth_file)

                    # Write to csv file
                    generated_files_csv.writelines(pth_file + '\n')

            generated_files_csv.close()

            d_master = {'test_loader': test_loader,
                        'model_untrained_weights': model_untrained,
                        'model_structure': model_initialized,
                        'optimizer_structure': optimizer_initialized,
                        'files_generated': files,
                        'distance_pair_index':ASSEMBLY}

            save_dict(d_master, save_dir + '/' + model_identifier + '/master_' + model_identifier + '.pkl')

            break  # Break statement in case the end was not reached (test accuracy termination)





