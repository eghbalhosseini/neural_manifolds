from __future__ import print_function
import argparse
import torch
import copy
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from neural_manifold_utils import CFAR100_fake_dataset, CFAR100_fake_dataset_mftma, train, test, save_dict
import os, sys
import socket
from datetime import datetime
import getpass
print('__cuda available ',torch.cuda.is_available())
print('__Python VERSION:', sys.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Device name:', torch.cuda.get_device_name(0))

user=getpass.getuser()
print(user)
if user=='eghbalhosseini':
    save_dir='/Users/eghbalhosseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/Users/eghbalhosseini/MyData/neural_manifolds/synthetic_datasets/'
elif user=='ehoseini':
    save_dir='/om/user/ehoseini/MyData/neural_manifolds/network_training_on_synthetic/'
    data_dir='/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/'

parser = argparse.ArgumentParser(description='neural manifold train network')
parser.add_argument('datafile', type=str, default="synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat",help='')
args=parser.parse_args()

if __name__=='__main__':
    # load dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = 32
    test_batch_size = 1024
    epochs = 100
    exm_per_class=100
    train_dataset = CFAR100_fake_dataset(data_dir=os.path.join(data_dir, args.datafile))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=test_batch_size, shuffle=True)

    train_dataset_mtmfa = CFAR100_fake_dataset_mftma(data_dir=os.path.join(data_dir, args.datafile))
    num_hierarchy=len(train_dataset_mtmfa.vals.hierarchical_class_ids)
    hier_classes = [x - 1 for x in train_dataset_mtmfa.vals.hierarchical_class_ids]
    hier_n_class = [int(max(x) + 1) for x in hier_classes]
    hier_dat_mtmfa=[]

    for idx, x in enumerate(hier_classes):
        dat_mfmta = []
        dat_mtmfa = copy.deepcopy(train_dataset_mtmfa)
        dat_mtmfa.target = hier_classes[idx]
        hier_dat_mtmfa.append(copy.deepcopy(dat_mtmfa));
    hier_sample_mtmfa= [make_manifold_data(x, hier_n_class[idx], exm_per_class, seed=0) for idx, x in enumerate(hier_dat_mtmfa)]
    # specs
    train_spec = {'train_batch_size': train_batch_size,
                  'test_batch_size': test_batch_size,
                  'num_epochs': epochs,
                  'structure': train_dataset.structure,
                  'n_class': train_dataset.n_class,
                  'exm_per_class': train_dataset.exm_per_class,
                  'beta': train_dataset.beta,
                  'sigma': train_dataset.sigma,
                  'norm': train_dataset.is_norm
                  }


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # build model
    model = models.vgg16(num_classes=train_dataset.n_class)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # create train log for tensorflow
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_dir, 'runs', current_time + '_' + socket.gethostname())
    #writer = SummaryWriter(log_dir=log_dir)
    #writer.add_hparams(hparam_dict=train_spec, metric_dict={})
    # do the training :
    train_test_accu=[]
    for epoch in range(1, epochs + 1):
        train_accuracy = train(model, device, train_loader, optimizer)
        test_accuracy = test(model, device, test_loader, epoch)
        # extract activation
        model = model.eval()
        activations_cell = [extractor(model,x,layer_types=['Conv2d', 'Linear']) for x in hier_sample_mtmfa]
        if test_accuracy > 70:
            train_success=True
            train_test_accu.append([train_accuracy, test_accuracy, train_success])
            break
        else:
            train_success=False
            train_test_accu.append([train_accuracy, test_accuracy,train_success])

    data_ = {'activations_cells': activations_cell,
             'exm_per_class': exm_per_class,
             'num_hierarchy': num_hierarchy,
             'train_test_accuracy': train_test_accu,
             'structure':train_dataset.structure,
             'train_success':train_success
             }

    result_save_path = save_dir + 'mftma_data_VGG16_synthdata_' + data_['structure'] + '_n_hier_' + str(data_['num_hierarchy']) \
                       + '_exm_per_class_' + str(data_['exm_per_class']) +'train_success' +str(data_['train_success'])
    save_dict(data_, result_save_path)
    # save the model in case it was successful
    model_save_path = save_dir + save_dir + 'VGG16_synthdata_' + data_['structure'] + '_n_hier_' + str(data_['num_hierarchy']) \
                       + '_exm_per_class_' + str(data_['exm_per_class']) +'train_success' +str(data_['train_success'])
    if train_success:
        torch.save(model.state_dict(), model_save_path)
    else:
        print('unsuccessful training')

    # to do : add a function for extracting activations,


