from __future__ import print_function
import argparse
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from neural_manifold_utils import CFAR100_fake_dataset, train, test
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
    train_batch_size = 32
    test_batch_size = 1024
    epochs = 100
    train_dataset = CFAR100_fake_dataset(data_dir=os.path.join(data_dir, args.datafile))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=test_batch_size, shuffle=True)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(num_classes=train_dataset.n_class)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # create train log for tensorflow
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_dir, 'runs', current_time + '_' + socket.gethostname())
    #writer = SummaryWriter(log_dir=log_dir)
    #writer.add_hparams(hparam_dict=train_spec, metric_dict={})
    # do the training :
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test_accuracy = test(model, device, test_loader, epoch)
        if test_accuracy > 70:
            train_success=True
            break
        else:
            train_success=False
    #writer.flush()
    #writer.close()
    # save the model in case it was successful
    model_save_path = save_dir + 'VGG16_synthdata_' + train_dataset.structure + '_nclass_' + str(
        int(train_dataset.n_class)) + '_n_exm_' + str(int(train_dataset.exm_per_class))
    if train_success:
        torch.save(model.state_dict(), model_save_path)
    else:
        print('unsuccessful training')
    # end of file.

