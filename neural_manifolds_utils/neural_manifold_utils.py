import numpy as np
from torch.utils.data import Dataset
import mat73
from scipy.io import loadmat
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class sub_data(Dataset):
    def __init__(self, data_path, shape=(1,3072), transform=None):
        self.data_path=data_path
        mat = mat73.loadmat(data_path)
        ops = mat['ops_out']
        self.data = ops['data']  # obs! if loading using sio, do: ops['data'][0][0]
        self.targets = ops['class_id'].squeeze()
        self.targets = self.targets - 1
        self.transform = transform
        self.shape = shape
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

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        single_data = self.data[idx]
        single_target = self.targets[idx]
        
        if self.transform is not None:
            single_data = self.transform(single_data)
            
        single_data = single_data.reshape(self.shape)
        data_tensor = torch.from_numpy(single_data)
        data_tensor = data_tensor.type(torch.FloatTensor)
        
        target_tensor = torch.from_numpy(np.array(single_target))
        target = target_tensor.long()
        
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
    fcs=[]
    targets=[]
    batchs=[]
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
        "fc": fcs,
        "target": target,
        "batch":  batchs,
        "epoch": epoch,
        "test_acc": test_accuracies,
        "train_acc": train_accuracies}

    # if is_cuda:
    if train_spec['is_cuda']:
        try:
            epoch_dat['test_acc'] = np.concatenate(epoch_dat['test_acc'])
            epoch_dat['train_acc'] = np.concatenate(epoch_dat['train_acc'])
            epoch_dat['fc'] = np.concatenate(epoch_dat['fc'], axis=0)
            epoch_dat['target'] = np.concatenate(epoch_dat['target'])
            epoch_dat['batch'] = np.concatenate(epoch_dat['batch'])
        except:
            print('couldnt reformat the data')
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
        epoch, test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))
    test_acu = 100. * correct / len(test_loader.sampler)
    return test_acu

def train_test(epoch,model,device,train_loader,test_loader,optimizer,train_spec,writer):
    model.train()

    target_all = []
    batch_all = []
    test_accuracies = []
    train_accuracies = []
    log_interval = train_spec['log_interval']
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print('In training batch idx loop')
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx % log_interval == 0) & (batch_idx != 0):
            # print('data len:', len(data))
            # print('target len: ', len(target))

            # Training error
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # _, pred = (torch.max(output_sq, 1)) # other way of computing max pred - better?
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy_train = (100. * correct / len(target))
            train_accuracies.append(accuracy_train)

            # Extract independent test set during training
            iteration = iter(test_loader)
            data_test, target_test = next(iteration)

            with torch.no_grad():  # don't save gradient
                # Allow eval for test set inference
                model.eval()
                data_test, target_test = data_test.to(device), target_test.to(device)
                output_test = model(data_test)
                output_test = torch.squeeze(output_test)

            target_all.append(target_test.cpu())
            batch_all.append(target_test.cpu() * 0 + batch_idx)

            # Test error
            pred_test = output_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_test = pred_test.eq(target_test.view_as(pred_test)).sum().item()
            accuracy_test = (100. * correct_test / len(target_test))
            test_accuracies.append(accuracy_test)

            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train Accuracy: ({:.0f}%), Test Accuracy: ({:.0f}%)'.format(
                    epoch, batch_idx * len(data), (len(train_loader) * len(target)),
                           100. * batch_idx / len(train_loader), loss.item(),
                    accuracy_train, accuracy_test))
            n_iter = batch_idx + (epoch - 1) * len(train_loader)  # eg epoch 2: 75 + 1*750
            
            if writer:
                writer.add_scalar('Loss - Train', loss, n_iter)
                writer.add_scalar('Accuracy - Train', accuracy_train, n_iter)
                writer.add_scalar('Accuracy - Test', accuracy_test, n_iter)
                # writer.add_embedding(fc,tag='test_batch',global_step=n_iter,metadata=target_test)

    epoch_dat = {
        "target": target_all,
        "batch": batch_all,
        "epoch": epoch,
        "test_acc": test_accuracies,
        "train_acc": train_accuracies}

    # if is_cuda:
    epoch_dat['test_acc'] = np.stack(epoch_dat['test_acc'])
    epoch_dat['train_acc'] = np.stack(epoch_dat['train_acc'])
    epoch_dat['target'] = np.concatenate(epoch_dat['target'])
    epoch_dat['batch'] = np.concatenate(epoch_dat['batch'])

    return epoch_dat

class NN(nn.Module):
    def __init__(self, num_classes=50, num_fc1=3072, num_fc2=1024, num_fc3=256):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(num_fc1, num_fc2)
        self.fc2 = nn.Linear(num_fc2, num_fc3)
        self.fc3 = nn.Linear(num_fc3, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class CNN(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        # print('conv1 size: ', np.shape(self.conv1))
        self.conv1_bn = nn.BatchNorm2d(10,eps=1e-09)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20,eps=1e-09)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        bn_1 = self.conv1_bn(self.conv1(x))
        x_pool2d = F.relu(F.max_pool2d(bn_1, 2))
        bn_2 = self.conv2_bn(self.conv2(x_pool2d))
        x_maxpool2d = F.relu(F.max_pool2d(self.conv2_drop(bn_2), 2))
        x_maxpool2d = x_maxpool2d.view(-1, 320) #flatten
        bn_fc = self.fc1_bn(self.fc1(x_maxpool2d))
        x_fc = F.relu(bn_fc)
        x_drop = F.dropout(x_fc, training=self.training)
        x_fc2 = self.fc2(x_drop)
        return F.log_softmax(x_fc2, dim=1)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


