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
#import matplotlib.pyplot as plt
from random import sample
from utils import save_dir
import matplotlib.pyplot as plt


class sub_data(Dataset):
    def __init__(self, data_path, shape=(1, 936), transform=None):
        self.data_path = data_path
        mat = mat73.loadmat(data_path)
        try:
            ops = mat['ops_out']
        except:
            ops = mat['ops_comp']
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
        self.data_latent = ops.data_latent
        self.n_feat = int(ops.n_feat)
        self.n_latent = int(ops.n_latent)
        self.is_norm = bool(ops.norm)
        self.hierarchical_target = [x - 1 for x in ops.hierarchical_class_ids]
        self.transformation_mats = self.create_hier_transform()

    def create_hier_transform(self):
        base_targets = self.hierarchical_target[0]
        target_classes = [np.unique(x) for x in self.hierarchical_target]
        target_index = [[np.argwhere(x == y) for y in np.unique(x)] for x in self.hierarchical_target]
        target_base_assignment =[[np.unique(np.asarray([[base_targets[z]] for z in y])) for y in x ] for x in target_index]
        tranformation_mats=[np.zeros((len(target_classes[0]),len(x))) for x in target_classes]

        for matrix_idx, matrix in enumerate(target_base_assignment):
            for column, row_array in enumerate(matrix):
                for row in row_array:
                    tranformation_mats[matrix_idx][int(row), column] = 1
        return tranformation_mats


    def create_hier_transorm(self):
        base_targets=self.hierarchical_target[0]
        target_classes=[x.unique() for x in self.hierarchical_target]
        target_index=[[np.argwhere(x==y) for y in x.unique()] for x in self.hierarchical_target]
        pass

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        single_data = self.data[idx]
        single_target = self.targets[idx]
        single_hier_target=[x[idx] for x in self.hierarchical_target]
        if self.transform is not None:
            single_data = self.transform(single_data)

        single_data = single_data.reshape(self.shape)
        data_tensor = torch.from_numpy(single_data)
        data_tensor = data_tensor.type(torch.FloatTensor)

        target_tensor = torch.from_numpy(np.array(single_target))
        target_long = target_tensor.long()
        target_dict = dict(target=target_long, hier_target=single_hier_target)
        return data_tensor, target_dict


def train(epoch, model, device, train_loader, test_loader, optimizer, train_spec):
    model.train()
    test_accuracies = []
    train_accuracies = []
    fcs = []
    targets = []
    batchs = []
    log_interval = train_spec['log_interval']
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % log_interval == 0) & (batch_idx != 0):
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
        "batch": batchs,
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


def train_test(epoch, model, device, train_loader, test_loader, optimizer, train_spec, writer):
    model.train()

    target_all = [] # for the test batches (not training)
    test_accuracies = []  # Return list of test accuracies for each epoch train_test is called
    train_accuracies = []
    log_interval = train_spec['log_interval']

    for batch_idx, (data, target_dict) in enumerate(train_loader):
        target = target_dict['target']
        hier_target = target_dict['hier_target']

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        output = model(data)
        loss = F.nll_loss(output, target)  # nn.MSELoss(output, target)#
        loss.backward()
        optimizer.step()

        if (batch_idx % log_interval == 0) & (batch_idx != 0):

            # Extract hierarchical probabilities
            pred_nolog = np.exp(output.detach().numpy())
            # np.sum(np.exp((F.log_softmax(x, dim=1)).detach().numpy()[1, 1:2]))

            # get hierarchical probs for each sample:
            hier_probs = [np.matmul(pred_nolog, hier) for hier in test_loader.dataset.transformation_mats]
            hier_pred = [torch.tensor(x).argmax(dim=1,keepdim=True) for x in hier_probs]

            # Training error
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # _, pred = (torch.max(output, 1)) # other way of computing max pred - better?

            # Assert that hierarchical and normal target matches:
            if not torch.equal(hier_pred[0], pred):
                print('hier prediction \n')
                print(torch.flatten(hier_pred[0]))
                print('prediction \n')
                print(torch.flatten(pred))

            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy_train = (100. * correct / len(target))
            train_accuracies.append(accuracy_train)

            # Hierarchical accuracy
            hier_correct = [x.eq(torch.tensor(hier_target[idx]).view_as(x)).sum().item() for idx, x in enumerate(hier_pred)]
            hier_accuracy_train = [(100.*x/len(hier_target[idx])) for idx, x in enumerate(hier_correct)]

            # Extract gradients from training
            model_modules = model.__dict__['_modules']
            grad_dict = {}
            for k, v in model_modules.items():
                grad_dict[k] = v.weight.grad

            # Extract independent test set during training
            iteration = iter(test_loader)
            data_test, target_test_dict = next(iteration)
            target_test = target_test_dict['target']
            hier_target_test = target_test_dict['hier_target']

            with torch.no_grad():  # don't save gradient, faster inference
                # Allow eval for test set inference
                model.eval()
                data_test, target_test = data_test.to(device), target_test.to(device)
                output_test = model(data_test)
                pred_nolog_test = np.exp(output_test.detach().numpy()) # predictions not in log space

            target_all.append(target_test.cpu())

            # Test error
            pred_test = output_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_test = pred_test.eq(target_test.view_as(pred_test)).sum().item()
            accuracy_test = (100. * correct_test / len(target_test))
            test_accuracies.append(accuracy_test)
            # hierarchical test error
            hier_probs_test = [np.matmul(pred_nolog_test, hier) for hier in test_loader.dataset.transformation_mats]
            hier_pred_test = [torch.tensor(x).argmax(dim=1, keepdim=True) for x in hier_probs_test]

            hier_correct_test = [x.eq(torch.tensor(hier_target_test[idx]).view_as(x)).sum().item() for idx, x in
                            enumerate(hier_pred_test)]
            hier_accuracy_test = [(100. * x / len(hier_target_test[idx])) for idx, x in enumerate(hier_correct_test)]

            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Train Accuracy: ({:.0f}%), Test Accuracy: ({:.0f}%)'.format(
                    epoch, batch_idx * len(data), (len(train_loader) * len(target)),
                           100. * batch_idx / len(train_loader), loss.item(),
                    accuracy_train, accuracy_test))

            n_iter = batch_idx + (epoch - 1) * len(train_loader)  # eg epoch 2: 75 + 1*1250

            if writer:
                writer.add_scalar('Loss - Train', loss, n_iter)
                writer.add_scalar('Accuracy - Train', accuracy_train, n_iter)
                writer.add_scalar('Accuracy - Test', accuracy_test, n_iter)
                # writer.add_embedding(fc,tag='test_batch',global_step=n_iter,metadata=target_test)

            # Save weights and accuracies
            state = {
                'state_dict': model.state_dict(),
                'grad_dict': grad_dict,
                'optimizer': optimizer.state_dict(),
                'train_acc': accuracy_train,
                'test_acc': accuracy_test,
                'hier_train_acc': hier_accuracy_train,
                'hier_test_acc': hier_accuracy_test,
                'data_test': data_test,
                'target_test': target_test,
                'pred_test': pred_test, # category prediction, test set
                'pred_test_prob': pred_nolog_test, # probability not in log space of the 64 categories, test set
                'hier_target_test': hier_target_test,
                'epoch': epoch,
                'batchidx': batch_idx, }  # add test targets etc for starters

            # Define file names for saving
            pth_name = train_spec['model_identifier'] + '-epoch=' + str(epoch).zfill(2) + '-batchidx=' + str(batch_idx) + '.pth'
            torch.save(state, os.path.join(save_dir, train_spec['model_identifier'], train_spec['training_folder'], pth_name))
            print("Saving model for epoch {:d}, batch idx {:d}\n".format(epoch, batch_idx))

            # Print train and test accuracies
            if epoch == 1 and batch_idx == log_interval:
                append_write = 'w'  # make a new file if not
            else:
                append_write = 'a'

            train_acc_txt = open(
                save_dir + '/' + train_spec['model_identifier'] + '/' + train_spec['training_folder'] +
                '/acc_train_' + train_spec['model_identifier'] + '.csv', append_write)
            train_acc_txt.writelines(str(accuracy_train) + '\n')
            train_acc_txt.close()

            test_acc_txt = open(
                save_dir + '/' + train_spec['model_identifier'] + '/' + train_spec['training_folder'] +
                '/acc_test_' + train_spec['model_identifier'] + '.csv', append_write)
            test_acc_txt.writelines(str(accuracy_test) + '\n')
            test_acc_txt.close()

    return test_accuracies


class NN(nn.Module):
    def __init__(self, num_classes=64, num_fc1=936, num_fc2=624, num_fc3=208,
                 init_type='gaussian', lower_bound=-1, upper_bound=1, mu=0, std=1, bias=False):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(num_fc1, num_fc2)
        self.fc2 = nn.Linear(num_fc2, num_fc3)
        self.fc3 = nn.Linear(num_fc3, num_classes)

        if init_type == 'uniform':
            torch.nn.init.uniform_(self.fc1.weight, a=lower_bound, b=upper_bound)
            torch.nn.init.uniform_(self.fc2.weight, a=lower_bound, b=upper_bound)
            torch.nn.init.uniform_(self.fc3.weight, a=lower_bound, b=upper_bound)

        if init_type == 'gaussian':
            torch.nn.init.normal_(self.fc1.weight, mean=mu, std=np.sqrt(std)) # pytorch uses std^2, so this ensures that the std is the one inputted in the function, and not std^2
            torch.nn.init.normal_(self.fc2.weight, mean=mu, std=np.sqrt(std))
            torch.nn.init.normal_(self.fc3.weight, mean=mu, std=np.sqrt(std))

        if init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fc1.weight) # gain is 1 by default
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.fc3.weight)
        #
        # if init_type == 'xavier_normal':
        #     torch.nn.init.xavier_normal_(m.weight.data) # gain is 1 by default

        if not bias:
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.bias)
            torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.size())
        x = torch.squeeze(x)
        # print(x.size())
        x_out = F.log_softmax(x, dim=1)

        return x_out

        # return torch.sigmoid(self.fc3(x))
        # return torch.log_softmax(self.fc3(x))

class linear_NN(nn.Module):
    def __init__(self, num_classes=64, num_fc1=936, num_fc2=624, num_fc3=208,
                 init_type='gaussian', lower_bound=-1, upper_bound=1, mu=0, std=1, bias=False):
        super(linear_NN, self).__init__()
        self.fc1 = nn.Linear(num_fc1, num_fc2)
        self.fc2 = nn.Linear(num_fc2, num_fc3)
        self.fc3 = nn.Linear(num_fc3, num_classes)

        if init_type == 'uniform':
            torch.nn.init.uniform_(self.fc1.weight, a=lower_bound, b=upper_bound)
            torch.nn.init.uniform_(self.fc2.weight, a=lower_bound, b=upper_bound)
            torch.nn.init.uniform_(self.fc3.weight, a=lower_bound, b=upper_bound)

        if init_type == 'gaussian':
            torch.nn.init.normal_(self.fc1.weight, mean=mu, std=np.sqrt(std)) # pytorch uses std^2, so this ensures that the std is the one inputted in the function, and not std^2
            torch.nn.init.normal_(self.fc2.weight, mean=mu, std=np.sqrt(std))
            torch.nn.init.normal_(self.fc3.weight, mean=mu, std=np.sqrt(std))

        if init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fc1.weight) # gain is 1 by default
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.fc3.weight)

        if not bias:
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.bias)
            torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # print(x.size())
        x = torch.squeeze(x)
        # print(x.size())
        x_out = F.log_softmax(x, dim=1)
        # print(yy.size())

        return x_out



class CNN(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        # print('conv1 size: ', np.shape(self.conv1))
        self.conv1_bn = nn.BatchNorm2d(10, eps=1e-09)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20, eps=1e-09)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        bn_1 = self.conv1_bn(self.conv1(x))
        x_pool2d = F.relu(F.max_pool2d(bn_1, 2))
        bn_2 = self.conv2_bn(self.conv2(x_pool2d))
        x_maxpool2d = F.relu(F.max_pool2d(self.conv2_drop(bn_2), 2))
        x_maxpool2d = x_maxpool2d.view(-1, 320)  # flatten
        bn_fc = self.fc1_bn(self.fc1(x_maxpool2d))
        x_fc = F.relu(bn_fc)
        x_drop = F.dropout(x_fc, training=self.training)
        x_fc2 = self.fc2(x_drop)
        return F.log_softmax(x_fc2, dim=1)


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def show_cov(dataset, frac=50):
    fname = dataset.data_path
    fname_split = fname.split('/')
    subsample_idx = sample(list(range(1, len(dataset.data))), int(len(dataset.data) / frac))
    subsample_labels = dataset.targets[subsample_idx]
    argsorted = np.argsort(subsample_labels)

    subsample_data = dataset.data[subsample_idx, :]
    sorted_data = subsample_data[argsorted]  # sort according to the label ordering
    sorted_labels = subsample_labels[argsorted]  # for sanity checking

    cov = np.cov((sorted_data))

    plt.figure()
    plt.imshow(cov)
    plt.savefig(save_dir + 'cov_' + fname_split[-1][:-4] + '.pdf')

def leaf_traverse(root, flat_children):
    '''
    Get all the layers of the model
    '''
    if len(list(root.children())) == 0:
        flat_children.append(root)
    else:
        for child in root.children():
            leaf_traverse(child, flat_children)

def add_layer_names(flat_children):
    '''
    Count the layers in the model and add names to them
    '''
    count = 1
    for child in flat_children:
        name = "layer_" + str(count) + "_" + child._get_name()
        child.__setattr__('layer_name', name)
        count += 1


