from mftma.utils.activation_extractor import extractor
from collections import defaultdict
import numpy as np
import torch


def project( activation, max_dim=5000):
    for layer, data, in activation.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > max_dim:
            print("Projecting {}".format(layer))
            M = np.random.randn(max_dim, N)
            M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]
        activation[layer] = X
    return activation

class mftma_extractor(object):
    def __init__(self,model=None, exm_per_class=50, nclass=50, data=None,max_dim=5000):
        self.extractor=extractor
        self.exm__per_class=exm_per_class
        self.nclass=nclass
        self.data=data
        self.max_dim=max_dim
        self.project=project

    # there should be a section for hierarchical data used

def make_manifold_data(dataset, sampled_classes, examples_per_class, max_class=None, seed=0,randomize=False):
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
        randomize= True/False, if false the function starts from the first index and samples sequentially until all groups are filled,
        otherwise it will randomly sample from the dataset.

    Returns:
        data: Iterable containing manifold input data
    '''
    if max_class is None:
        max_class = sampled_classes
    assert sampled_classes <= max_class, 'Not enough classes in the dataset'
    assert examples_per_class * max_class <= len(dataset), 'Not enough examples per class in dataset'

    # Set the seed
    np.random.seed(seed)
    # Storage for samples
    sampled_data = defaultdict(list)
    # Sample the labels
    sampled_labels = np.random.choice(list(range(max_class)), size=sampled_classes, replace=False)
    # Shuffle the order to iterate through the dataset
    idx = [i for i in range(len(dataset))]
    if randomize:
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

