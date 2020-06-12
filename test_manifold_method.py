#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:17:40 2020

@author: eghbalhosseini
"""
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze
#%% 
def CGmanopt(X, objective_function, A, **kwargs):
    '''
    Minimizes the objective function subject to the constraint that X.T * X = I_k using the
    conjugate gradient method

    Args:
        X: Initial 2D array of shape (n, k) such that X.T * X = I_k
        objective_function: Objective function F(X, A) to minimize.
        A: Additional parameters for the objective function F(X, A)

    Keyword Args:
        None

    Returns:
        Xopt: Value of X that minimizes the objective subject to the constraint.
    '''

    manifold = Stiefel(X.shape[0], X.shape[1])
    def cost(X):
        c, _ = objective_function(X, A)
        return c
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    solver = ConjugateGradient(logverbosity=0)
    Xopt = solver.solve(problem)
    return Xopt, None

def square_corrcoeff_full_cost(V, X, grad=True):
    '''
    The cost function for the correlation analysis. This effectively measures the square difference
    in correlation coefficients after transforming to an orthonormal basis given by V.

    Args:
        V: 2D array of shape (N, K) with V.T * V = I
        X: 2D array of shape (P, N) containing centers of P manifolds in an N=P-1 dimensional
            orthonormal basis
    '''
    # Verify that the shapes are correct
    P, N = X.shape
    N_v, K = V.shape
    assert N_v == N

    # Calculate the cost
    C = np.matmul(X, X.T)
    c = np.matmul(X, V)
    c0 = np.diagonal(C).reshape(P, 1) - np.sum(np.square(c), axis=1, keepdims=True)
    Fmn = np.square(C - np.matmul(c, c.T))/np.matmul(c0, c0.T)
    cost = np.sum(Fmn)/2

    if grad is False:  # skip gradient calc since not needed, or autograd is used
        gradient = None
    else:
        # Calculate the gradient
        X1 = np.reshape(X, [1, P, N, 1])
        X2 = np.reshape(X, [P, 1, N, 1])
        C1 = np.reshape(c, [P, 1, 1, K])
        C2 = np.reshape(c, [1, P, 1, K])

        # Sum the terms in the gradient
        PF1 = ((C - np.matmul(c, c.T))/np.matmul(c0, c0.T)).reshape(P, P, 1, 1) 
        PF2 = (np.square(C - np.matmul(c, c.T))/np.square(np.matmul(c0, c0.T))).reshape(P, P, 1, 1)
        Gmni = - PF1 * C1 * X1
        Gmni += - PF1 * C2 * X2
        Gmni +=  PF2 * c0.reshape(P, 1, 1, 1) * C2 * X1
        Gmni += PF2 * (c0.T).reshape(1, P, 1, 1) * C1 * X2
        gradient = np.sum(Gmni, axis=(0, 1))

    return cost, gradient
def fun_FA(centers, maxK, max_iter, n_repeats, s_all=None, verbose=False, conjugate_gradient=True):
    '''
    Extracts the low rank structure from the data given by centers

    Args:
        centers: 2D array of shape (N, P) where N is the ambient dimension and P is the number of centers
        maxK: Maximum rank to consider
        max_iter: Maximum number of iterations for the solver
        n_repeats: Number of repetitions to find the most stable solution at each iteration of K
        s: (Optional) iterable containing (P, 1) random normal vectors

    Returns:
        norm_coeff: Ratio of center norms before and after optimzation
        norm_coeff_vec: Mean ratio of center norms before and after optimization
        Proj: P-1 basis vectors
        V1_mat: Solution for each value of K
        res_coeff: Cost function after optimization for each K
        res_coeff0: Correlation before optimization
    '''
    N, P = centers.shape
    # Configure the solver
    opts =  {
                'max_iter': max_iter,
                'gtol': 1e-6,
                'xtol': 1e-6,
                'ftol': 1e-8
            }

    # Subtract the global mean
    mean = np.mean(centers.T, axis=0, keepdims=True)
    Xb = centers.T - mean
    xbnorm = np.sqrt(np.square(Xb).sum(axis=1, keepdims=True))

    # Gram-Schmidt into a P-1 dimensional basis
    q, r = qr(Xb.T, mode='economic')
    X = np.matmul(Xb, q[:, 0:P-1])

    # Sore the (P, P-1) dimensional data before extracting the low rank structure
    X0 = X.copy()
    xnorm = np.sqrt(np.square(X0).sum(axis=1, keepdims=True))

    # Calculate the correlations
    C0 = np.matmul(X0, X0.T)/np.matmul(xnorm, xnorm.T)
    res_coeff0 = (np.sum(np.abs(C0)) - P) * 1/(P * (P - 1))

    # Storage for the results
    V1_mat = []
    C0_mat = []
    norm_coeff = []
    norm_coeff_vec = []
    res_coeff = []

    # Compute the optimal low rank structure for rank 1 to maxK
    V1 = None
    for i in range(1, maxK + 1):
        best_stability = 0

        for j in range(1, n_repeats + 1):
            # Sample a random normal vector unless one is supplied
            if s_all is not None and len(s_all) >= i:
                s = s_all[i*j - 1]
            else:
                s = np.random.randn(P, 1)

            # Create initial V. 
            sX = np.matmul(s.T, X)
            if V1 is None:
                V0 = sX
            else:
                V0 = np.concatenate([sX, V1.T], axis=0)
            V0, _ = qr(V0.T, mode='economic') # (P-1, i)

            # Compute the optimal V for this i
            V1tmp, output = CGmanopt(V0, partial(square_corrcoeff_full_cost, grad=False), X, **opts)

            # Compute the cost
            cost_after, _ = square_corrcoeff_full_cost(V1tmp, X, grad=False)

            # Verify that the solution is orthogonal within tolerance
            assert np.linalg.norm(np.matmul(V1tmp.T, V1tmp) - np.identity(i), ord='fro') < 1e-10

            # Extract low rank structure
            X0 = X - np.matmul(np.matmul(X, V1tmp), V1tmp.T)

            # Compute stability of solution
            denom = np.sqrt(np.sum(np.square(X), axis=1))
            stability = min(np.sqrt(np.sum(np.square(X0), axis=1))/denom)

            # Store the solution if it has the best stability
            if stability > best_stability:
                best_stability = stability
                best_V1 = V1tmp
            if n_repeats > 1 and verbose:
                print(j, 'cost=', cost_after, 'stability=', stability)

        # Use the best solution
        V1 = best_V1

        # Extract the low rank structure
        XV1 = np.matmul(X, V1)
        X0 = X - np.matmul(XV1, V1.T)

        # Compute the current (normalized) cost
        xnorm = np.sqrt(np.square(X0).sum(axis=1, keepdims=True))
        C0 = np.matmul(X0, X0.T)/np.matmul(xnorm, xnorm.T)
        current_cost = (np.sum(np.abs(C0)) - P) * 1/(P * (P - 1))
        if verbose:
            print('K=',i,'mean=',current_cost)

        # Store the results
        V1_mat.append(V1)
        C0_mat.append(C0)
        norm_coeff.append((xnorm/xbnorm)[:, 0])
        norm_coeff_vec.append(np.mean(xnorm/xbnorm))
        res_coeff.append(current_cost)
 
        # Break the loop if there's been no reduction in cost for 3 consecutive iterations
        if (
                i > 4 and 
                res_coeff[i-1] > res_coeff[i-2] and
                res_coeff[i-2] > res_coeff[i-3] and
                res_coeff[i-3] > res_coeff[i-4]
           ):
            if verbose:
                print("Optimal K0 found")
            break
    return norm_coeff, norm_coeff_vec, q[:, 0:P-1], V1_mat, res_coeff, res_coeff0

#%% 
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.CIFAR100('~/MyData/cifar100_data', train=True, download=True,
                   transform=transform_train)
test_dataset = datasets.CIFAR100('~/MyData/cifar100_data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)
                   ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(num_classes=100)
model = model.to(device)
model.load_state_dict(torch.load('/Users/eghbalhosseini/MyData/neural_manifolds/cfar100_Alexnet',map_location=torch.device('cpu')))
model = model.eval()
#%% 
sampled_classes = 100
examples_per_class = 50
data = make_manifold_data(train_dataset, sampled_classes, examples_per_class, seed=0)
data = [d.to(device) for d in data]

#%% 
activations = extractor(model, data, layer_types=['Conv2d', 'Linear'])
list(activations.keys())
#%% layer to look at 
layer_id='layer_3_Conv2d'
# 1. project data 
data=activations[layer_id]
X = [d.reshape(d.shape[0], -1).T for d in data] # flatten the data
N = X[0].shape[0]
if N > 5000:
    print("Projecting {}".format(layer_id))
    M = np.random.randn(5000, N)
    M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
    X = [np.matmul(M, d) for d in X]

#%% manifold analysis pipeline 

from scipy.linalg import qr
from functools import partial

from cvxopt import solvers, matrix
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient

# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 1000000
solvers.options['abstol'] = 1e-12
solvers.options['reltol'] = 1e-12
solvers.options['feastol'] = 1e-12

# rename variable to match 
XtotT=X
kappa=0
n_t=300
n_reps=10 
# 
num_manifolds = len(XtotT)
Xori = np.concatenate(XtotT, axis=1) # Shape (N, sum_i P_i)
X_origin = np.mean(Xori, axis=1, keepdims=True)
Xtot0 = [XtotT[i] - X_origin for i in range(num_manifolds)] # center the data 
# 
centers = [np.mean(XtotT[i], axis=1) for i in range(num_manifolds)]
centers = np.stack(centers, axis=1)
center_mean = np.mean(centers, axis=1, keepdims=True)

# Center correlation analysis
UU, SS, VV = np.linalg.svd(centers - center_mean)
total = np.cumsum(np.square(SS)/np.sum(np.square(SS)))
maxK = np.argmax([t if t < 0.95 else 0 for t in total]) + 11
#%% extrat low rank 

