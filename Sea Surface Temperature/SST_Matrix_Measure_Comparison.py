import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt 
import numpy.ma as ma

import torch
import torch.nn as nn
import torch.optim as optim

import scipy
from scipy.spatial import cKDTree

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import ot
import time

import imageio
import os
from IPython.display import Image, HTML

############# Import dataset and split into train/test sets
############# The following dataset can be downloaded from "https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2/new/catalog.html"
dataset = nc.Dataset('sst.oisst.mon.mean.1982.nc')
print(dataset.variables.keys())
sst = dataset.variables['sst'][:]
mask = sst.mask[0]
values = sst.data
states =[]
for i in range(len(values)):
    states.append(values[i][~mask])
states = np.array(states)

states_train = states


#Experiment parameters
######################
subset_size = 50
slope = 1   #0.005
steps = 5
epsilon = 1e-4 #teleportation
dt = 0.01
percent = 0.005
sample_size = 400
nstep = 5

############### Form POD for dimension reduction of full state which has dimension ~45,000.
Nmodes = 4 #number of POD modes (this is the state dimension we perform estimation in)

def Snap_POD(DATA,Nmodes):
    S = DATA.T
    Sbar = np.mean(S,axis = 1).reshape(len(S),1)
    X = S - Sbar
    C = X.T.dot(X)
    Lambda,W = np.linalg.eig(C)
    idx = np.argsort(Lambda)[::-1]
    W = W[:,idx]
    Lambda = Lambda[idx]
    W = W[:,:Nmodes]
    Lambdapow = Lambda[:Nmodes]**(-1/2)
    Lambdapow = np.diag(Lambdapow)
    modes = (X.dot(W)).dot(Lambdapow)
    coeffs = modes.T.dot(X)
    return modes, coeffs.T, Sbar

def Reconstruct_POD(modes,coeffs, mean):
    return (modes.dot(coeffs.T) +  mean).T

def Remap(reconstructed):
    full = np.zeros((len(reconstructed),180,360))
    ixs = np.where(mask == False)
    for i in range(len(reconstructed)):
            full[i][ixs[0],ixs[1]] = reconstructed[i]
    return ma.array(full,mask = sst.mask[:len(reconstructed)])

def rollout(net,X):
    for _ in range(nstep):
        X = net(X)*dt + X
    return X
    
def invariant_measure(matrix):
    N = len(matrix)
    rhs = (-1) * (epsilon/N) * torch.ones(N)
    rho = torch.linalg.solve(((1 - epsilon) * matrix - torch.eye(N)),rhs)
    return rho
    
modes, coeffs, mean = Snap_POD(states_train,Nmodes) #we will apply our learning algorithm to "coeffs." No need to touch "modes" or "mean". Note that we still will need to rescale the features of coeffs to feed it into a neural network 
reconstructed = Reconstruct_POD(modes,coeffs,mean) #this is how you reconstruct the full state based on coeffs 
remapped = Remap(reconstructed) #this is how you map back to the "world-map" view for plotting purposes

plt.title('Original')
plt.imshow(sst[235],origin = 'lower')
plt.show()
plt.title('POD Reduced')
plt.imshow(remapped[235],origin = 'lower') 

trajectory = coeffs 

scaler = StandardScaler()
scaled_coeffs = scaler.fit_transform(trajectory)  

# Select training points
randpts = scaled_coeffs[:sample_size]
Trandpts = scaled_coeffs[1:sample_size+1]

# Convert to torch tensors
randpts = torch.tensor(randpts, dtype=torch.float32)
Trandpts = torch.tensor(Trandpts, dtype=torch.float32)

relu = nn.ReLU()

def decay(x):
    return torch.log(1+torch.exp(-x*slope))

def w(xs):
    dists =  torch.cdist(xs, Voronoi_centers, p =2)
    pre_w = decay(dists)
    return  pre_w / torch.sum(pre_w, dim=1,keepdim = True)

def Ulam(points,Tpoints):

    mat = torch.zeros((subset_size, subset_size))
    #before normalization
    with torch.no_grad():
          randpts_idxs = torch.tensor(tree.query(points.detach().numpy())[1], dtype=torch.int)
    weights = w(Tpoints)
    mat.index_add_(0, randpts_idxs, weights)
    row_sums = mat.sum(dim=1)
    normalized_mat = torch.where(row_sums == 0, mat, (mat.T / row_sums))
    return normalized_mat



#### Create cells 
Voronoi_centers = MiniBatchKMeans(n_clusters=subset_size).fit(randpts).cluster_centers_
tree = cKDTree(Voronoi_centers)
Voronoi_centers = torch.tensor(Voronoi_centers,dtype = torch.float)
U_true = Ulam(randpts,Trandpts)
U_true_np = U_true.detach().cpu().numpy()


#### Initialize network
torch.manual_seed(1235)
net1 = nn.Sequential(
            nn.Linear(4, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 4))

optimizer1 = optim.Adam(net1.parameters(),lr = 1e-3)
net1.train()
loss1 = []
N_iters = 20000
net1_randpts = randpts.clone()  # safe copy, avoid modifying original tensor

net1_randpts = rollout(net1,net1_randpts)  # out-of-place update
U_net = Ulam(randpts, net1_randpts)
initial_L1 = torch.linalg.norm(U_net - U_true)


######## Train matrix matching
for i in range(N_iters):
    # Zero gradients for each optimizer
    optimizer1.zero_grad()
    # Update for net1 (Ulam)
    net1_randpts = randpts
    V_field = net1(net1_randpts)
    net1_randpts = rollout(net1,net1_randpts)  
    U_net = Ulam(randpts, net1_randpts)
    L1 = torch.linalg.norm(U_net - U_true)
    L1.backward()
    optimizer1.step()
    loss1.append(L1.item())  # Directly use .item() for scalar value

    if i % 500 == 0:
        print(f'Iteration {i}, Loss(Ulam): {L1.item()}')
        if i % 2000 == 0:
            x1 = randpts[0].clone().detach().reshape(1, 4)
            vals1 = [x1.detach().numpy().flatten()]  # ensure all entries are numpy arrays

            for _ in range(50):
                x1 = rollout(net1,x1)
                vals1.append(x1.detach().numpy().flatten())  # now all elements are uniform shape/type
            vals1 = np.array(vals1)
            fig, ax = plt.subplots(figsize=(10, 2), dpi=300)

            coeff_idx = 0
            ax.plot(randpts[:50,0], ".-", label="Ground Truth", c="black")
            ax.plot(vals1[:50,0], ".-", label="Trajectory Simulation")
             
            ax.legend(loc="best")

            #axes[-1].set_xlabel("Month(t)")

            plt.tight_layout()
            plt.show()
        if L1.item() < initial_L1*percent:#0.05:
            print(f"Early stopping at iteration {i}, Loss: {L1.item()}")
            break
        
        

torch.manual_seed(1235)
net2 = nn.Sequential(
            nn.Linear(4, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 4))

optimizer2 = optim.Adam(net2.parameters(),lr = 1e-3)
net2.train()
loss2 = []
net2_randpts = randpts.clone()  # safe copy, avoid modifying original tensor


im_true = invariant_measure(U_true)
net2_randpts = rollout(net2,net2_randpts)
U_net = Ulam(randpts, net2_randpts)
im_net = invariant_measure(U_net)
initial_L2 =  torch.linalg.norm(im_net - im_true, dtype=torch.float32)

######## Train measure matching
for i in range(N_iters):
    # Zero gradients for each optimizer
    optimizer2.zero_grad()
    net2_randpts = randpts
    net2_randpts = rollout(net2,net2_randpts)
    
    U_net = Ulam(randpts, net2_randpts)
    im_net = invariant_measure(U_net)
    L2 =  torch.linalg.norm(im_net - im_true, dtype=torch.float32)
    L2.backward()
    optimizer2.step()
    loss2.append(L2.item())

    if i % 500 == 0:
        print(f'Iteration {i}, Loss(measure): {L2.item()}')
        if i % 2000 == 0:
            x1 = randpts[0].clone().detach().reshape(1, 4)
            vals1 = [x1.detach().numpy().flatten()]  # ensure all entries are numpy arrays

            for _ in range(50):
                x1 = rollout(net2,x1)
                vals1.append(x1.detach().numpy().flatten())  # now all elements are uniform shape/type
            vals1 = np.array(vals1)
            fig, ax = plt.subplots(figsize=(10, 2), dpi=300)

            coeff_idx = 0
            ax.plot(randpts[:50,0], ".-", label="Ground Truth", c="black")
            ax.plot(vals1[:50,0], ".-", label="Trajectory Simulation")
             
            ax.legend(loc="best")

            #axes[-1].set_xlabel("Month(t)")

            plt.tight_layout()
            plt.show()
        if L2.item() < initial_L2*percent:#0.05:
            print(f"Early stopping at iteration {i}, Loss: {L2.item()}")
            break
        
     
        
     
        
##### Evaluate the performance of the trained networks 
x1 = torch.tensor(scaler.transform(coeffs[sample_size+1].reshape(1,-1)),dtype = torch.float).clone().detach().reshape(1, 4)
vals1 = []
for _ in range(len(coeffs)-1-sample_size):
    vals1.append(x1.detach().numpy().flatten())  # now all elements are uniform shape/type
    x1 = rollout(net1,x1)

predicted_coeffs = scaler.inverse_transform(vals1)  

x2 = torch.tensor(scaler.transform(coeffs[sample_size+1].reshape(1,-1)),dtype = torch.float).clone().detach().reshape(1, 4)
vals2 = []
for _ in range(len(coeffs)-1-sample_size):
    vals2.append(x2.detach().numpy().flatten())  # now all elements are uniform shape/type
    x2 = rollout(net2,x2)


predicted_coeffs2 = scaler.inverse_transform(vals2)  


net1coeffs = scaler.inverse_transform(rollout(net1,torch.tensor(scaler.transform(coeffs[sample_size:]),dtype = torch.float)).detach().numpy())
net2coeffs = scaler.inverse_transform(rollout(net2,torch.tensor(scaler.transform(coeffs[sample_size:]),dtype = torch.float)).detach().numpy())


a, b = np.ones((len(coeffs)-sample_size-1,)) / (len(coeffs)-sample_size-1), np.ones((len(coeffs)-sample_size-1,)) / (len(coeffs)-sample_size-1)
M1, M2 = ot.dist(coeffs[sample_size+1:], predicted_coeffs),ot.dist(coeffs[sample_size+1:], predicted_coeffs2)##
_,G01 = ot.emd(a, b, M1,log = True)
_,G02 = ot.emd(a, b, M2,log = True)
    #W2 ERRORS
cost1W,cost2W = G01['cost']**(1/2),G02['cost']**(1/2)

cost1L = np.mean((net1coeffs[:-1]-coeffs[sample_size+1:])**2)**(1/2)
cost2L = np.mean((net2coeffs[:-1]-coeffs[sample_size+1:])**2)**(1/2)


print('COSTS:', cost1W, cost2W, cost1L, cost2L)


reconstructed1 = Reconstruct_POD(modes,predicted_coeffs,mean) #this is how you reconstruct the full state based on coeffs 
remapped1 = Remap(reconstructed1)
reconstructed2 = Reconstruct_POD(modes,predicted_coeffs2,mean)
remapped2 = Remap(reconstructed2)

recon = Reconstruct_POD(modes,coeffs[sample_size+1:],mean)
remapped = Remap(recon)



ylabels = ["First POD coefficient", "Second POD coefficient", "Third POD coefficient"]

fig, axes = plt.subplots(3, 1, figsize=(10, 6), dpi=300, sharex=True)

for i, ax in enumerate(axes):
    coeff_idx = i
    ax.plot(coeffs[sample_size+1:sample_size+51, coeff_idx], ".-", label="Ground Truth", c="black")
    ax.plot(predicted_coeffs[:50, coeff_idx], ".-", label="Matrix Matching")
    ax.plot(predicted_coeffs2[:50, coeff_idx], ".-", label="Measure Matching")
    ax.set_ylabel(ylabels[i])
    # ax.set_ylim(coeffs[:,coeff_idx].min()-10,coeffs[:,coeff_idx].max()+10)
    if i == 0:
        ax.legend(loc="best")

axes[-1].set_xlabel("Month(t)")

plt.tight_layout()
plt.show()


lon_min, lon_max = -180, 180
lat_min, lat_max = -90, 90
extent = [lon_min, lon_max, lat_min, lat_max]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300,
                         gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1})

# First image
im1 = axes[0].imshow(np.mean(abs(remapped - remapped1), axis=0),
                     origin='lower',  extent=extent,vmin=0, vmax=np.max(np.mean(abs(remapped - remapped1), axis=0)),cmap='plasma')
axes[0].set_title("Markov Matrix Matching")#XXX Matching
axes[0].set_xlabel("Longtitude")
axes[0].set_ylabel("Latitude")

# Second image
im2 = axes[1].imshow(np.mean(abs(remapped - remapped2), axis=0),
                     origin='lower',  extent=extent,vmin=0, vmax=np.max(np.mean(abs(remapped - remapped1), axis=0)),cmap='plasma')
axes[1].set_title("Invariant Measure Matching")
axes[1].set_xlabel("Longtitude")
#axes[1].set_ylabel("Latitude")

# Add an axis for the colorbar beside the last plot
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Mean Absolute Difference")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave room on the right for colorbar
plt.show()

