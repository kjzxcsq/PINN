import torch
import torch.nn as nn
from functorch import make_functional, vmap, jacrev
import numpy as np
from matplotlib import pyplot as plt
device = 'cuda'

class NN(nn.Module):
    def __init__(self, layer_sizes):
        super(NN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.normal_(m.bias, mean=0, std=1)
            self.linears.append(m)

        # Normalization
        X = torch.rand([100000, 1])
        self.mu_X, self.sigma_X = X.mean(), X.std()
        
    def forward(self, x):
        x = (x - self.mu_X) / self.sigma_X
        for linear in self.linears[:-1]:
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x 

x = torch.linspace(0, 1, 100).unsqueeze(-1).to(device)
# x = torch.randn(100, 1).to(device)

layer_sizes = [1] + [100] * 3 + [1]
net = NN(layer_sizes).to(device)

fnet, params = make_functional(net)

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)

def empirical_ntk(fnet_single, params, x1, x2, compute='trace'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False
        
    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

ntk_result = empirical_ntk(fnet_single, params, x, x, 'trace')
print(ntk_result.shape)

# Compute eigenvalues
lambda_K, eigvec_K = np.linalg.eig(ntk_result.detach().cpu().numpy())

# Sort in descresing order
lambda_K = np.sort(np.real(lambda_K))[::-1]

# Visualize the eigenvectors of the NTK
fig, axs= plt.subplots(2, 3, figsize=(12, 6))
X = np.linspace(0, 1, len(x))
axs[0, 0].plot(X, np.real(eigvec_K[:,0]))
axs[0, 1].plot(X, np.real(eigvec_K[:,1]))
axs[0, 2].plot(X, np.real(eigvec_K[:,2]))
axs[1, 0].plot(X, np.real(eigvec_K[:,3]))
axs[1, 1].plot(X, np.real(eigvec_K[:,4]))
axs[1, 2].plot(X, np.real(eigvec_K[:,5]))
plt.show()

# Visualize the eigenvalues of the NTK
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(lambda_K)
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel('index')
ax.set_ylabel(r'$\lambda$') 
plt.legend()
plt.show()