import torch
import torch.nn as nn
from functorch import make_functional, vmap, vjp, jvp, jacrev
from torch import sin, cos, pi, matmul
device = 'cuda'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        self.fc = nn.Linear(21632, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        return x

class NN_FF(nn.Module):
    def __init__(self, layer_sizes, sigma):
        super(NN_FF, self).__init__()
        self.B1 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma)
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.normal_(m.bias, mean=0, std=1)
            self.linears.append(m)

        # Normalization
        X = torch.rand([100000, 1])
        self.mu_X, self.sigma_X = X.mean(), X.std()
        
    def forward(self, x):
        x = (x - self.mu_X) / self.sigma_X
        x = torch.cat(( sin(torch.matmul(x, self.B1)),
                        cos(torch.matmul(x, self.B1))), dim=-1)
        for linear in self.linears[:-1]:
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x 

# class NN_FF(nn.Module):
#     def __init__(self, layer_sizes, sigma):
#         super(NN_FF, self).__init__()
#         # layer_sizes
#         self.layer_sizes = layer_sizes

#         # Initialize Fourier features
#         self.B1 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma)

#         # Initialize NN
#         self.weights = nn.ParameterList()
#         self.biases = nn.ParameterList()
#         for l in range(len(self.layer_sizes) - 1):
#             W = nn.Parameter(torch.rand(layer_sizes[l], layer_sizes[l + 1]))
#             b = nn.Parameter(torch.rand(layer_sizes[l + 1]))
#             nn.init.xavier_normal_(W, gain=1)
#             nn.init.normal_(b, mean=0, std=1)
#             self.weights.append(W)
#             self.biases.append(b)

#         # Normalization
#         X = torch.rand([100000, 1])
#         self.mu_X, self.sigma_X = X.mean(), X.std()
        
#     def forward(self, x):
#         x = (x - self.mu_X) / self.sigma_X
#         x = torch.cat(( sin(matmul(x, self.B1)),
#                         cos(matmul(x, self.B1))), dim=-1)
#         for l in range(len(self.layer_sizes) - 2):
#             W = self.weights[l]
#             b = self.biases[l]
#             x = torch.tanh(matmul(x, W) + b)
#         x = matmul(x, self.weights[-1]) + self.biases[-1]
#         return x 


# x_train = torch.randn(20, 3, 32, 32, device=device)
# x_test = torch.randn(5, 3, 32, 32, device=device)
x_train = torch.randn(20, 1, device=device)
x_test = torch.randn(5, 1, device=device)

layer_sizes = [1] + [100] * 2 + [1]

# net = DNN().to(device)
net = NN_FF(layer_sizes, 1).to(device)
fnet, params = make_functional(net)

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)

def empirical_ntk(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    # result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = torch.stack([torch.einsum('Naf,Maf->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

result = empirical_ntk(fnet_single, params, x_train, x_train)
print(result.shape)