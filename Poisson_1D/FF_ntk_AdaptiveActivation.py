import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import matmul, sin, cos, pi 
from functorch import make_functional, vmap, vjp, jvp, jacrev
import time
device = 'cuda'

class NN_FF(nn.Module):
    def __init__(self, layer_sizes, sigma, n=10, is_fourier_layer_trainable=True):
        super(NN_FF, self).__init__()
        if is_fourier_layer_trainable:
            self.B1 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma)
        else:
            self.B1 = torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma
            self.B1 = self.B1.to(device)

        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.normal_(m.bias, mean=0, std=1)
            self.linears.append(m)

        self.A1 = nn.ParameterList()
        for i in range(1, len(self.linears)):
            a1 = nn.Parameter(torch.rand(layer_sizes[i]))
            # a1 = nn.Parameter(torch.rand(1))
            self.A1.append(a1)
        
        # self.A1 = nn.Parameter(torch.rand(1))
        self.n = n

        # Normalization
        X = torch.rand([100000, 1])
        self.mu_X, self.sigma_X = X.mean(), X.std()
        
    def forward(self, x):
        x = (x - self.mu_X) / self.sigma_X
        x = torch.cat(( sin(torch.matmul(x, self.B1)),
                        cos(torch.matmul(x, self.B1))), dim=-1)
        for linear, a in zip(self.linears[:-1], self.A1):
        # for linear in self.linears[:-1]:
            # x = torch.tanh(linear(x))
            x = torch.tanh(self.n * a * linear(x))
            # x = torch.tanh(self.n * self.A1 * linear(x))
        x = self.linears[-1](x)
        return x 

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

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)


if __name__ == '__main__':
    a = 2
    b = 20
    n = 0.1

    # Exact solution
    def u(x, a, b, n):
        return sin(a * pi * x) + n * sin(b * pi * x)
    
    # Exact PDE residual
    def u_xx(x, a, b, n):
        return -(a*pi)**2 * sin(a * pi * x) - n*(b*pi)**2 * sin(b * pi * x)
    
    # Test data
    x_test = torch.autograd.Variable(torch.linspace(0, 1, 1000).unsqueeze(-1)).to(device)
    y = u(x_test, a, b, n)
    y_xx = u_xx(x_test, a, b, n)

    # Hyperparameters
    is_fourier_layer_trainable = True
    is_compute_ntk = True
    sigma = 10
    activation_n = 10
    lr = 1e-3
    lr_n = 10
    layer_sizes = [1] + [100] * 3 + [1]
    train_size = 100
    epochs = 1000

    # net
    net = NN_FF(layer_sizes, sigma, activation_n, is_fourier_layer_trainable).to(device)
    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())

    # loss
    loss_fn = nn.MSELoss().to(device)

    # optimizer
    if is_fourier_layer_trainable:
        fourier_param = []
        linear_param = []
        for name, param in net.named_parameters():
            if name in ['B1']:
                fourier_param += [param]
            else:
                linear_param += [param]

        optimizer = torch.optim.Adam([  {'params': fourier_param, 'lr': lr*lr_n},
                                        {'params': linear_param, 'lr': lr}])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    #logger
    loss_res_log = []
    loss_bcs_log = []
    l2_error_log = []
    ntk_log = []
    A1_log = []

    # Train
    start_time = time.time()
    for steps in range(1, epochs+1):
        net.train()
        # Sample
        # x_train = torch.autograd.Variable(torch.rand([train_size, 1]), requires_grad=True).to(device)
        x_train = torch.autograd.Variable(torch.linspace(0, 1, train_size).unsqueeze(-1), requires_grad=True).to(device)

        # Predict
        y_1 = net(torch.tensor([1.0]).to(device))
        y_0 = net(torch.tensor([0.0]).to(device))
        y_hat = net(x_train)

        dy_x = torch.autograd.grad(y_hat, x_train, grad_outputs=torch.ones(y_hat.shape).to(device), create_graph=True)[0]
        dy_xx = torch.autograd.grad(dy_x, x_train, grad_outputs=torch.ones(dy_x.shape).to(device), create_graph=True)[0]

        # Residual loss
        loss_res = loss_fn(dy_xx, u_xx(x_train, a, b, n).to(device))
        # Boundary loss
        loss_bcs = 128*(  loss_fn(y_1, torch.tensor([0.0]).to(device))
                        +loss_fn(y_0, torch.tensor([0.0]).to(device)))
        # Total loss
        loss = loss_res + loss_bcs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0:
            net.eval()
            elapsed = time.time() - start_time
            l2_error = torch.linalg.norm(y - net(x_test), 2) / torch.linalg.norm(y, 2)

            print('Steps: {:5d},  Loss: {:.3e},  Loss_res: {:.3e},  Loss_bcs: {:.3e},  L2_error: {:.3e},  Time: {:.3f}s'
                .format(steps, loss.item(), loss_res.item(), loss_bcs.item(), l2_error.item(), elapsed))
            
            loss_bcs_log.append(loss_bcs.item())
            loss_res_log.append(loss_res.item())
            l2_error_log.append(l2_error.item())
            # A1_log.append(net.A1[0].item())
            # print(net.A1[0])
            # print(net.A1[1])

            if is_compute_ntk:
                fnet, params = make_functional(net)
                result_ntk = empirical_ntk(fnet_single, params, x_train, x_train, 'trace')
                ntk_log.append(result_ntk)
            
            start_time = time.time()
            scheduler.step()


    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.set_ylim(-1.5, 1.5)
    ax1.plot(x_test.detach().cpu().numpy(), y.detach().cpu().numpy(), label='true', linewidth=2)
    ax1.plot(x_test.detach().cpu().numpy(), net(x_test).detach().cpu().numpy(), label='pred', linewidth=1, linestyle='--')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax2.plot(x_test.detach().cpu().numpy(), y.detach().cpu().numpy() - net(x_test).detach().cpu().numpy(), linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Point-wise error')
    ax2.grid(True)

    plt.yscale('log')
    ax3.plot(100*np.arange(len(loss_res_log)), loss_res_log, label='$\mathcal{L}_{r}$', linewidth=2)
    ax3.plot(100*np.arange(len(loss_bcs_log)), loss_bcs_log, label='$\mathcal{L}_{b}$', linewidth=2)
    ax3.plot(100*np.arange(len(l2_error_log)), l2_error_log, label=r'$L^2$ error', linewidth=2)
    ax3.set_xlabel('epochs')
    ax3.legend()
    ax3.grid(True)
    # fig.savefig('./Poisson_1D/AdaptiveActivation/s{:d}_b{:d}_lr{:d}_aa_globalwise'.format(sigma, b, lr_n))
    # fig.savefig('./Poisson_1D/AdaptiveActivation/s{:d}_b{:d}_lr{:d}_aa_layerwise'.format(sigma, b, lr_n))
    # fig.savefig('./Poisson_1D/AdaptiveActivation/s{:d}_b{:d}_lr{:d}_aa_neuronwise'.format(sigma, b, lr_n))
    plt.show()
    
    if is_compute_ntk:
        # Create loggers for the eigenvalues of the NTK
        lambda_K_log = []
        for K in ntk_log:
            # Compute eigenvalues
            lambda_K, eigvec_K = np.linalg.eig(K.detach().cpu().numpy())
            
            # Sort in descresing order
            lambda_K = np.sort(np.real(lambda_K))[::-1]
            
            # Store eigenvalues
            lambda_K_log.append(lambda_K)

        # Eigenvalues of NTK
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(lambda_K_log[0], label = 'n=0')
        ax.plot(lambda_K_log[-1], '--', label = 'n=40,000')
        plt.xscale('log')
        plt.yscale('log')
        ax.set_xlabel('index')
        ax.set_ylabel(r'$\lambda_{uu}$')
        ax.set_title(r'Eigenvalues of ${K}_{uu}$')
        ax.legend()
        plt.show()

        # Visualize the eigenvectors of the NTK
        fig, axs= plt.subplots(2, 3, figsize=(12, 6))
        X_u = np.linspace(0, 1, train_size)
        axs[0, 0].plot(X_u, np.real(eigvec_K[:,0]))
        axs[0, 1].plot(X_u, np.real(eigvec_K[:,1]))
        axs[0, 2].plot(X_u, np.real(eigvec_K[:,2]))
        axs[1, 0].plot(X_u, np.real(eigvec_K[:,3]))
        axs[1, 1].plot(X_u, np.real(eigvec_K[:,4]))
        axs[1, 2].plot(X_u, np.real(eigvec_K[:,5]))
        plt.show()

        # Visualize the eigenvalues of the NTK
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(lambda_K_log[0], label=r'$\sigma={}$'.format(sigma))
        plt.xscale('log')
        plt.yscale('log')
        ax.set_xlabel('index')
        ax.set_ylabel(r'$\lambda$') 
        ax.set_title('Spectrum')
        plt.legend()
        plt.show()

        # fig, ax = plt.subplots(figsize=(6, 5))
        # ax.plot(100*np.arange(len(A1_log)), A1_log, label='A1', linewidth=2)
        # ax.set_xlabel('epochs')
        # ax.legend()
        # ax.grid(True)
        # plt.show()