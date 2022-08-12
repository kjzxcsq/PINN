import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import sin, cos, exp, sum, pi
from functorch import make_functional, vmap, hessian, jacrev
import time
device = 'cuda'

class NN_FF(nn.Module):
    def __init__(self, layer_sizes, sigma, scaling_factor=10, is_fourier_layer_trainable=True, adaptive_activation='L-LAAF'):
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
        
        # Adaptive activation
        if adaptive_activation == 'GAAF':
            self.A1 = nn.Parameter(torch.rand(1))

        elif adaptive_activation == 'L-LAAF':
            self.A1 = nn.ParameterList()
            for i in range(1, len(self.linears)):
                a1 = nn.Parameter(torch.rand(1))
                self.A1.append(a1)

        elif adaptive_activation == 'N-LAAF':
            self.A1 = nn.ParameterList()
            for i in range(1, len(self.linears)):
                a1 = nn.Parameter(torch.rand(layer_sizes[i]))
                self.A1.append(a1)

        elif adaptive_activation != 'NONE':
            assert False

        # Scaling factor
        self.scaling_factor = scaling_factor

        # Normalization
        X = torch.rand([100000, 1])*6-3
        self.mu_X, self.sigma_X = X.mean(), X.std()
        
    def forward(self, x):
        # Normalization
        x = (x - self.mu_X) / self.sigma_X

        # Fourier layer
        x = torch.cat(( sin(torch.matmul(x, self.B1)),
                        cos(torch.matmul(x, self.B1))), dim=-1)

        # Adaptive activation
        if adaptive_activation == 'GAAF':
            for linear in self.linears[:-1]:
                x = torch.tanh(self.scaling_factor * self.A1 * linear(x))

        elif adaptive_activation in ['L-LAAF', 'N-LAAF']:
            for linear, a in zip(self.linears[:-1], self.A1):
                x = torch.tanh(self.scaling_factor * a * linear(x))

        elif adaptive_activation == 'NONE':
            for linear in self.linears[:-1]:
                x = torch.tanh(linear(x))

        x = self.linears[-1](x)
        return x 

def compute_ntk(jac1, jac2, compute='full'):
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
    result = result.sum(0).squeeze()
    return result

def compute_jac(func, params, x):
    jac = vmap(jacrev(func), (None, 0))(params, x)
    jac = [j.flatten(2) for j in jac]
    return jac

def net_u(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)

def net_r(params, x):
    return hessian(fnet, argnums=1)(params, x).squeeze().unsqueeze(-1)

def empirical_ntk(fnet_single, params, x1, x2, compute='full'):
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
    result = result.sum(0).squeeze()
    return result

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)


if __name__ == '__main__':
    # Hyperparameters
    # adaptive_activation = 'GAAF'
    # adaptive_activation = 'L-LAAF'
    adaptive_activation = 'N-LAAF'
    # adaptive_activation = 'NONE'
    is_fourier_layer_trainable = True
    is_compute_ntk = True
    sigma = 1
    scaling_factor = 10
    lr = 2e-4
    lr_n = 10
    layer_sizes = [1] + [50] + [50] * 3 + [1]
    train_size = 300
    epochs = 15000

    # Exact solution
    def u(x):
        return 0.2*sin(6*x) if x <= 0 else 1+0.1*x*cos(18*x)
        
    # Training data
    x_train = (torch.linspace(-3, 3, train_size)).unsqueeze(-1).to(device)
    y_train = torch.tensor([u(x) for x in x_train]).unsqueeze(-1).to(device)

    # Testing data
    x_test = (torch.linspace(-3, 3, 1000).unsqueeze(-1)).to(device)
    y_test = torch.tensor([u(x) for x in x_test]).unsqueeze(-1).to(device)

    # net
    net = NN_FF(layer_sizes, sigma, scaling_factor, is_fourier_layer_trainable, adaptive_activation).to(device)
    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())

    # Loss
    loss_fn = nn.MSELoss().to(device)

    # Optimizer
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
    
    # Logger
    loss_res_log = []
    l2_error_log = []
    K_log = []
    A1_log = []

    # Train
    start_time = time.time()
    for steps in range(1, epochs+1):
        net.train()

        # Predict
        y_hat = net(x_train)

        # Residual loss
        loss_res = loss_fn(y_hat, y_train)

        # Slope recovery term
        loss_s = 0
        if adaptive_activation == 'L-LAAF':
            for a in net.A1:
                loss_s += exp(a)
            loss_s = len(net.A1) / loss_s

        elif adaptive_activation == 'N-LAAF':
            for a in net.A1:
                loss_s += exp(torch.mean(a))
            loss_s = len(net.A1) / loss_s

        # Total loss
        loss = loss_res + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0 or steps == 10:
            net.eval()
            elapsed = time.time() - start_time
            l2_error = torch.linalg.norm(y_test - net(x_test), 2) / torch.linalg.norm(y_test, 2)

            print('Steps: {:5d},  Loss: {:.3e},  Loss_res: {:.3e},  L2_error: {:.3e},  Time: {:.3f}s'
                .format(steps, loss.item(), loss_res.item(), l2_error.item(), elapsed))
            
            loss_res_log.append(loss_res.item())
            l2_error_log.append(l2_error.item())

            if is_compute_ntk:
                fnet, params = make_functional(net)
                # spectrum = empirical_ntk(fnet_single, params, x_train, x_train, 'trace')

                J_r = compute_jac(net_u, params, x_train)
                K_value = compute_ntk(J_r, J_r, 'full').detach().cpu().numpy()
                K_log.append(K_value)
            
            start_time = time.time()
            scheduler.step()

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.set_ylim(-1.5, 1.5)
    ax1.plot(x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), label='true', linewidth=2)
    ax1.plot(x_test.detach().cpu().numpy(), net(x_test).detach().cpu().numpy(), label='pred', linewidth=1, linestyle='--')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy() - net(x_test).detach().cpu().numpy(), linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Point-wise error')
    ax2.grid(True)

    plt.yscale('log')
    ax3.plot(100*np.arange(len(loss_res_log)), loss_res_log, label='$\mathcal{L}_{r}$', linewidth=2)
    ax3.plot(100*np.arange(len(l2_error_log)), l2_error_log, label=r'$L^2$ error', linewidth=2)
    ax3.set_xlabel('epochs')
    ax3.legend()
    ax3.grid(True)
    # fig.savefig('./NonlinearDiscontinuous/plot/s{:d}_lr{:d}_{:s}'.format(sigma, lr_n, adaptive_activation))
    plt.show()
    
    if is_compute_ntk:
        # Create loggers for the eigenvalues of the NTK
        lambda_K_log = []
        for K in K_log:
            # Compute eigenvalues
            lambda_K, eigvec_K = np.linalg.eig(K)
            
            # Sort in descresing order
            lambda_K = np.sort(np.real(lambda_K))[::-1]
            
            # Store eigenvalues
            lambda_K_log.append(lambda_K)
        

        # Change of the NTK
        NTK_change_list = []
        K0 = K_log[0]
        for K in K_log:
            diff = np.linalg.norm(K - K0) / np.linalg.norm(K0) 
            NTK_change_list.append(diff)

        # Eigenvalues of NTK
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(lambda_K_log[0], label = 'n=10')
        ax1.plot(lambda_K_log[-1], '--', label = f'n={epochs}')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('index')
        ax1.set_ylabel(r'$\lambda$')
        ax1.set_title(r'Eigenvalues of ${K}$')
        ax1.legend()

        ax2.plot(NTK_change_list)
        ax2.set_title('Change of the NTK')
        # fig.savefig('./NonlinearDiscontinuous/plot/s{:d}_lr{:d}_{:s}_NTK'.format(sigma, lr_n, adaptive_activation))
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

        # fig, ax = plt.subplots(figsize=(6, 5))
        # ax.plot(100*np.arange(len(A1_log)), A1_log, label='A1', linewidth=2)
        # ax.set_xlabel('epochs')
        # ax.legend()
        # ax.grid(True)
        # plt.show()
