import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import sin, cos, pi 
from functorch import make_functional, vmap, jacrev, hessian
import time
device = 'cuda'

class NN_FF(nn.Module):
    def __init__(self, layer_sizes, sigma, is_fourier_layer_trainable=True):
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
    return hessian(fnet, argnums=1)(params, x).squeeze().unsqueeze(-1) / (0.288675)**2

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

    # D = x1.shape[0]
    # N = len(jac1)
    # result2 = torch.zeros((D, D)).to(device)
    # for k in range(N):
    #     j1 = torch.reshape(jac1[k], (D, -1))
    #     j2 = torch.reshape(jac2[k], (D, -1))
    #     K = torch.matmul(j1, torch.transpose(j2, 0, 1))
    #     result2 = result2 + K

    # assert torch.allclose(result, result2, atol=1e-5)
    return result

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)


if __name__ == '__main__':
    a = 2
    b = 20
    n = 1

    # Exact solution
    def u(x, a, b, n):
        return sin(a * pi * x) + n * sin(b * pi * x)
    
    # Exact PDE residual
    def u_xx(x, a, b, n):
        return -(a*pi)**2 * sin(a * pi * x) - n*(b*pi)**2 * sin(b * pi * x)
    

    # Test data
    x_test = torch.autograd.Variable(torch.linspace(0, 1, 100).unsqueeze(-1)).to(device)
    y = u(x_test, a, b, n)
    y_xx = u_xx(x_test, a, b, n)

    # Hyperparameters
    is_fourier_layer_trainable = True
    is_compute_ntk = True
    sigma = 10
    lr = 1e-3
    lr_n = 10
    layer_sizes = [1] + [100] * 4 + [1]
    train_size = 100
    epochs = 1000

    # net
    net = NN_FF(layer_sizes, sigma, is_fourier_layer_trainable).to(device)
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
    K_uu_log = []
    K_ur_log = []
    K_rr_log = []
    spectrum_log = []

    # Computational domain
    bc1_coords = torch.tensor([[0.0], [0.0]])
    bc2_coords = torch.tensor([[1.0], [1.0]])
    dom_coords = torch.tensor([[0.0], [1.0]])
    
    # Sample
    X_bc1 = bc1_coords[0, 0] * torch.ones(train_size // 2, 1)
    X_bc2 = bc2_coords[1, 0] * torch.ones(train_size // 2, 1)
    X_u = torch.autograd.Variable(torch.vstack((X_bc1, X_bc2)), requires_grad=True).to(device)
    
    X_r = torch.autograd.Variable(torch.linspace(dom_coords[0, 0], 
                                                 dom_coords[1, 0], train_size).unsqueeze(-1), requires_grad=True).to(device)

    # Train
    start_time = time.time()
    for steps in range(1, epochs+1):
        net.train()
        
        # Predict
        y_1 = net(torch.tensor([1.0]).to(device))
        y_0 = net(torch.tensor([0.0]).to(device))
        y_hat = net(X_r)

        dy_x = torch.autograd.grad(y_hat, X_r, grad_outputs=torch.ones(y_hat.shape).to(device), create_graph=True)[0]
        dy_xx = torch.autograd.grad(dy_x, X_r, grad_outputs=torch.ones(dy_x.shape).to(device), create_graph=True)[0]

        # Residual loss
        loss_res = loss_fn(dy_xx, u_xx(X_r, a, b, n).to(device))
        # Boundary loss
        loss_bcs = 100*( loss_fn(y_1, torch.tensor([0.0]).to(device))
                        +loss_fn(y_0, torch.tensor([0.0]).to(device)))
        # nxu = net(X_u)
        # uxu = u(X_u, a, b, n)
        # loss_bcs2 = loss_fn(nxu, uxu)*200
        # loss_bcs2 = 200*loss_fn(net(X_u), u(X_u, a, b, n))
        # assert torch.allclose(loss_bcs, loss_bcs2)
        # Total loss
        loss = loss_res + loss_bcs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0 or steps == 10:
            net.eval()
            elapsed = time.time() - start_time
            l2_error = torch.linalg.norm(y - net(x_test), 2) / torch.linalg.norm(y, 2)

            print('Steps: {:5d},  Loss: {:.3e},  Loss_res: {:.3e},  Loss_bcs: {:.3e},  L2_error: {:.3e},  Time: {:.3f}s'
                .format(steps, loss.item(), loss_res.item(), loss_bcs.item(), l2_error.item(), elapsed))
            
            loss_bcs_log.append(loss_bcs.item())
            loss_res_log.append(loss_res.item())
            l2_error_log.append(l2_error.item())

            if is_compute_ntk:
                fnet, params = make_functional(net)
                # spectrum = empirical_ntk(fnet_single, params, X_r, X_r, 'full').detach().cpu().numpy()

                J_u = compute_jac(net_u, params, X_u)
                J_r = compute_jac(net_r, params, X_r)
                J_spectrum = compute_jac(net_u, params, X_r)
                K_uu_value = compute_ntk(J_u, J_u, 'full').detach().cpu().numpy()
                K_ur_value = compute_ntk(J_u, J_r, 'full').detach().cpu().numpy()
                K_rr_value = compute_ntk(J_r, J_r, 'full').detach().cpu().numpy()
                spectrum = compute_ntk(J_spectrum, J_spectrum, 'full').detach().cpu().numpy()

                K_uu_log.append(K_uu_value)
                K_ur_log.append(K_ur_value)
                K_rr_log.append(K_rr_value)
                spectrum_log.append(spectrum)
            
            start_time = time.time()
            scheduler.step()

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.set_ylim(-2, 2)
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
    # fig.savefig('./Poisson_1D/plot/s{:d}_b{:d}_lr{:d}'.format(sigma, b, lr_n))
    plt.show()
    
    if is_compute_ntk:
        # Create loggers for the eigenvalues of the NTK
        lambda_K_log = []
        K_list = []
        lambda_spectrum_log = []
        # spectrum_list = []
        for k in range(len(K_uu_log)):
            K_uu = K_uu_log[k]
            K_ur = K_ur_log[k]
            K_rr = K_rr_log[k]

            K = np.concatenate([np.concatenate([K_uu, K_ur], axis = 1),
                                np.concatenate([K_ur.T, K_rr], axis = 1)], axis = 0)
            K_list.append(K)

            # Compute eigenvalues
            lambda_K, eigvec_K = np.linalg.eig(K)
            
            # Sort in descresing order
            lambda_K = np.sort(np.real(lambda_K))[::-1]
            
            # Store eigenvalues
            lambda_K_log.append(lambda_K)


            spectrum = spectrum_log[k]
            # spectrum_list.append(spectrum)
            lambda_spectrum, _ = np.linalg.eig(spectrum)
            lambda_spectrum = np.sort(np.real(lambda_spectrum))[::-1]
            lambda_spectrum_log.append(lambda_spectrum)

        # Eigenvalues of NTK
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(lambda_K_log[0], label = 'n=10')
        ax.plot(lambda_K_log[-1], '--', label = 'n={:d}'.format(epochs))
        plt.xscale('log')
        plt.yscale('log')
        ax.set_xlabel('index')
        ax.set_ylabel(r'$\lambda$')
        ax.set_title(r'Eigenvalues of ${K}$')
        ax.legend()
        plt.show()

        # Spectrum
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(lambda_spectrum_log[0], label = 'n=10')
        ax.plot(lambda_spectrum_log[-1], '--', label = 'n={:d}'.format(epochs))
        plt.xscale('log')
        plt.yscale('log')
        ax.set_xlabel('index')
        ax.set_ylabel(r'$\lambda$')
        ax.set_title(r'spectrum')
        ax.legend()
        plt.show()

        # Visualize the eigenvectors of the NTK
        fig, axs= plt.subplots(2, 3, figsize=(12, 6))
        X_u = np.linspace(0, 1, 2*train_size)
        axs[0, 0].plot(X_u, np.real(eigvec_K[:,0]))
        axs[0, 1].plot(X_u, np.real(eigvec_K[:,1]))
        axs[0, 2].plot(X_u, np.real(eigvec_K[:,2]))
        axs[1, 0].plot(X_u, np.real(eigvec_K[:,3]))
        axs[1, 1].plot(X_u, np.real(eigvec_K[:,4]))
        axs[1, 2].plot(X_u, np.real(eigvec_K[:,5]))
        plt.show()

        # Change of the NTK
        NTK_change_list = []
        K0 = K_list[0]
        for K in K_list:
            diff = np.linalg.norm(K - K0) / np.linalg.norm(K0) 
            NTK_change_list.append(diff)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(NTK_change_list)
        ax.set_title('Change of the NTK')
        plt.show()
