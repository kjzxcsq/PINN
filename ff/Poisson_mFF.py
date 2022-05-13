import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import sin, cos, pi 
import time

class NN_FF(nn.Module):
    def __init__(self, layer_sizes, sigma):
        super(NN_FF, self).__init__()
        self.B1 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * 1)
        self.B2 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma)
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes) - 2):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.normal_(m.bias, mean=0, std=1)
            self.linears.append(m)
        m = nn.Linear(2 * layer_sizes[-2], layer_sizes[-1])
        nn.init.xavier_normal_(m.weight, gain=1)
        nn.init.normal_(m.bias, mean=0, std=1)
        self.linears.append(m)

        # Normalization
        X = torch.rand([100000, 1])
        self.mu_X, self.sigma_X = X.mean(), X.std()
        
    def forward(self, x):
        x = (x - self.mu_X) / self.sigma_X
        X1 = torch.cat(( sin(torch.matmul(x, self.B1)),
                        cos(torch.matmul(x, self.B1))), dim=-1)
        X2 = torch.cat(( sin(torch.matmul(x, self.B2)),
                        cos(torch.matmul(x, self.B2))), dim=-1)
        for linear in self.linears[:-1]:
            X1 = torch.tanh(linear(X1))
            X2 = torch.tanh(linear(X2))
        X = torch.cat((X1, X2), dim=-1)
        X = self.linears[-1](X)
        return X 

if __name__ == '__main__':
    a = 2
    b = 50
    n = 0.1 

    # Exact solution
    def u(x, a, b, n):
        return sin(a * pi * x) + n * sin(b * pi * x)
    
    # Exact PDE residual
    def u_xx(x, a, b, n):
        return -(a*pi)**2 * sin(a * pi * x) - n*(b*pi)**2 * sin(b * pi * x)
    
    # Normalization
    X = torch.rand([100000, 1])
    mu_X, sigma_X = X.mean(), X.std()

    # Test data
    x_test = torch.autograd.Variable(torch.linspace(0, 1, 1000, requires_grad=True).unsqueeze(-1), requires_grad=True).cuda()
    y = u(x_test, a, b, n)
    y_xx = u_xx(x_test, a, b, n)

    # Hyperparameters
    sigma = 10
    lr = 1e-3
    lr_n = 100
    layer_sizes = [1] + [100] * 2 + [1]
    train_size = 128
    epochs = 50000

    # net & loss & optimizer
    net = NN_FF(layer_sizes, sigma).cuda()
    B_param = []
    linear_param = []
    for name, param in net.named_parameters():
        if name in ['B1', 'B2']:
            B_param += [param]
        else:
            linear_param += [param]

    loss_fn = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam([  {'params': B_param, 'lr': lr*lr_n},
                                    {'params': linear_param, 'lr': lr}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    #logger
    loss_res_log = []
    loss_bcs_log = []
    l2_error_log = []

    # Train
    start_time = time.time()
    for steps in range(1, epochs+1):
        net.train()
        # Sample
        # x_train = torch.autograd.Variable(torch.rand([train_size, 1]), requires_grad=True).cuda()
        x_train = torch.autograd.Variable((torch.rand([train_size, 1]) - mu_X) / sigma_X, requires_grad=True).cuda()
        # print(x_train.mean())
        # print(x_train.std())
        y_1 = net(torch.tensor([1.0]).cuda())
        y_0 = net(torch.tensor([0.0]).cuda())
        y_hat = net(x_train)

        dy_x = torch.autograd.grad(y_hat, x_train, grad_outputs=torch.ones(y_hat.shape).cuda(), create_graph=True)[0]
        dy_xx = torch.autograd.grad(dy_x, x_train, grad_outputs=torch.ones(dy_x.shape).cuda(), create_graph=True)[0]

        # Residual loss
        loss_res = loss_fn(dy_xx, u_xx(x_train, a, b, n).cuda())
        # Boundary loss
        loss_bcs = 128*(  loss_fn(y_1, torch.tensor([0.0]).cuda())
                        +loss_fn(y_0, torch.tensor([0.0]).cuda()))
        # Total loss
        loss = loss_res + loss_bcs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0:
            net.eval()
            elapsed = time.time() - start_time
            l2_error = torch.linalg.norm(y - net(x_test), 2) / torch.linalg.norm(y, 2)

            print('Steps: {:5d}, Loss: {:.3e}, Loss_res: {:.3e}, Loss_bcs: {:.3e}, L2_error: {:.3e}, Time: {:.3f}'
                .format(steps, loss.item(), loss_res.item(), loss_bcs.item(), l2_error.item(), elapsed))
            
            loss_bcs_log.append(loss_bcs.item())
            loss_res_log.append(loss_res.item())
            l2_error_log.append(l2_error.item())
            
            start_time = time.time()

        if steps % 1000 == 0:
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
    fig.savefig('./ff/plot/s{:d}_b{:d}_lr{:d}_m'.format(sigma, b, lr_n))
    plt.show()