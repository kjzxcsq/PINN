import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import sin, cos, pi 
import time

class FNN(nn.Module):
    def __init__(self, layer_sizes, sigma):
        super(FNN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        
        # self.W = torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma
        # self.W = self.W.cuda()

        # self.B1 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 3)) * sigma)
        # self.B2 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 3)) * sigma)
        
        self.B1 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] // 2)) * sigma)
        self.B2 = nn.Parameter(torch.normal(0, 1, size=(1, layer_sizes[1] - len(self.B1[0]))) * sigma)
        
        self.W = nn.Parameter(torch.rand([1, layer_sizes[1] - 2 * len(self.B1[0])]))

    def forward(self, x):
        # x = torch.cat(( sin(torch.matmul(x, self.W)), 
        #                 cos(torch.matmul(x, self.W))), dim=-1)

        # x = torch.cat(( sin(torch.matmul(x, self.B1)),
        #                 cos(torch.matmul(x, self.B2)),
        #                 torch.matmul(x, self.W)), dim=-1)

        x = torch.cat(( sin(torch.matmul(x, self.B1)),
                        cos(torch.matmul(x, self.B2))), dim=-1)
        
        for linear in self.linears[1:-1]:
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x 

if __name__ == '__main__':
    n = 20
    a = 1
    loss_history = []
    x_test = torch.autograd.Variable(torch.linspace(0, 1, 10000, requires_grad=True).unsqueeze(-1), requires_grad=True)
    y = sin(2 * pi * x_test) + a * sin(n * pi * x_test)
    y_x = 2 * pi * cos(2 * pi * x_test) + a * n * pi * cos(n * pi * x_test)
    y_xx = -4 * pi ** 2 * sin(2 * pi * x_test) - a * n ** 2 * pi ** 2 * sin(n * pi * x_test)

    layer_sizes = [1] + [100] * 3 + [1]
    lr = 1e-3
    epochs = 10000
    sigma = 10
    train_size = 128

    net = FNN(layer_sizes, sigma)
    B_param = []
    linear_param = []
    for name, param in net.named_parameters():
        if name in ['B1', 'B2']:
            B_param += [param]
        else:
            linear_param += [param]

    loss_fn = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam([  {'params': B_param, 'lr': lr*10},
                                    {'params': linear_param, 'lr': lr}])
    

    # x_train = torch.linspace(0, 1, 100, requires_grad=True).unsqueeze(-1)
    # x_train = x_train

    plt.ion()
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 9))
    ax.plot(x_test.detach().cpu().numpy(), y.detach().cpu().numpy(), color='blue', label='true')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax2.plot(x_test.detach().cpu().numpy(), y_x.detach().cpu().numpy(), color='blue', label='true')
    ax2.set_xlabel('x')
    ax2.set_ylabel('dy_x')
    ax2.grid(True)
    ax3.plot(x_test.detach().cpu().numpy(), y_xx.detach().cpu().numpy(), color='blue', label='true')
    ax3.set_xlabel('x')
    ax3.set_ylabel('dy_xx')
    ax3.grid(True)
    start_time = time.time()
    for steps in range(1, epochs+1):
        x_train = torch.autograd.Variable(torch.rand([train_size, 1]), requires_grad=True)
        x_train = x_train
        y_1 = net(torch.tensor([1.0]))
        y_0 = net(torch.tensor([0.0]))
        y_hat = net(x_train)

        dy_x = torch.autograd.grad(y_hat, x_train, grad_outputs=torch.ones(y_hat.shape), create_graph=True)[0]
        dy_xx = torch.autograd.grad(dy_x, x_train, grad_outputs=torch.ones(dy_x.shape), create_graph=True)[0]

        loss_1 = loss_fn(dy_xx, -4 * pi ** 2 * sin(2 * pi * x_train) - a * n ** 2 * pi ** 2 * sin(n * pi * x_train))
        loss_2 = 10*loss_fn(y_1, torch.tensor([0.0]))
        loss_3 = 10*loss_fn(y_0, torch.tensor([0.0]))
        loss = loss_1 + loss_2 + loss_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if steps % 100 == 0:
            end_time = time.time()
            print(f'----------------------------------steps:{steps}----------------------------------')
            print(f'loss_1 : {loss_1.item()}')
            print(f'loss_2 : {loss_2.item()}')
            print(f'loss_3 : {loss_3.item()}')
            print(f'y_1    : {y_1.item()}')
            print(f'y_0    : {y_0.item()}')
            print(f'time   : {end_time-start_time}')
            start_time = end_time
            # print(net.B1[0])

            y_hat = net(x_test)
            dy_x = torch.autograd.grad(y_hat, x_test, grad_outputs=torch.ones(y_hat.shape), create_graph=True)[0]
            dy_xx = torch.autograd.grad(dy_x, x_test, grad_outputs=torch.ones(dy_x.shape), create_graph=True)[0]
            l, = ax.plot(x_test.detach().cpu().numpy(), net(x_test).detach().cpu().numpy(), color='red', label='pred')
            l2, = ax2.plot(x_test.detach().cpu().numpy(), dy_x.detach().cpu().numpy(), color='red', label='pred')
            l3, = ax3.plot(x_test.detach().cpu().numpy(), dy_xx.detach().cpu().numpy(), color='red', label='pred')
            ax.legend()
            ax2.legend()
            ax3.legend()
            plt.pause(0.0000001)
            if steps != epochs:
                l.remove()
                l2.remove()
                l3.remove()
    plt.ioff()
    plt.show()
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

    ax1.hist(net.B1[0].detach().cpu().numpy(), bins=50, density=True, facecolor='blue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('B1')
    ax1.grid(True)

    ax2.hist(net.B2[0].detach().cpu().numpy(), bins=50, density=True, facecolor='blue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('B2')
    ax2.grid(True)

    ax3.hist(net.W[0].detach().cpu().numpy(), bins=50, density=True, facecolor='blue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('W')
    ax3.grid(True)

    # plt.ylim(0, 100)
    plt.yscale('log')
    ax4.plot(np.arange(len(loss_history)), loss_history, color='red', label='loss history')
    ax4.set_xlabel('epochs')
    ax4.legend()
    ax4.grid(True)
    plt.show()