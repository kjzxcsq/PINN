# Diffusion equation
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import FNN

def sample(size):
    tx = torch.cat((torch.rand([size, 1]), torch.rand([size, 1]) * 2 + torch.full([size, 1], -1.0)), dim=1)
    x_ic = torch.rand([600, 1]) * 2 + torch.full([600, 1], -1.0)
    tx_ic = torch.cat((torch.full([600, 1], 0.0), x_ic), dim=1)
    tx_bc_l = torch.cat((torch.rand([300, 1]), torch.full([300, 1], -1.0)), dim=1)
    tx_bc_r = torch.cat((torch.rand([300, 1]), torch.full([300, 1],  1.0)), dim=1)
    return tx, x_ic.cuda(), tx_ic.cuda(), tx_bc_l.cuda(), tx_bc_r.cuda()


if __name__ == '__main__':
    size = 10000
    tx, x_ic, tx_ic, tx_bc_l, tx_bc_r = sample(size)
    tx = torch.autograd.Variable(tx, requires_grad=True)
    tx = tx.cuda()
    t = tx[:, 0].unsqueeze(-1)
    x = tx[:, 1].unsqueeze(-1)

    layer_sizes = [2] + [64] * 4 + [1]
    net = FNN(layer_sizes)
    net = net.cuda()

    lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()

    epochs=10000
    for steps in range(epochs):
        y_hat = net(tx)
        dy = torch.autograd.grad(y_hat, tx, grad_outputs=torch.ones(y_hat.shape).cuda(), create_graph=True)[0]
        dy_t = dy[:, 0].unsqueeze(-1)
        dy_x = dy[:, 1].unsqueeze(-1)
        dy_xx = torch.autograd.grad(dy_x, tx, grad_outputs=torch.ones(dy_x.shape).cuda(), create_graph=True)[0][:, 1].unsqueeze(-1)

        loss_1 = loss_fn(dy_t, dy_xx - (torch.e ** (-t)) * (1 - torch.pi ** 2) * torch.sin(torch.pi * x))
        loss_2 = loss_fn(net(tx_ic), torch.sin(torch.pi * x_ic))
        loss_3 = loss_fn(net(tx_bc_l), torch.full([300, 1], 0.0).cuda())
        loss_4 = loss_fn(net(tx_bc_r), torch.full([300, 1], 0.0).cuda())
        loss = loss_1 + loss_2 + loss_3 + loss_4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0:
            print(f'--------------------------------steps:{steps}--------------------------------')
            print(f'loss_1 : {loss_1.item()}')
            print(f'loss_2 : {loss_2.item()}')
            print(f'loss_3 : {loss_3.item()}')
            print(f'loss_4 : {loss_4.item()}')

   
    t = np.arange(0, 1, 0.01)
    x = np.arange(-1, 1, 0.01)
    T, X = np.meshgrid(t, x)
    Y = (np.e ** (-T)) * np.sin(np.pi * X)

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot_surface(T, X, Y, cmap='rainbow', rstride=1, cstride=1)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('y')
    plt.savefig('./pytorch/pde2_true.png')
    plt.close(fig)

    fig = plt.figure()
    ax2 = fig.add_subplot(projection='3d')
    _t = torch.from_numpy(T.reshape(-1, 1))
    _x = torch.from_numpy(X.reshape(-1, 1))
    tx = torch.cat((_t, _x), dim=1).cuda()
    Y_HAT = net(tx.float()).detach().cpu().numpy().reshape(len(x), len(t))
    ax2.plot_surface(T, X, Y_HAT, cmap='rainbow', rstride=1, cstride=1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('y')
    plt.savefig('./pytorch/pde2_pred.png')
    plt.close(fig)