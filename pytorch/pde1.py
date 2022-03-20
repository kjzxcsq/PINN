import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm

class FNN(nn.Module):
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
    
    def forward(self, x):
        for linear in self.linears[:-1]:
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x


# 用于生成拟合函数、边界条件、初值条件的t和x
def sample(size):
    tx = torch.cat((torch.rand([size, 1]), torch.full([size, 1], -1) + torch.rand([size, 1]) * 2), dim=1)
    x_init = torch.full([size, 1], -1) + torch.rand([size, 1]) * 2
    tx_ic = torch.cat((torch.full([size, 1], 0), x_init), dim=1)
    tx_bc_l = torch.cat((torch.rand([size, 1]), torch.full([size, 1], -1)), dim=1)
    tx_bc_r = torch.cat((torch.rand([size, 1]), torch.full([size, 1],  1)), dim=1)
    return tx, x_init.cuda(), tx_ic.cuda(), tx_bc_l.cuda(), tx_bc_r.cuda()


if __name__ == '__main__':
    size = 2000
    lr = 1e-4
    _tx, x_init, tx_ic, tx_bc_l, tx_bc_r = sample(size)
    layer_sizes = [2] + [128] * 4 + [1]
    fnn = FNN(layer_sizes)
    fnn = fnn.cuda()

    optimizer = torch.optim.Adam(fnn.parameters(), lr)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()
    epochs=10000

    tx = torch.autograd.Variable(_tx, requires_grad=True)
    tx = tx.cuda()

    for steps in range(epochs):
        u_hat = fnn(tx)
        u_hat = u_hat.cuda()
        du = torch.autograd.grad(u_hat, tx, grad_outputs=torch.ones(u_hat.shape).cuda(), create_graph=True)[0]
        du_t = du[:, 0].unsqueeze(-1)
        du_x = du[:, 1].unsqueeze(-1)
        du_xx = torch.autograd.grad(du_x, tx, grad_outputs=torch.ones(u_hat.shape).cuda(), create_graph=True)[0][:, 1].unsqueeze(-1)

        loss_1 = loss_fn(du_t + u_hat * du_x, (0.01 / torch.pi) * du_xx)
        loss_2 = loss_fn(fnn(tx_ic), -torch.sin(torch.pi * x_init))
        loss_3 = loss_fn(fnn(tx_bc_l), torch.full([size, 1], 0.0).cuda())
        loss_4 = loss_fn(fnn(tx_bc_r), torch.full([size, 1], 0.0).cuda())
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
    
    # 画图比较
    data = scipy.io.loadmat('./pytorch/burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
 
    temp = np.empty((2, 1))
    i = 0
    j = 0
    pred = np.zeros((100, 256))
    for _t in t:
        temp[0] = _t
        for _x in x:
            temp[1] = _x
            ctemp = torch.Tensor(temp.reshape(1, -1)).cuda()
            pred[i][j] = fnn(ctemp).detach().cpu().numpy()
            j = j + 1
            if j == 256:
                j = 0
        i = i + 1
    T, X = np.meshgrid(t, x, indexing='ij')
    pred_surface = np.reshape(pred, (t.shape[0], x.shape[0]))
    Exact_surface = np.reshape(Exact, (t.shape[0], x.shape[0]))
 
    # plot the approximated values
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-1, 1])
    ax.plot_surface(T, X, pred_surface, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.savefig('./pytorch/Preddata.png')
    plt.close(fig)
    # plot the exact solution
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim([-1, 1])
    ax.plot_surface(T, X, Exact_surface, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    plt.savefig('./pytorch/Turedata.png')
    plt.close(fig)
