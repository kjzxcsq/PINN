## 求解: y'' + y' = cos(x) - sin(x) , y(0) = 1 , 解为: y = sin(x) + 1 - sin(1)
import torch
from torch import nn
import matplotlib.pyplot as plt

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

if __name__=='__main__':
    x = torch.linspace(1, 3, 1000, requires_grad=True).unsqueeze(-1)
    y = torch.sin(x) + 1 - torch.sin(torch.tensor(1.0))

    layer_sizes = [1] + [16] * 4 + [1]
    lr = 1e-4
    epochs = 10000

    fnn = FNN(layer_sizes)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(fnn.parameters(), lr)

    plt.ion
    for steps in range(epochs):
        y_1 = fnn(torch.tensor([1.0]))
        y_hat = fnn(x)
        dy_x = torch.autograd.grad(y_hat, x, grad_outputs=torch.ones_like(y_hat), create_graph=True)[0]
        dy_xx = torch.autograd.grad(dy_x, x, grad_outputs=torch.ones_like(dy_x), create_graph=True)[0]

        loss_1 = loss_fn(dy_xx + dy_x, torch.cos(x) - torch.sin(x))
        loss_2 = loss_fn(y_1, torch.tensor([1.0]))
        loss = loss_1 + loss_2

        if steps % 100 == 99:
            plt.cla()
            plt.scatter(x.detach().numpy(), y.detach().numpy())
            plt.plot(x.detach().numpy(), y_hat.detach().numpy(), c='red', lw='3')
            plt.text(0.5, 0, 'Loss=%.5f' % loss.item(), fontdict={'size': 10, 'color': 'red'})
            plt.pause(0.00001)

            y_3 = fnn(torch.tensor([3.0]))
            print(f'----------------------------------steps:{steps+1}----------------------------------')
            print(f'loss_1 : {loss_1.item()}')
            print(f'loss_2 : {loss_2.item()}')
            print(f'y_1    : {y_1}')
            print(f'y_3    : {y_3}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.ioff()
    plt.show()
    y2 = fnn(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), c='red', label='True')
    plt.plot(x.detach().numpy(), y2.detach().numpy(), c='blue', label='Pred')
    plt.legend(loc='best')
 
    plt.show()   
        
