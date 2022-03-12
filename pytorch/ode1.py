## 求解: y' = y , y(0) = 1 , 解为:y=e^x
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FNN(nn.Module):
    def __init__(self, layer_sizes):
        super(FNN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = nn.functional.relu(linear(x))
        x = self.linears[-1](x)
        return x


if __name__ == '__main__': 
    x = torch.linspace(0, 2, 1000, requires_grad=True).unsqueeze(-1)
    # x = torch.autograd.Variable(x, requires_grad=True)
    y = torch.exp(x)

    layer_sizes = [1] + [128] * 4 + [1]
    lr = 1e-4
    epochs=10000
    
    fnn = FNN(layer_sizes)
    loss_fn=nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(fnn.parameters(), lr)

    plt.ion()
    for steps in range(epochs):
        y_0 = fnn(torch.zeros(1))
        y_hat = fnn(x)
        dy_x = torch.autograd.functional.hessian(y_hat, x, create_graph=True)[0]
        # dy_x = torch.autograd.grad(y_hat, x, grad_outputs=torch.ones_like(y_hat), create_graph=True)[0]
        loss_1 = loss_fn(y_hat, dy_x)
        loss_2 = loss_fn(y_0, torch.ones(1))
        loss = loss_1 + loss_2
        if steps % 100 == 0:
            plt.cla()
            plt.scatter(x.detach().numpy(),y.detach().numpy())
            plt.plot(x.detach().numpy(), y_hat.detach().numpy(), c='red',lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 10, 'color': 'red'})
            plt.pause(0.01)
            print(f'--------------------------------steps:{steps}--------------------------------')
            print(f'loss_1:{loss_1.item()}')
            print(f'loss_2:{loss_2.item()}')
            print(f'y_0: {y_0}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.ioff()
    plt.show()
    y_1 = fnn(torch.ones(1))
    print(f'y_1:{y_1}')
    y2 = fnn(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), c='red', label='True')
    plt.plot(x.detach().numpy(), y2.detach().numpy(), c='blue', label='Pred')
    plt.legend(loc='best')
 
    plt.show()