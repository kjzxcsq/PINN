import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from numpy import pi

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,5)
        self.fc2 = nn.Linear(5,1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def f(x):
    y = torch.sin(x)
    return y

def u(x):
    y = torch.sin(x)
    return y

# def loss_fn(net, x_data):
#     return (net(x_train) - u(x_train)).pow(2).sum() #测试拟合函数
    
# =============================================================================
def loss_fn(net, x_data):
    x_b = torch.tensor([[0.],[pi]])
    y_b = torch.tensor([[0.],[0.]])
    y_pred_b = net(x_b)
    y = net(x_data)
    u_x = torch.autograd.grad(y, x_data, create_graph=True,
                              grad_outputs=torch.ones(y.shape))[0]
    u_xx = torch.autograd.grad(u_x, x_data, create_graph=True,
                              grad_outputs=torch.ones(u_x.shape))[0]
    Res = - u_xx - f(x_data) 
    y1 = (y_b - y_pred_b).pow(2).mean()
    y2 = Res.pow(2).mean()
    return y1 + y2
# =============================================================================

x_train = torch.unsqueeze(torch.linspace(0,pi,64),dim=1)
net = Net()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

for i in range(10000):
    x = Variable(x_train, requires_grad=True)
    loss = loss_fn(net, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   
   
    if (i + 1) % 100 == 0:
        print(f'epoch: {i + 1}  loss = {loss.item()}')

x_test = torch.unsqueeze(torch.linspace(0,pi,100),dim=1)
y_test = u(x_test)
y_predict = net(x_test)

plt.figure(1)
plt.plot(x_test, y_test, label='true solution')
plt.plot(x_test, y_predict.detach(), 'bo:', label='fitted/PINN solution')
plt.xlabel('x')
plt.ylabel('solution (x)')
plt.legend()
plt.show()
