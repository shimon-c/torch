import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x = self.fc2(x1)
        return x, x1


x = torch.randn(10, 10)
y = torch.randn(10, 10)
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=1e-0)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    output, aux = model(x)
    loss = criterion(output, y)
    loss = loss + (aux ** 2).mean()
    loss.backward()
    optimizer.step()

    print('Epoch {}, loss {}, aux norm {}'.format(
        epoch, loss.item(), aux.norm()))
