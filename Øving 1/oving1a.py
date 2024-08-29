import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV, skipping the first line
data = pd.read_csv('length_weight.csv', names=['length', 'weight'], skiprows=1)

x_train = torch.tensor(data['length'].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data['weight'].values, dtype=torch.float32).reshape(-1, 1)

x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
x_train = (x_train - x_mean) / x_std
y_train = (y_train - y_mean) / y_std

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

# Since we normalize our data, we can use a smaller learning rate
optimizer = torch.optim.SGD([model.W, model.b], lr=0.01)

for epoch in range(1000):
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Output model parameters and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train * x_std + x_mean, y_train * y_std + y_mean, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x (length)')
plt.ylabel('y (weight)')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]], dtype=torch.float32)
plt.plot(x * x_std + x_mean, model.f(x).detach() * y_std + y_mean, label='$f(x) = xW+b$')
plt.legend()
plt.show()
