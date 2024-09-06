import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV, skipping the first line
data = pd.read_csv('day_head_circumference.csv', names=['day', 'head_circumference'], skiprows=1)

x_train = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data['head_circumference'].values, dtype=torch.float32).reshape(-1, 1)

class NonLinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        # Manually implement the sigmoid function
        linear_combination = x @ self.W + self.b
        sigmoid = 1 / (1 + torch.exp(-linear_combination))
        return 20 * sigmoid + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = NonLinearRegressionModel()

optimizer = torch.optim.Adam([model.W, model.b], lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

for epoch in range(10000):
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    scheduler.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

# Output model parameters and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
print("Final loss:", model.loss(x_train, y_train).item())

# Generate predictions for plotting
x_plot = torch.linspace(0, x_train.max(), 1000).reshape(-1, 1)
with torch.no_grad():
    y_plot = model.f(x_plot).detach()

# Plot the original data points and the predicted curve
plt.scatter(x_train, y_train, label='Original Data', color='blue')
plt.plot(x_plot, y_plot, label='Non-linear fit (Manual Sigmoid)', color='red')
plt.xlabel('Day')
plt.ylabel('Head Circumference')
plt.legend()
plt.show()