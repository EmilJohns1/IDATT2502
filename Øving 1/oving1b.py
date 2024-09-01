import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV, skipping the first line
data = pd.read_csv('day_length_weight.csv', names=['day', 'length', 'weight'], skiprows=1)

# Convert to tensors
x_train = torch.tensor(data[['weight', 'length']].values, dtype=torch.float32)
y_train = torch.tensor(data['day'].values, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0, 0.0]], requires_grad=True)  # Initialize weights for two features
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W.T + self.b  # x @ self.W.T is matrix multiplication

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# We change the value of 'lr' and the 'epoch' to try to optimize the model
optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)

# For every 10000 steps the learning rate is multiplied by 0.95
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95)

for epoch in range(500000):
    loss = model.loss(x_train, y_train)
    loss.backward()

    # Ensure that the gradients aren't too large
    torch.nn.utils.clip_grad_norm_([model.W, model.b], max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Call on the scheduler to decrease the learning rate
    scheduler.step()

    if epoch % 10000 == 0:  # Print every 10000 epochs
        print(f'Epoch {epoch}: Loss = {loss.item()}')

# Print final model parameters and loss
print("Final W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
print("Final loss:", model.loss(x_train, y_train).item())
print("Final learning rate: ", optimizer)

# Plotting the 3D model
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original data
ax.scatter(data['weight'], data['length'], data['day'], c='r', marker='o', label='Data')

# Create a grid for prediction
weight_range = torch.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 20)
length_range = torch.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 20)
weight_grid, length_grid = torch.meshgrid(weight_range, length_range, indexing='ij')
grid = torch.stack([weight_grid.flatten(), length_grid.flatten()], dim=1)

# Compute model predictions for the grid
predicted_days = model.f(grid).detach().numpy()
predicted_days = predicted_days.reshape(weight_grid.shape)

# Plot prediction surface
ax.plot_surface(weight_grid.numpy(), length_grid.numpy(), predicted_days, color='b', alpha=0.5, rstride=1, cstride=1, label='Model')

# Add axis labels
ax.set_xlabel('Weight')
ax.set_ylabel('Length')
ax.set_zlabel('Day')

# Add legend
ax.legend()
plt.show()