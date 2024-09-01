import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the dataset for the XOR operator
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # Input values (A, B)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR outputs

def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

# Define the neural network class for the XOR operator
class XORModel:
    def __init__(self):
        # Initialize weights and biases with random values
        self.W1 = torch.tensor([[7.43929911, 5.68582106], [7.44233704, 5.68641663]], requires_grad=True)  # Weights for hidden layer
        self.b1 = torch.tensor([[-3.40935969, -8.69532299]], requires_grad=True)  # Biases for hidden layer
        self.W2 = torch.tensor([[13.01280117], [-13.79168701]], requires_grad=True)  # Weights for output layer
        self.b2 = torch.tensor([[-6.1043458]], requires_grad=True)  # Bias for output layer

    def forward(self, x):
        h = sigmoid(x @ self.W1 + self.b1)  # First layer
        out = sigmoid(h @ self.W2 + self.b2)  # Second layer
        return out

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.forward(x), y)


def visualize_model_3d(model):
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the decision surface with some transparency
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_tensor = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)
            Z[i, j] = model.forward(input_tensor).detach().numpy()[0, 0]

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.3)  # Adjust alpha for transparency

    # Plot the specific training points with higher visibility
    colorlist = ['blue', 'red', 'green', 'yellow']
    for i, (x_val, y_val) in enumerate(x_train.numpy()):
        z_val = model.forward(torch.tensor([[x_val, y_val]], dtype=torch.float32)).detach().numpy()[0, 0]
        ax.scatter(x_val, y_val, z_val, color=colorlist[i], s=100, label=f'Point ({x_val}, {y_val})', edgecolor='k', alpha=1.0)

    # Set labels and title
    ax.set_xlabel('Input A')
    ax.set_ylabel('Input B')
    ax.set_zlabel('Output')
    ax.set_title('XOR Model Visualization of Training Points')

    # Create a legend and display the plot
    plt.legend(loc='best')
    plt.show()

# Instantiate the model
model = XORModel()

# Use Stochastic Gradient Descent (SGD) for optimization
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=0.1)

# Training loop
for epoch in range(1000):
    loss = model.loss(x_train, y_train)  # Compute the loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Final visualization after training
visualize_model_3d(model)

print("Final weights and biases after training:")
print("W1:", model.W1.detach().numpy())
print("b1:", model.b1.detach().numpy())
print("W2:", model.W2.detach().numpy())
print("b2:", model.b2.detach().numpy())
