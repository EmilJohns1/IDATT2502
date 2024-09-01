import torch
import matplotlib.pyplot as plt

# Define the dataset for the NAND operator
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # Input values (A, B)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float32)  # NAND outputs

def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

# Define the neural network class for the NAND operator
class NANDModel:
    def __init__(self):
        self.W = torch.tensor([[-1.0], [-1.0]], requires_grad=True)  # Initial weights for 2 input neurons
        self.b = torch.tensor([[1.5]], requires_grad=True)           # Initial bias for the output neuron

    def f(self, x):
        return sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

# Instantiate the model
model = NANDModel()

# Use Stochastic Gradient Descent (SGD) for optimization
optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)

# Training loop
for epoch in range(1000):
    loss = model.loss(x_train, y_train)  # Compute the loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Output final model parameters and loss
print(f"Final W: {model.W.detach().numpy()}, Final b: {model.b.item()}, Final loss: {model.loss(x_train, y_train).item()}")

# Testing the model
with torch.no_grad():
    test_output = model.f(x_train)

# Display the results
print("NAND Operator Truth Table Predictions:")
for inp, out in zip(x_train, test_output):
    print(f"Input: {inp.tolist()}, Predicted Output: {out.item():.3f}")

# Visualize the model predictions
plt.plot(x_train.numpy()[:, 0], y_train.numpy(), 'o', label='Training Data (A, B)')
plt.plot(x_train.numpy()[:, 0], test_output.numpy(), 'x', label='Model Output', color='red')
plt.xlabel('Input A')
plt.ylabel('Output')
plt.legend(loc='lower center')
plt.title('NAND Operator Model Predictions')
plt.show()
