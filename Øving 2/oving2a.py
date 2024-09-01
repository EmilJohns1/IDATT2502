import torch
import matplotlib.pyplot as plt

# Define a simple dataset for the NOT operator
x_train = torch.tensor([[0], [1]], dtype=torch.float32)
y_train = torch.tensor([[1], [0]], dtype=torch.float32)

# Define the sigmoid function using PyTorch for compatibility with autograd
def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

# Define the model class for the NOT operator
class NOTModel:
    def __init__(self):
        self.W = torch.tensor([[-1.0]], requires_grad=True)  # Initial weight
        self.b = torch.tensor([[0.5]], requires_grad=True)   # Initial bias

    def f(self, x):
        return sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        # Use PyTorch's binary cross-entropy loss for binary classification tasks
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

# Instantiate the model
model = NOTModel()

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
print(f"Final W: {model.W.item()}, Final b: {model.b.item()}, Final loss: {model.loss(x_train, y_train).item()}")

# Testing the model
with torch.no_grad():
    test_input = torch.tensor([[0], [1]], dtype=torch.float32)
    test_output = model.f(test_input)

# Display the results
print(f"Input: 0, Predicted Output: {test_output[0].item():.3f}")
print(f"Input: 1, Predicted Output: {test_output[1].item():.3f}")

# Visualize the model predictions
plt.plot(x_train.numpy(), y_train.numpy(), 'o', label='$(x^{(i)}, y^{(i)})$')
plt.plot(test_input.numpy(), test_output.numpy(), 'x', label='Predicted Output', color='red')
plt.xlabel('x (Input)')
plt.ylabel('y (Output)')
plt.legend()
plt.title('NOT Operator Model Predictions')
plt.show()
