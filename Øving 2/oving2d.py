import torch
import matplotlib.pyplot as plt
import torchvision
from torch import nn, optim
import numpy as np

# Load observations from the MNIST dataset
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True,
                                         transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())

x_train = mnist_train.data.view(-1, 784).float()
y_train = torch.nn.functional.one_hot(mnist_train.targets, num_classes=10).float()
x_test = mnist_test.data.view(-1, 784).float()
y_test = torch.nn.functional.one_hot(mnist_test.targets, num_classes=10).float()


# Define the neural network class for digit classification
class NumberModel(nn.Module):
    def __init__(self):
        super(NumberModel, self).__init__()
        # Initialize weights and biases
        self.fc1 = nn.Linear(784, 128)  # Input layer: 784 features (28x28) to 128 hidden units
        self.fc2 = nn.Linear(128, 10)  # Output layer: 128 hidden units to 10 classes (digits 0-9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)  # Linear transformation to output
        return torch.softmax(x, dim=1)  # Log softmax to get class probabilities

    def loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)


# Instantiate the model
model = NumberModel()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for i in range(0, len(x_train), 64):  # Mini-batch gradient descent
        images = x_train[i:i + 64]
        labels = y_train[i:i + 64]
        optimizer.zero_grad()
        outputs = model(images)
        loss = model.loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / (len(x_train) // 64)}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i in range(0, len(x_test), 1000):
        images = x_test[i:i + 1000]
        labels = y_test[i:i + 1000]
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, 1)).sum().item()

print(f'Accuracy: {100 * correct / total}%')


# Visualize some predictions
def visualize_predictions(model, num_images=5):
    model.eval()
    images, labels = next(iter(torch.utils.data.DataLoader(mnist_test, batch_size=5)))
    outputs = model(images.view(-1, 784))
    _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}, Pred: {predicted[i].item()}')
        ax.axis('off')
    plt.show()


visualize_predictions(model)

# Print final weights and biases
print("Final weights and biases after training:")
print("fc1 weights:", model.fc1.weight.detach().numpy())
print("fc1 biases:", model.fc1.bias.detach().numpy())
print("fc2 weights:", model.fc2.weight.detach().numpy())
print("fc2 biases:", model.fc2.bias.detach().numpy())
