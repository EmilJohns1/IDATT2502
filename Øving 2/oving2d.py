import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import os

# Load the MNIST dataset
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model with 784 -> 128 -> 10
class SimpleNNModel(nn.Module):
    def __init__(self):
        super(SimpleNNModel, self).__init__()
        self.linear1 = nn.Linear(784, 128)  # 28x28 input neurons to 128 hidden neurons
        self.linear2 = nn.Linear(128, 10)   # 128 hidden neurons to 10 output neurons

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image into a 1D vector of 784 pixels
        x = torch.relu(self.linear1(x))  # Apply ReLU to hidden layer
        x = self.linear2(x)  # Linear layer to output
        return torch.softmax(x, dim=1)  # Apply softmax to get probabilities

model = SimpleNNModel()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

#Visualize 10 random test images with their predicted labels

images, labels = next(iter(test_loader))  # Get a batch of images and labels
indices = random.sample(range(len(images)), 10)  # Randomly sample 10 indices
random_images = images[indices]
random_labels = labels[indices]

model.eval()
with torch.no_grad():
    outputs = model(random_images)
    _, predicted = torch.max(outputs, 1)

# Plot the images and their predicted labels
fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
for idx in range(10):
    axes[idx].imshow(random_images[idx].squeeze(), cmap='gray')
    axes[idx].set_title(f'Pred: {predicted[idx].item()}')
    axes[idx].axis('off')

plt.show()

# Create a directory to save the weight images
os.makedirs('weights_images', exist_ok=True)

# Visualize and save the weight matrices for the first linear layer
weights = model.linear1.weight.data.numpy()

for i in range(10):  # Visualizing weights connected to 10 different hidden neurons
    weight_image = weights[i].reshape(28, 28)  # Reshape from 784 to 28x28
    plt.imshow(weight_image, cmap='viridis')
    plt.title(f'Weights for Hidden Neuron {i}')
    plt.colorbar()

    # Save the image
    plt.savefig(f'weights_images/hidden_neuron_weight_{i}.png')
    plt.close()  # Close the figure to prevent display