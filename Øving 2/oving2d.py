import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# 1. Load the MNIST dataset
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. Define the model using softmax
class SimpleSoftmaxModel(nn.Module):
    def __init__(self):
        super(SimpleSoftmaxModel, self).__init__()
        self.linear = nn.Linear(784, 128)  # 28x28 input neurons, 128 output neurons
        self.linear2 = nn.Linear(128, 10)  # 128 input neurons translated to our 10 output classes which represent our numbers

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return torch.softmax(x, dim=1)  # Apply softmax


model = SimpleSoftmaxModel()

# 3. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

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

# 4. Evaluate the model
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


# 5. Visualize 10 random test images with their predicted labels
# Select a batch of images from the test set
images, labels = next(iter(test_loader))  # Get a batch of images and labels

# Select 10 random indices from this batch
indices = random.sample(range(len(images)), 10)  # Randomly sample 10 indices

# Get the images and labels corresponding to the random indices
random_images = images[indices]
random_labels = labels[indices]

# Get model predictions
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
