import torch
import torch.nn as nn
import torchvision

# Load observations from the Fashion MNIST dataset. The observations are divided into a training set and a test set
fashion_mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = fashion_mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((fashion_mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(fashion_mnist_train.targets.shape[0]), fashion_mnist_train.targets] = 1  # Populate output

fashion_mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = fashion_mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((fashion_mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(fashion_mnist_test.targets.shape[0]), fashion_mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.logits = nn.Sequential(
            # First convolution: input (1@28x28) -> output (32@28x28)
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),  # Batch normalization for the 32 layers
            nn.ReLU(),  # ReLU after the convolution
            nn.MaxPool2d(kernel_size=2),  # output (32@14x14)

            # Second convolution: input (32@14x14) -> output (64@14x14)
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64), # Batch normalization for the 64 layers
            nn.ReLU(),  # ReLU after the convolution
            nn.MaxPool2d(kernel_size=2),  # output (64@7x7)

            nn.Flatten(),  # Flatten the output: (64 * 7 * 7)

            # Dense: input (64*7*7) -> output (1x1024)
            nn.Linear(64 * 7 * 7, 1024),
            nn.BatchNorm1d(1024), # Batch normalization for the 1024 layer
            nn.ReLU(),  # ReLU after the dense layer
            nn.Dropout(p=0.5),  # Dropout to regularize the dense layer

            # Dense: input (1x1024) -> output (1x10)
            nn.Linear(1024, 10)
        )

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(10):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test))
