# Oppgave 2 - XOR Model Visualization

This document contains visualizations of the XOR model's decision surface and training points.

## 3D Visualization of XOR Model with randomly selected weights

Below is the 3D visualization of the XOR model after training. The plot shows the decision surface and the training points.

Using the following weights:

### Initial Weights and Biases:

- **Hidden Layer Weights (`W1`)**:
  ```python
    self.W1 = torch.tensor([[5.0, -5.0], [5.0, -5.0]], requires_grad=True)  # Weights for hidden layer
    self.b1 = torch.tensor([[1.5, 1.5]], requires_grad=True)  # Biases for hidden layer
    self.W2 = torch.tensor([[5.0], [5.0]], requires_grad=True)  # Weights for output layer
    self.b2 = torch.tensor([1.5]

![XOR Model Visualization](https://gyazo.com/91d28955c3c3929f49c4c58d14dc24a9)

## Description

- **Training Points**: The red points represent the training data.
- **Decision Surface**: The surface shows the model's output for different input values.

## Analysis

We can see that the model is correct for [0, 0], [0, 1] and [1, 0], however [1, 1] is incorrect, so we need to optimize the model by changing the weights.

### Final Weights and Biases:

- **Hidden Layer Weights (`W1`)**:
  ```python
    self.W1 = torch.tensor([[7.43929911, 5.68582106], [7.44233704, 5.68641663]], requires_grad=True)  # Weights for hidden layer
    self.b1 = torch.tensor([[-3.40935969, -8.69532299]], requires_grad=True)  # Biases for hidden layer
    self.W2 = torch.tensor([[13.01280117], [-13.79168701]], requires_grad=True)  # Weights for output layer
    self.b2 = torch.tensor([[-6.1043458]], requires_grad=True)  # Bias for output layer'

![XOR Model Visualization](https://gyazo.com/b89df8c78fe8539c37f9798ce9b4d7ee)

## Description

- **Training Points**: The colored points represent the training data.
- **Decision Surface**: The surface shows the model's output for different input values.

## Analysis

We can now see that the model is correct for [0, 0], [0, 1], [1, 0] and also [1, 1].
