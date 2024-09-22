import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

        self.hidden_state = None
        self.cell_state = None

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (num_layers, batch_size, hidden_size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


# Character encodings
char_encodings = [
    [1., 0., 0., 0., 0., 0, 0, 0],  # ' '
    [0., 1., 0., 0., 0., 0, 0, 0],  # 'h'
    [0., 0., 1., 0., 0., 0, 0, 0],  # 'e'
    [0., 0., 0., 1., 0., 0, 0, 0],  # 'l'
    [0., 0., 0., 0., 1., 0, 0, 0],  # 'o'
    [0., 0., 0., 0., 0., 1., 0, 0],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0],  # 'r'
    [0., 0., 0., 0., 0., 0., 0, 1.],  # 'd'
]

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

# Training data for "hello world"
x_train = torch.tensor([char_encodings[i] for i in [0, 1, 2, 3, 3, 4, 0, 5, 4, 6, 3, 7, 0]])
y_train = torch.tensor([char_encodings[i] for i in [1, 2, 3, 3, 4, 0, 5, 4, 6, 3, 7, 0, 1]])

encoding_size = len(char_encodings)

# Instantiate the model
model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)

# Training loop
for epoch in range(500):
    model.reset()  # Reset hidden and cell states at the beginning of each epoch
    model.loss(x_train.unsqueeze(1).float(), y_train.float()).backward()  # (sequence_len, batch_size, input_size)
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        # Generate characters starting with the input ' h'
        model.reset()  # Reset state before generating a new sequence
        text = ' h'  # Start with space and 'h'

        # Feed the space ' ' and 'h' into the model
        input_char = torch.tensor([[char_encodings[0]]]).float()  # ' '
        model.f(input_char)
        input_char = torch.tensor([[char_encodings[1]]]).float()  # 'h'
        y = model.f(input_char)

        # Predict and generate the next 50 characters
        for _ in range(50):
            next_char_index = y.argmax(1).item()
            text += index_to_char[next_char_index]

            # Update the input to be the predicted character
            input_char = torch.tensor([[char_encodings[next_char_index]]]).float()
            y = model.f(input_char)

        print(f"Epoch {epoch + 1}: {text}")