import torch
import torch.nn as nn
import numpy as np

class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128, batch_first=True)  # Use batch_first=True
        self.dense = nn.Linear(128, len(emojis))  # Output size should match number of emojis

        # Initialize hidden and cell states
        self.hidden_state = None
        self.cell_state = None

    def reset(self, batch_size):  # Reset states prior to new input sequence
        self.hidden_state = torch.zeros(1, batch_size, 128)  # Shape: (num_layers, batch_size, hidden_size)
        self.cell_state = torch.zeros(1, batch_size, 128)    # Shape: (num_layers, batch_size, hidden_size)

    def logits(self, x):  # x shape: (batch_size, sequence_length, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[:, -1, :])  # Use only the last time step

    def forward(self, x):  # x shape: (batch_size, sequence_length, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (batch_size, sequence_length, encoding size), y shape: (batch_size, number_of_emojis)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

emojis = {
    'hat': '🎩',
    'rat': '🐀',
    'cat': '🐈',
    'flat': '🏠',
    'matt': '👨',
    'cap': '🧢',
    'son': '👶'
}

# Create lists for emojis and characters
index_to_emoji = list(emojis.values())
index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

char_encodings = []
for i in range(len(index_to_char)):
    encoding = [0] * len(index_to_char)
    encoding[i] = 1
    char_encodings.append(encoding)

""""
We create a list like this which represents our letters
[1., 0., 0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0],  # ' '
[0., 1., 0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0],  # 'h'
"""

emoji_encodings = []
for i in range(len(emojis)):
    encoding = [0] * len(emojis)
    encoding[i] = 1
    emoji_encodings.append(encoding)

""""
We create a similar list like this for our emojis
hat [1., 0., 0., 0., 0., 0., 0.]
rat [0., 1., 0., 0., 0., 0., 0.] etc..
"""

# Convert to numpy arrays for tensor conversion
char_encodings = np.array(char_encodings)
emoji_encodings = np.array(emoji_encodings)

# Create a mapping for characters to their encodings
letters = {letter: char_encodings[i] for i, letter in enumerate(index_to_char)}

# Prepare training data
x_train = np.array([
    [char_encodings[index_to_char.index('h')], char_encodings[index_to_char.index('a')],
     char_encodings[index_to_char.index('t')], char_encodings[index_to_char.index(' ')]],
    [char_encodings[index_to_char.index('r')], char_encodings[index_to_char.index('a')],
     char_encodings[index_to_char.index('t')], char_encodings[index_to_char.index(' ')]],
    [char_encodings[index_to_char.index('c')], char_encodings[index_to_char.index('a')],
     char_encodings[index_to_char.index('t')], char_encodings[index_to_char.index(' ')]],
    [char_encodings[index_to_char.index('f')], char_encodings[index_to_char.index('l')],
     char_encodings[index_to_char.index('a')], char_encodings[index_to_char.index('t')]],
    [char_encodings[index_to_char.index('m')], char_encodings[index_to_char.index('a')],
     char_encodings[index_to_char.index('t')], char_encodings[index_to_char.index('t')]],
    [char_encodings[index_to_char.index('c')], char_encodings[index_to_char.index('a')],
     char_encodings[index_to_char.index('p')], char_encodings[index_to_char.index(' ')]],
    [char_encodings[index_to_char.index('s')], char_encodings[index_to_char.index('o')],
     char_encodings[index_to_char.index('n')], char_encodings[index_to_char.index(' ')]],
])

y_train = np.array([
    emoji_encodings[0],  # 'h a t  ' -> [1., 0., 0., 0., 0., 0., 0.]
    emoji_encodings[1],  # rat [0., 1., 0., 0., 0., 0., 0.] etc..
    emoji_encodings[2],  # cat
    emoji_encodings[3],  # flat
    emoji_encodings[4],  # matt
    emoji_encodings[5],  # cap
    emoji_encodings[6],  # son
])

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float)  # Shape: (7, 4, 13)
y_train = torch.tensor(y_train, dtype=torch.float)  # Shape: (7, 7)

# Instantiate the model
model = LongShortTermMemoryModel(encoding_size=len(char_encodings[0]))

# Define the optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    model.reset(batch_size=x_train.size(0))
    model.train()
    optimizer.zero_grad()
    loss_value = model.loss(x_train, y_train)  # Calculate loss
    loss_value.backward()                       # Backpropagation
    optimizer.step()                           # Update weights

    if epoch % 100 == 99:
        print(f'Epoch {epoch + 1}, Loss: {loss_value.item()}')

# Testing the model with specific words
def test_model(model, test_words):
    model.eval()
    with torch.no_grad():
        for test_word in test_words:
            padded_test = [char_encodings[index_to_char.index(char)] for char in test_word] + [char_encodings[0]] * (4 - len(test_word))
            test_tensor = torch.tensor(np.array([padded_test])).float()  # Convert to a numpy array first
            model.reset(batch_size=test_tensor.size(0))  # Reset states for the test batch
            output = model(test_tensor)
            predicted_index = output.argmax(dim=-1).item()  # Get index of the predicted emoji
            predicted_emoji = index_to_emoji[predicted_index]
            print(f'Input: "{test_word}", Predicted Emoji: {predicted_emoji}')

# Test the model with words that were not in the training data
test_model(model, ["rt", "rats", "cat", "ct"])