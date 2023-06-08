# written by GPT4
import torch
import torch.nn as nn

# Define a toy sequence
sequence = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
sequence_lengths = torch.tensor([3, 2, 1])

# Define the target sequence with padding
target = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])

# Calculate the model output (example output)
output = torch.randn(sequence.size(0), sequence.size(1), 1)

# Create a mask based on the sequence lengths
mask = torch.zeros(sequence.size(0), sequence.size(1), 1)
for i, length in enumerate(sequence_lengths):
    mask[i, :length] = 1

# Calculate the loss using the mask
loss_function = nn.BCEWithLogitsLoss(reduction='none')
loss = loss_function(output, target)
loss = loss * mask
loss = loss.sum() / sequence_lengths.sum().float()

# Backward pass
loss.backward()