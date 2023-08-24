import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # Hyperparameter to control regularization strength

    def forward(self, predicted_output, target_output, sequence_masks):
        # Compute your existing loss (e.g., reconstruction loss)
        reconstruction_loss = your_existing_loss_function(predicted_output, target_output)

        # Calculate the regularization term based on inferred boundaries
        batch_size, seq_length, _ = predicted_output.shape

        # Compute a mask to identify boundaries (where the data changes)
        diff = torch.abs(predicted_output[:, 1:] - predicted_output[:, :-1])
        is_boundary = torch.any(diff > 0, dim=-1, keepdim=True)

        # Apply sequence masks to handle varying sequence lengths
        is_boundary = is_boundary * sequence_masks.unsqueeze(-1)

        # Calculate the number of boundaries for each sequence in the batch
        num_boundaries = torch.sum(is_boundary, dim=1)

        # Compute the regularization term based on the number of boundaries
        regularization_term = self.alpha * torch.mean(num_boundaries)

        # Combine the reconstruction loss and regularization term
        total_loss = reconstruction_loss + regularization_term

        return total_loss






"""
In this updated loss function:

We calculate the boundary indicators based on changes in the encoder output values. The is_boundary tensor will have a True value at positions where a boundary is inferred.

We apply the sequence masks to zero out the boundary indicators for positions beyond the actual sequence lengths in the batch.

We calculate the number of boundaries for each sequence in the batch and then compute the regularization term based on the average number of boundaries.

This approach allows you to incorporate the inferred boundaries into your custom loss function while handling varying sequence lengths within the batch. Adjust the alpha hyperparameter as needed to control the regularization strength.

"""