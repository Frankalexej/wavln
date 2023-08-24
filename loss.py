import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLoss: 
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def get_loss(self, y_hat, y, mask): 
        """
        Computes the masked loss given a loss function, predicted values, ground truth values, and a mask tensor.
        """
        b, t, f = y_hat.shape
        mask = mask.unsqueeze(-1).expand((b, t, f)).bool()  # both y_hat and y are of same size
        # Apply mask to the predicted and ground truth values
        y_hat_masked = y_hat.masked_select(mask)
        y_masked = y.masked_select(mask)

        # Calculate the loss using the masked values
        loss = torch.sum(self.loss_fn(y_hat_masked, y_masked)) / torch.sum(mask)
        # loss = loss_fn(y_hat_masked, y_masked)

        return loss


class CombinedLoss: 
    def __init__(self, loss_class, alpha=0.1):
        self.loss_class = loss_class
        self.alpha = alpha
    
    def get_loss(self, y_hat, y, mask, z2): 
        # h2 is second layer output of encoder HMLSTM
        # of shape (B, L)
        reconstruction_loss = self.loss_class.get_loss(y_hat, y, mask)

        b, t, f = z2.shape
        mask = mask.unsqueeze(-1).expand((b, t, f)).bool()  # both y_hat and y are of same size
        # Apply mask to the predicted and ground truth values
        z2_masked = z2.masked_select(mask)

        # Calculate the number of boundaries for each sequence in the batch
        reg_loss = torch.sum(z2_masked) / torch.sum(mask)

        # Compute the regularization term based on the number of boundaries
        regularization_term = self.alpha * reg_loss

        return (reconstruction_loss, regularization_term)