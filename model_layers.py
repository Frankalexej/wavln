import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch.nn import Module, Parameter
import torch.nn as nn
import math

class DoubleLin(Module): 
    def __init__(self, n_in, n_mid, n_out): 
        super(DoubleLin, self).__init__()
        self.lin1 = nn.Linear(n_in, n_mid)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(n_mid, n_out)
        # self.batch_norm = nn.BatchNorm1d(num_features=n_out)
    
    def forward(self, x): 
        x = self.lin2(self.relu(self.lin1(x)))
        return x
    
# class LinearPack(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout_rate=0.5):
#         super(LinearPack, self).__init__()

#         self.linear = nn.Linear(in_dim, out_dim)
#         self.relu = nn.ReLU()
#         # self.relu = nn.LeakyReLU()
#         # self.relu = nn.Tanh()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.linear(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         return x

class LinearPack(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        super(LinearPack, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layernorm = nn.LayerNorm(out_dim)  # Layer normalization
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        # self.relu = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.layernorm(x)  # Apply LayerNorm after linear transformation
        x = self.relu(x)
        # x = self.dropout(x)  # Apply dropout after activation (if using dropout)
        return x

"""
NOTE: 
HM_LSTM is a user-defined, customizable model, making use of the defined HM_LSTMCell
However, how many layers depends on this model definition. 
c-values (cell states) are not returned as output
h-values are returned, as well as z-values (scalars)
Making use of z values, we can deduce the positions of boundaries and infer the intended segments. 
Based on such information, it is viable to grab the "representations" of the intended segments

Question: 
Can we use this model to construct an AE? 

- Need to test whether h_1 and h_2 are of same shape
- If so, this means that h_2, the higher-level RNN still records each timestep, 
    but when not reaching boundary, the outputs are copied from previous steps. 
- On the other hand, using the short segments' representations should suffice, 
    as the information is not lost greatly. 
"""


"""
NOTE: IMPORTANT! 
We will manually define a special token to mean the start of sequence. Hence we can feed that token into the decoder as the first token of the sequence. 
Preferably this token can be all zero or all x. I don't know. Better like a noise or absolute silence. [wait for later invention]
"""