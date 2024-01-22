import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch.nn import Module, Parameter
import torch.nn as nn
import math
from utils import hard_sigm, bound


class HM_LSTMCell(Module):
    def __init__(self, bottom_size, hidden_size, top_size, a, last_layer):
        super(HM_LSTMCell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.a = a
        self.last_layer = last_layer    # whether this layer is the last, True/False
        '''
        U_11 means the state transition parameters from layer l (current layer) to layer l
        U_21 means the state transition parameters from layer l+1 (top layer) to layer l
        W_01 means the state transition parameters from layer l-1 (bottom layer) to layer l
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.U_11 = Parameter(torch.rand((4 * self.hidden_size + 1, self.hidden_size), dtype=torch.float, device=self.device))
        if not self.last_layer:
            self.U_21 = Parameter(torch.rand((4 * self.hidden_size + 1, self.top_size), dtype=torch.float, device=self.device))
        self.W_01 = Parameter(torch.rand((4 * self.hidden_size + 1, self.bottom_size), dtype=torch.float, device=self.device))
        self.bias = Parameter(torch.rand((4 * self.hidden_size + 1,), dtype=torch.float, device=self.device))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        # h_bottom.size = bottom_size * batch_size
        s_recur = torch.matmul(self.W_01, h_bottom)
        if not self.last_layer:
            s_topdown_ = torch.matmul(self.U_21, h_top)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = torch.zeros(s_recur.size(), device=self.device, requires_grad=False)
        s_bottomup_ = torch.matmul(self.U_11, h)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_

        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = torch.sigmoid(f_s[0:self.hidden_size, :])  # hidden_size * batch_size
        i = torch.sigmoid(f_s[self.hidden_size:self.hidden_size*2, :])
        o = torch.sigmoid(f_s[self.hidden_size*2:self.hidden_size*3, :])
        g = torch.tanh(f_s[self.hidden_size*3:self.hidden_size*4, :])
        z_hat = hard_sigm(self.a, f_s[self.hidden_size*4:self.hidden_size*4+1, :])

        one = torch.ones(f.size(), device=self.device, requires_grad=False)
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)

        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = z * o * torch.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * torch.tanh(c_new)

        # if z == 1: (FLUSH)
        #     c_new = i * g
        #     h_new = o * Func.tanh(c_new)
        # elif z_bottom == 0: (COPY)
        #     c_new = c
        #     h_new = h
        # else: (UPDATE)
        #     c_new = f * c + i * g
        #     h_new = o * Func.tanh(c_new)

        z_new = bound().forward(z_hat)

        return h_new, c_new, z_new


class HM_LSTM(Module):
    def __init__(self, a, input_size, size_list):
        super(HM_LSTM, self).__init__()
        self.a = a
        self.input_size = input_size
        self.size_list = size_list

        self.cell_1 = HM_LSTMCell(self.input_size, self.size_list[0], self.size_list[1], self.a, False)
        self.cell_2 = HM_LSTMCell(self.size_list[0], self.size_list[1], None, self.a, True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

    def forward(self, inputs, hidden):
        # inputs.size = (batch_size, time steps, embed_size/input_size)
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        if hidden == None:
            device = self.device
            h_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
            c_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
            z_t1 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)
            h_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
            c_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
            z_t2 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = hidden
        z_one = torch.ones((1, batch_size), dtype=torch.float, device=self.device, requires_grad=False)

        h_1 = []
        h_2 = []
        z_1 = []
        z_2 = []
        for t in range(time_steps):
            h_t1, c_t1, z_t1 = self.cell_1(c=c_t1, h_bottom=inputs[:, t, :].t(), h=h_t1, h_top=h_t2, z=z_t1, z_bottom=z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_t2, z_bottom=z_t1)  # 0.01s used
            h_1 += [h_t1.t()]
            h_2 += [h_t2.t()]
            z_1 += [z_t1.t()]
            z_2 += [z_t2.t()]

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return torch.stack(h_1, dim=1), torch.stack(h_2, dim=1), torch.stack(z_1, dim=1), torch.stack(z_2, dim=1), hidden

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
        # self.layernorm = nn.LayerNorm(out_dim)  # Layer normalization
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        # self.relu = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        # x = self.layernorm(x)  # Apply LayerNorm after linear transformation
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after activation (if using dropout)
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