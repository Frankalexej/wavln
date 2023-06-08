import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as Func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.autograd import Variable
import torch
from layers import HM_LSTM, DoubleLin
from utils import masked_NLLLoss
import time
from padding import generate_mask_from_padding, mask_it


class HM_Net(Module):
    def __init__(self, a, size_list, dict_size, embed_size):
        super(HM_Net, self).__init__()
        self.dict_size = dict_size
        self.size_list = size_list
        self.drop = nn.Dropout(p=0.5)
        self.embed_in = nn.Embedding(dict_size, embed_size)
        self.HM_LSTM = HM_LSTM(a, embed_size, size_list)
        self.weight = nn.Linear(size_list[0]+size_list[1], 2)
        self.embed_out1 = nn.Linear(size_list[0], dict_size)
        self.embed_out2 = nn.Linear(size_list[1], dict_size)
        self.relu = nn.ReLU()
        # self.logsoftmax = nn.LogSoftmax()
        # self.loss = masked_NLLLoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target, hidden):
        # inputs : batch_size * time_steps
        # mask : batch_size * time_steps

        # NOTE: volatile == True -> no gradient (valid/test)
        # NOTE: normal embed
        emb = self.embed_in(Variable(inputs, volatile=not self.training))  # batch_size * time_steps * embed_size
        emb = self.drop(emb)
        h_1, h_2, z_1, z_2, hidden = self.HM_LSTM(emb, hidden)  # batch_size * time_steps * hidden_size

        # mask = Variable(mask, requires_grad=False)
        # batch_loss = Variable(torch.zeros(batch_size).cuda())

        h_1 = self.drop(h_1)  # batch_size * time_steps * hidden_size
        h_2 = self.drop(h_2)
        h = torch.cat((h_1, h_2), 2)

        # NOTE: g(batch_size * time_steps, 2)
        g = Func.sigmoid(self.weight(h.view(h.size(0)*h.size(1), h.size(2))))
        g_1 = g[:, 0:1]  # batch_size * time_steps, 1
        g_2 = g[:, 1:2]  # batch_size * time_steps, 1

        # g_1.expand(g_1.size(0), self.dict_size) -> copy g_1 for self.dict_size times
        # h_1 viewed as batch_size * time_steps, hidden_size
        h_e1 = g_1.expand(g_1.size(0), self.dict_size)*self.embed_out1(h_1.view(h_1.size(0)*h_1.size(1), h_2.size(2)))
        h_e2 = g_2.expand(g_2.size(0), self.dict_size)*self.embed_out2(h_2.view(h_2.size(0)*h_2.size(1), h_2.size(2)))

        h_e = self.relu(h_e1 + h_e2)  # batch_size*time_steps, hidden_size
        batch_loss = self.loss(h_e, Variable(target))

        return batch_loss, hidden, z_1, z_2

    def init_hidden(self, batch_size):
        h_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        c_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        z_t1 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        h_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        c_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        z_t2 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden


class PhonLearn_Net(Module):
    # NOTE: the "summaries" of subsegments is the bottleneck of the model (I think)
    # NOTE: size_list[2] is not for HM_LSTM, but for the middle dim of DoubleLin
    def __init__(self, a, size_list, in_size, out_size, hid_size):
        # input = (batch_size, time_steps, in_size); in_size = 39
        super(PhonLearn_Net, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.size_list = size_list
        self.drop = nn.Dropout(p=0.5)
        # self.embed_in = nn.Embedding(dict_size, embed_size)
        # temp not use embed_in, directly go to RNN
        self.enc_rnn = HM_LSTM(a, in_size, size_list)
        self.dec_rnn = nn.LSTM(size_list[1], out_size)  # only using h_2, there only size_list[1]
        self.enc_lin = DoubleLin(size_list[1], size_list[2], hid_size)
        self.dec_lin = DoubleLin(hid_size, size_list[2], size_list[1])

    def forward(self, inputs, in_lens, in_mask, hidden):
        # inputs : batch_size * time_steps * in_size

        # NOTE: volatile == True -> no gradient (valid/test)
        # NOTE: normal embed
        # enc_x = Variable(inputs, volatile=not self.training)  # batch_size, time_steps, in_size
        enc_x = inputs  # (B, L, I)
        # emb = self.drop(emb)
        h_1, h_2, z_1, z_2, hidden = self.enc_rnn(enc_x, hidden)  # (B, L, H)
        
        
        # h_1 = self.drop(h_1)  # batch_size, time_steps, hidden_size
        h_2 = mask_it(h_2, in_mask)  # masking at this time is just to clean out the non-zeros, I don't really think they have any use (but just put put)
        h_2 = self.drop(h_2)  # batch_size, time_steps, hidden_size

        hid_r = self.enc_lin(h_2)

        # at present we directly use the layer-2 output (whole seq) as input to decoder. Later should add attention. 
        
        dec_x = self.dec_lin(hid_r)
        
        dec_x = mask_it(dec_x, in_mask)
        
        # no need to calculate dec_x_lens, since this is same as in_lens
        # dec_x_lens = [len(x) for x in dec_x]   # NOTE: not well written here
        
        dec_x_lens = in_lens
        
        # using thing will not change the procedure, just leave it here for a while, people say that it will speed up a bit. 
        dec_x = pack_padded_sequence(dec_x, dec_x_lens, batch_first=True, enforce_sorted=False)

        dec_x, _ = self.dec_rnn(dec_x)
        
        dec_x, dec_x_lens = pad_packed_sequence(dec_x, batch_first=True) # (B, L, H)
        
        return dec_x, (hidden, z_1, z_2)

    def init_hidden(self, batch_size):
        # detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
        c_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
        z_t1 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)
        h_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
        c_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
        z_t2 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden
    
    def encode(self, inputs, in_mask, hidden): 
        # enc_x = Variable(inputs, volatile=not self.training)
        enc_x = inputs  # (B, L, I)
        h_1, h_2, z_1, z_2, hidden = self.enc_rnn(enc_x, hidden)
        h_2 = mask_it(h_2, in_mask)
        h_2 = self.drop(h_2)  # batch_size, time_steps, hidden_size
        hid_r = self.enc_lin(h_2)
        return (h_2, z_1, z_2)

