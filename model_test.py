import torch.nn as nn
from torch.nn import Module
import torch
from layers import HM_LSTM, LinearPack
from padding import mask_it
from attention import ScaledDotProductAttention


class Encoder(Module): 
    def __init__(self, a, size_list, in_size, in2_size, hid_size):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.in2_size = in2_size
        self.hid_size = hid_size
        self.size_list = size_list
        # temp not use embed_in, directly go to RNN
        self.lin_1 = LinearPack(in_dim=in_size, out_dim=in2_size)
        self.rnn = HM_LSTM(a, in2_size, size_list)  # changed in_size to in2_size, since the actual data size is in2_size, instead of in_size
        self.lin_2 = LinearPack(in_dim=size_list[1], out_dim=hid_size)

    def inits(self, batch_size, device):
        h_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
        c_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
        z_t1 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)
        h_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
        c_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
        z_t2 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden

    def forward(self, inputs, in_mask, hidden):
        """
        Args:
            inputs: input data (B, L, I)
            in_mask: masking (B, L)
            hidden: HM_LSTM
        """
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        h_1, h_2, z_1, z_2, hidden = self.rnn(enc_x, hidden) # (B, L, I2) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)  # it seems that it might affect the outcome, so apply mask here as well
        hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        hid_r = mask_it(hid_r, in_mask)
        return hid_r
    
    def encode(self, inputs, in_mask, hidden): 
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        h_1, h_2, z_1, z_2, hidden = self.rnn(enc_x, hidden) # (B, L, I2) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)
        hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        hid_r = mask_it(hid_r, in_mask)
        return (hid_r, z_1, z_2)

class EncoderSimple(Module): 
    def __init__(self, a, size_list, in_size, in2_size, hid_size, num_layers=1):
        super(EncoderSimple, self).__init__()
        self.in_size = in_size
        self.in2_size = in2_size
        self.hid_size = hid_size
        self.size_list = size_list
        self.num_layers = num_layers
        # temp not use embed_in, directly go to RNN
        self.lin_1 = LinearPack(in_dim=in_size, out_dim=in2_size)
        # self.rnn = HM_LSTM(a, in2_size, size_list)  # changed in_size to in2_size, since the actual data size is in2_size, instead of in_size
        self.rnn = nn.LSTM(in2_size, size_list[1], num_layers=self.num_layers, batch_first=True)  # only using h_2, therefore only size_list[1]
        self.lin_2 = LinearPack(in_dim=size_list[1], out_dim=hid_size)

    def inits(self, batch_size, device):
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[1]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[1]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        return hidden

    def forward(self, inputs, in_mask, hidden):
        """
        Args:
            inputs: input data (B, L, I)
            in_mask: masking (B, L)
            hidden: HM_LSTM
        """
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        h_2, hidden = self.rnn(enc_x, hidden) # (B, L, I2) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)  # it seems that it might affect the outcome, so apply mask here as well
        hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        hid_r = mask_it(hid_r, in_mask)
        return hid_r
    
    def encode(self, inputs, in_mask, hidden): 
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        h_2, hidden = self.rnn(enc_x, hidden) # (B, L, I2) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)
        hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        hid_r = mask_it(hid_r, in_mask)
        return (hid_r, hidden)
    
class Decoder(Module): 
    def __init__(self, size_list, in2_size, hid_size, out_size, num_layers=1):
        super(Decoder, self).__init__()
        self.in2_size = in2_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.size_list = size_list
        self.num_layers = num_layers

        self.lin_1 = LinearPack(in_dim=out_size, out_dim=size_list[1])  # NOTE: we use out size as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], in2_size, num_layers=self.num_layers, batch_first=True)  # only using h_2, therefore only size_list[1]
        self.attention = ScaledDotProductAttention(q_in=in2_size, kv_in=hid_size, qk_out=in2_size, v_out=in2_size)
        self.lin_2 = LinearPack(in_dim=in2_size, out_dim=out_size)

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.in2_size), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.in2_size), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.out_size), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights


class DecoderSimple(Module): 
    def __init__(self, size_list, in2_size, hid_size, out_size, num_layers=1):
        super(DecoderSimple, self).__init__()
        self.in2_size = in2_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.size_list = size_list
        self.num_layers = num_layers

        # self.lin_1 = LinearPack(in_dim=out_size, out_dim=size_list[1])  # NOTE: we use out size as input size to lin, because we will take the direct output from last layer in dec. 
        self.lin_1 = LinearPack(in_dim=hid_size, out_dim=size_list[1])  # NOTE: testing. 
        self.rnn = nn.LSTM(size_list[1], in2_size, num_layers=self.num_layers, batch_first=True)  # only using h_2, therefore only size_list[1]
        # self.attention = ScaledDotProductAttention(q_in=in2_size, kv_in=hid_size, qk_out=in2_size, v_out=in2_size)
        self.lin_2 = LinearPack(in_dim=in2_size, out_dim=out_size)

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.in2_size), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.in2_size), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.out_size), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        dec_x = self.lin_1(hid_r)
        dec_x, hidden = self.rnn(dec_x, hidden)
        dec_x = self.lin_2(dec_x)

        attention_weights = None

        # outputs = []
        # attention_weights = []
        # for t in range(length):
        #     dec_x = self.lin_1(dec_in_token)
        #     dec_x, hidden = self.rnn(dec_x, hidden)
        #     dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
        #     dec_x = self.lin_2(dec_x)
        #     outputs.append(dec_x)
        #     attention_weights.append(attention_weight)

        #     # Use the current output as the next input token
        #     dec_in_token = dec_x

        # outputs = torch.stack(outputs, dim=1)   # stack along length dim
        # attention_weights = torch.stack(attention_weights, dim=1)
        # outputs = outputs.squeeze(2)
        # attention_weights = attention_weights.squeeze(2)
        outputs = dec_x
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights

class PhonLearn_Net(Module):
    # NOTE: the "summaries" of subsegments is the bottleneck of the model (I think)
    # NOTE: size_list[2] is not for HM_LSTM, but for the middle dim of DoubleLin
    def __init__(self, a, size_list, in_size, in2_size, hid_size, out_size):
        # input = (batch_size, time_steps, in_size); 
        super(PhonLearn_Net, self).__init__()
        # temp not use embed_in, directly go to RNN
        self.encoder = EncoderSimple(a=a, size_list=size_list, in_size=in_size, in2_size=in2_size, hid_size=hid_size)
        self.decoder = DecoderSimple(size_list=size_list, in2_size=in2_size, hid_size=hid_size, out_size=out_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')


    def forward(self, inputs, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        enc_hid = self.encoder.inits(batch_size=batch_size, device=self.device)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)
        enc_out = self.encoder(inputs, in_mask, enc_hid)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        
        return dec_out, attn_w
    
    def encode(self, inputs, in_mask): 
        batch_size = inputs.size(0)
        hidden = self.encoder.inits(batch_size=batch_size, device=self.device)
        return self.encoder.encode(inputs, in_mask, hidden)



class DirectPassModel(Module): 
    # NOTE: this model is to test whether the loss calculation is correct
    def __init__(self, a, size_list, in_size, in2_size, hid_size, out_size):
        # input = (batch_size, time_steps, in_size); 
        super(DirectPassModel, self).__init__()
        self.lin_1 = nn.Linear(in_size, hid_size)
        self.lin_2 = nn.Linear(hid_size, out_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, in_mask):
        hid = self.lin_1(inputs)
        out = self.lin_2(hid)
        return out, in_mask
    
    def encode(self, inputs, in_mask): 
        return inputs


class TwoRNNModel(Module): 
    # NOTE: this model is to test what will happen if I put one RNN in the encoder. This will give something like this: 
    # insize -> hid_size -> outsize
    # NOTE: this model is to test whether the loss calculation is correct
    def __init__(self, a, size_list, in_size, in2_size, hid_size, out_size):
        # input = (batch_size, time_steps, in_size); 
        super(TwoRNNModel, self).__init__()
        self.num_layers = 1
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.enc = nn.LSTM(in_size, hid_size, num_layers=self.num_layers, batch_first=True)  # only using h_2, therefore only size_list[1]
        self.dec = nn.LSTM(hid_size, in_size, num_layers=self.num_layers, batch_first=True)  # only using h_2, therefore only size_list[1]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_enc(self, batch_size, device):
        h0 = torch.zeros((self.num_layers, batch_size, self.hid_size), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.hid_size), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        return hidden
    
    def init_dec(self, batch_size, device):
        h0 = torch.zeros((self.num_layers, batch_size, self.out_size), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.out_size), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        return hidden

    def forward(self, inputs, in_mask):
        batch_size = inputs.size(0)
        enc_hid = self.init_enc(batch_size=batch_size, device=self.device)
        dec_hid = self.init_dec(batch_size=batch_size, device=self.device)
        hid, enc_hid = self.enc(inputs, enc_hid)
        out, dec_hid = self.dec(hid, dec_hid)
        return out, in_mask
    
    def encode(self, inputs, in_mask): 
        return inputs