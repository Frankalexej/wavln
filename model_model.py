import torch.nn as nn
from torch.nn import Module
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_layers import HM_LSTM, LinearPack
from model_padding import mask_it
from model_attention import ScaledDotProductAttention


class LastElementExtractor(nn.Module): 
    def __init__(self): 
        super(LastElementExtractor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu = torch.device('cpu')
    
    def forward(self, packed, lengths): 
        lengths = torch.tensor(lengths, device=self.device)
        sum_batch_sizes = torch.cat((
            torch.zeros(2, dtype=torch.int64, device=self.device),
            torch.cumsum(packed.batch_sizes, 0).to(self.device)
        ))
        sorted_lengths = lengths[packed.sorted_indices]
        last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0), device=self.device)
        last_seq_items = packed.data[last_seq_idxs]
        last_seq_items = last_seq_items[packed.unsorted_indices]
        return last_seq_items







class Encoder(Module): 
    def __init__(self, a, size_list, in_size, in2_size, in3_size, hid_size):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.in2_size = in2_size
        self.in3_size = in3_size    # this is newly added for lin2
        self.hid_size = hid_size
        self.size_list = size_list
        # temp not use embed_in, directly go to RNN
        self.lin_1 = LinearPack(in_dim=in_size, out_dim=in2_size)
        self.lin_2 = LinearPack(in_dim=in2_size, out_dim=in3_size)
        self.rnn = HM_LSTM(a, in3_size, size_list)  # changed in_size to in2_size, since the actual data size is in2_size, instead of in_size
        # self.lin_2 = LinearPack(in_dim=size_list[1], out_dim=hid_size)

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
        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        h_1, h_2, z_1, z_2, hidden = self.rnn(enc_x, hidden) # (B, L, I3) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)  # it seems that it might affect the outcome, so apply mask here as well
        # hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        # hid_r = mask_it(hid_r, in_mask)
        hid_r = h_2
        return (hid_r, z_1, z_2)
    
    def encode(self, inputs, in_mask, hidden): 
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        h_1, h_2, z_1, z_2, hidden = self.rnn(enc_x, hidden) # (B, L, I3) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)
        # hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        # hid_r = mask_it(hid_r, in_mask)
        hid_r = h_2
        return (hid_r, z_1, z_2)

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

class SimpleEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(SimpleEncoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[1], hidden_size=size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)

        return enc_x

class SimpleDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13   , 64, 16, 3]: similar to encoder, just layer 0 different
        super(SimpleDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[2]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[2]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
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
            dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
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

class PhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(PhxLearner, self).__init__()

        self.encoder = SimpleEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = SimpleDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        
        return dec_out, attn_w
    
    def encode(self, inputs, in_mask): 
        batch_size = inputs.size(0)
        hidden = self.encoder.inits(batch_size=batch_size, device=self.device)
        return self.encoder.encode(inputs, in_mask, hidden)






class RLEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(RLEncoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[0], hidden_size=size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        # self.act = nn.Tanh()
        # self.bn = nn.BatchNorm1d(size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        # enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = inputs

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        # enc_x = self.act(enc_x)
        
        # enc_x = enc_x.permute(0, 2, 1)
        # enc_x = self.bn(enc_x)
        # enc_x = enc_x.permute(0, 2, 1)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        # enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = inputs

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        # enc_x = self.act(enc_x)

        # enc_x = enc_x.permute(0, 2, 1)
        # enc_x = self.bn(enc_x)
        # enc_x = enc_x.permute(0, 2, 1)

        return enc_x

class RALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(RALDecoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[0], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        # dec_in_token = init_in
        dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            # dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_in_token, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
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
        # return outputs, None

class InitRALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(InitRALDecoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[0], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            # dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_in_token, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
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
    
class RleALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(RleALDecoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[0], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token =  self.lin_3(init_in)
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            # dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_in_token, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
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

class LHYEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(LHYEncoder, self).__init__()
        self.lin = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[1], hidden_size=size_list[3], num_layers=num_layers, batch_first=True)

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin(inputs) # (B, L, I2) -> (B, L, I3)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin(inputs) # (B, L, I2) -> (B, L, I3)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        return enc_x

class LHYDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LHYDecoder, self).__init__()
        self.rnn_in_dim = size_list[3]
        self.rnn_out_dim = size_list[1]

        self.rnn = nn.LSTM(self.rnn_in_dim, self.rnn_out_dim, num_layers=num_layers, batch_first=True)
        # self.lin = LinearPack(in_dim=size_list[1], out_dim=size_list[0])
        self.lin = nn.Linear(size_list[1], size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask, init_in, hidden, inputs_lens):
        dec_x = pack_padded_sequence(hid_r, inputs_lens, batch_first=True, enforce_sorted=False)

        dec_x, (hn, cn) = self.rnn(dec_x)  # (B, L, I1) -> (B, L, I2)

        dec_x, _ = pad_packed_sequence(dec_x, batch_first=True)

        dec_x = self.lin(dec_x)
        
        return dec_x
    
class LRLEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(LRLEncoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[1], hidden_size=size_list[2], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.lin_2 = nn.Linear(size_list[2], size_list[3])
        # self.act = nn.Tanh()
        # self.bn = nn.BatchNorm1d(size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        # enc_x = inputs

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        # enc_x = self.act(enc_x)
        
        # enc_x = enc_x.permute(0, 2, 1)
        # enc_x = self.bn(enc_x)
        # enc_x = enc_x.permute(0, 2, 1)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        # enc_x = inputs

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        # enc_x = self.act(enc_x)

        # enc_x = enc_x.permute(0, 2, 1)
        # enc_x = self.bn(enc_x)
        # enc_x = enc_x.permute(0, 2, 1)

        return enc_x

class LRALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LRALDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        # self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        self.lin_3 = nn.Linear(size_list[3], size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length
        # last_hid_r = self.last_element_extractor()

        # dec_in_token = init_in
        # dec_in_token = self.lin_3()
        dec_in_token = torch.zeros((hid_r.size(0), 1, self.size_list[0]), dtype=torch.float, device=hid_r.device, requires_grad=False)

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
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
        # return outputs, None

class LRLDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LRLDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], size_list[3], num_layers=num_layers, batch_first=True)
        self.lin_3 = nn.Linear(size_list[3], size_list[0])
        # self.act = nn.Tanh()
        self.last_element_extractor = LastElementExtractor()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, inputs_lens, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        # dec_in_token = init_in
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])
        packed_hid_r = pack_padded_sequence(hid_r, inputs_lens, batch_first=True, enforce_sorted=False)
        last_hid_r = self.last_element_extractor(packed_hid_r, inputs_lens).unsqueeze(1)
        dec_in_token = self.lin_3(last_hid_r)

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # dec_x = self.lin_2(dec_x)
            # dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            # attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        # attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        # attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
        # return outputs, None


class TwoLinPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(TwoLinPhxLearner, self).__init__()

        self.encoder = LRLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = LRALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    

class TwoLinNoAttentionPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(TwoLinNoAttentionPhxLearner, self).__init__()

        self.encoder = LRLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = LRLDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, input_lens, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)


class SimplerPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(SimplerPhxLearner, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = RALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)

class SimplerPhxLearnerInit(Module):
    # RL + RAL(Init)
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(SimplerPhxLearnerInit, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = InitRALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
class SimplerPhxLearnerle(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(SimplerPhxLearnerle, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.last_element_extractor = LastElementExtractor()
        self.decoder = RleALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        packed_hid_r = pack_padded_sequence(enc_out, input_lens, batch_first=True, enforce_sorted=False)
        last_hid_r = self.last_element_extractor(packed_hid_r, input_lens).unsqueeze(1)
        init_in = last_hid_r
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)



class LHYPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(LHYPhxLearner, self).__init__()

        self.encoder = LHYEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = LHYDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out = self.decoder(enc_out, in_mask, None, None, input_lens)
        
        return dec_out, None
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)




























class PhonLearn_Net(Module):
    # NOTE: the "summaries" of subsegments is the bottleneck of the model (I think)
    # NOTE: size_list[2] is not for HM_LSTM, but for the middle dim of DoubleLin
    def __init__(self, a, size_list, in_size, in2_size, in3_size, hid_size, out_size):
        # input = (batch_size, time_steps, in_size); 
        super(PhonLearn_Net, self).__init__()
        # temp not use embed_in, directly go to RNN
        self.encoder = Encoder(a=a, size_list=size_list, in_size=in_size, in2_size=in2_size, in3_size=in3_size, hid_size=hid_size)
        self.decoder = Decoder(size_list=size_list, in2_size=in3_size, hid_size=hid_size, out_size=out_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')


    def forward(self, inputs, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        enc_hid = self.encoder.inits(batch_size=batch_size, device=self.device)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out, z_1, z_2 = self.encoder(inputs, in_mask, enc_hid)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        
        return dec_out, attn_w, z_2
    
    def encode(self, inputs, in_mask): 
        batch_size = inputs.size(0)
        hidden = self.encoder.inits(batch_size=batch_size, device=self.device)
        return self.encoder.encode(inputs, in_mask, hidden)