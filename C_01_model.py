import torch.nn as nn
from torch.nn import Module
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import HM_LSTM, LinearPack
from padding import mask_it
import torch.nn.functional as F

class LinearPack(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        super(LinearPack, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        # self.relu = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, q_in, kv_in, qk_out, v_out):
        super(ScaledDotProductAttention, self).__init__()
        self.w_q = nn.Linear(q_in, qk_out)
        self.w_k = nn.Linear(kv_in, qk_out)
        self.w_v = nn.Linear(kv_in, v_out)
        self.d_k = qk_out

    def forward(self, q, k, v, mask=None):
        """
        q: Query tensor of shape (batch_size, num_queries, d_k)
        k: Key tensor of shape (batch_size, num_keys, d_k)
        v: Value tensor of shape (batch_size, num_values, d_v), num_keys = num_values
        mask: Mask tensor of shape (batch_size, num_queries, num_keys)

        Returns: Output tensor of shape (batch_size, num_queries, d_v)
        """
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Step 1: Compute the dot product between queries and keys
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, num_queries, num_keys)

        # Step 2: Scale the attention scores
        attn_scores = attn_scores / (self.d_k ** 0.5)

        # Step 3: Apply the mask (if any)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Compute the softmax of the attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_queries, num_keys)

        # Step 5: Multiply the attention weights with the values
        output = torch.bmm(attn_weights, v)  # (batch_size, num_queries, d_v)

        return output, attn_weights

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