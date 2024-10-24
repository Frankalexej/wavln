import torch.nn as nn
from torch.nn import Module
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_layers import LinearPack
from model_padding import mask_it
from model_attention import ScaledDotProductAttention
################ New Code Place [since 20240216] #######################################################################
class VAEEncoderV1(Module): 
    """
    VAE V1: Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VAEEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)

    def forward(self, inputs, inputs_lens):
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
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
class VAEDecoderV1(Module): 
    """
    VAE V1: Bidirectional LSTM + Linear; no attention. 
            This is not autoregressive, because we are making it predict each timestep only. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VAEDecoderV1, self).__init__()
        # self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=True)
        # self.attention = ScaledDotProductAttention(q_in=2 * size_list[3], kv_in=2 * size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, lengths): 
        dec_x = hid_r
        dec_x = pack_padded_sequence(dec_x, lengths, batch_first=True, enforce_sorted=False)
        dec_x, (hn, cn) = self.rnn(dec_x)  # (B, L, I1) -> (B, L, I2)
        dec_x, _ = pad_packed_sequence(dec_x, batch_first=True)
        dec_x = self.lin_2(dec_x)
        return dec_x
    
class VAENetV1(Module):
    # RL + RAL(Init) [VAE]
    def __init__(self, enc_size_list, dec_size_list, num_layers=1, dropout=0.5):
        # input = (batch_size, time_steps, in_size); 
        super(VAENetV1, self).__init__()

        self.encoder = VAEEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VAEDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # we take in bi-directional hidrs, and output only 8-dimensional
        self.mu_lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])
        self.logvar_lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, inputs, input_lens):
        enc_out = self.encoder(inputs, input_lens)

        mu = self.mu_lin(enc_out)
        logvar = self.logvar_lin(enc_out)
        z = self.reparameterise(mu, logvar)

        dec_out = self.decoder(z, input_lens)
        return dec_out, (mu, logvar)
    
    def encode(self, inputs, input_lens): 
        enc_out = self.encoder(inputs, input_lens)
        mu = self.mu_lin(enc_out)
        logvar = self.logvar_lin(enc_out)
        return mu, logvar
    

class VAENetV1a(Module):
    # RL + RAL(Init) [non-VAE]
    def __init__(self, enc_size_list, dec_size_list, num_layers=1, dropout=0.5):
        # input = (batch_size, time_steps, in_size); 
        super(VAENetV1a, self).__init__()

        self.encoder = VAEEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VAEDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # we take in bi-directional hidrs, and output only 8-dimensional
        self.lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])
        # self.logvar_lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])

    # def reparameterise(self, mu, logvar):
    #     epsilon = torch.randn_like(mu)
    #     return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, inputs, input_lens):
        enc_out = self.encoder(inputs, input_lens)

        # mu = self.mu_lin(enc_out)
        # logvar = self.logvar_lin(enc_out)
        # z = self.reparameterise(mu, logvar)

        z = self.lin(enc_out)

        dec_out = self.decoder(z, input_lens)
        return dec_out, None
    
    def encode(self, inputs, input_lens): 
        enc_out = self.encoder(inputs, input_lens)
        # mu = self.mu_lin(enc_out)
        # logvar = self.logvar_lin(enc_out)
        z = self.lin(enc_out)
        return z, None
    


################################# VQVAE #################################
class VQEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VQEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])

        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(-1)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x

class VQEncoderV3(Module): 
    """
    20240909
    Linear + Bidirectional LSTM + Linear (merge bidirectional output)
    We use ModuleList to stack LSTM layers and allow outputting all intermediate outputs. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VQEncoderV3, self).__init__()
        # store params to use
        self.hiddim = size_list[3]
        self.num_layers = num_layers

        # layers
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnnlist = nn.ModuleList(
            [nn.LSTM(input_size=size_list[3], hidden_size=size_list[3],
                        batch_first=True, bidirectional=True)]
        )
        for _ in range(1, num_layers): 
            self.rnnlist.append(
                nn.LSTM(input_size=size_list[3] * 2, hidden_size=size_list[3],
                        batch_first=True, bidirectional=True)
            )
        # self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
        #                    num_layers=num_layers, batch_first=True, 
        #                    dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(-1)

    def forward(self, inputs, inputs_lens): 
        b, l, _ = inputs.size()
        d = 2 # bidirectional
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        # initialize h and c
        # hidden_states = [(torch.zeros(d, b, self.hiddim), torch.zeros(d, b, self.hiddim)) for _ in range(self.num_layers)]
        for i, rnn in enumerate(self.rnnlist): 
            enc_x, (hn, cn) = rnn(enc_x)    # (B, L, I1) -> (B, L, I2)
            if i < self.num_layers - 1:  # No need to apply dropout after the last layer
                enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
                enc_x = self.dropout(enc_x)
                enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        # enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    # this is only to be called during inference, so dropout is not applied. 
    def encode_and_out(self, inputs, inputs_lens): 
        b, l, _ = inputs.size()
        d = 2 # bidirectional

        outs = []
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        outs.append(enc_x)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        # initialize h and c
        # hidden_states = [(torch.zeros(d, b, self.hiddim), torch.zeros(d, b, self.hiddim)) for _ in range(self.num_layers)]
        for i, rnn in enumerate(self.rnnlist): 
            enc_x, (hn, cn) = rnn(enc_x)    # (B, L, I1) -> (B, L, I2)
            out_enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            outs.append(out_enc_x)
        # enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x = self.lin_2(out_enc_x)   # this merges the bidirectional into one.
        return enc_x, outs # return the last output and all intermediate outputs.
    
class VQEncoderV5(Module): 
    """
    20240909
    Linear + Bidirectional LSTM + Linear (merge bidirectional output)
    We use ModuleList to stack LSTM layers and allow outputting all intermediate outputs. 

    20240919
    Each LSTM layer is followed by a linear layer. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VQEncoderV5, self).__init__()
        # store params to use
        self.hiddim = size_list[3]
        self.num_layers = num_layers

        # layers
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnnlist = nn.ModuleList(
            [nn.LSTM(input_size=size_list[3], hidden_size=size_list[3],
                        batch_first=True, bidirectional=True)]
        )
        for _ in range(1, num_layers): 
            self.rnnlist.append(
                nn.LSTM(input_size=size_list[3], hidden_size=size_list[3],
                        batch_first=True, bidirectional=True)
                # because we have a linear layer after each LSTM layer so we don't need to double the hidden size. 
            )

        self.linlist = nn.ModuleList(
            [nn.Linear(size_list[3] * 2, size_list[3])] * (num_layers - 1)
        )
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(-1)

    def forward(self, inputs, inputs_lens): 
        b, l, _ = inputs.size()
        d = 2 # bidirectional
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        # initialize h and c
        # hidden_states = [(torch.zeros(d, b, self.hiddim), torch.zeros(d, b, self.hiddim)) for _ in range(self.num_layers)]
        for i, rnn in enumerate(self.rnnlist): 
            enc_x, (hn, cn) = rnn(enc_x)    # (B, L, I1) -> (B, L, I2)
            if i < self.num_layers - 1:  # No need to apply dropout after the last layer
                enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
                enc_x = self.linlist[i](enc_x)
                enc_x = self.dropout(enc_x)
                enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        # enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    # this is only to be called during inference, so dropout is not applied. 
    def encode_and_out(self, inputs, inputs_lens): 
        b, l, _ = inputs.size()
        d = 2 # bidirectional

        outs = []
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        outs.append(enc_x)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        # initialize h and c
        # hidden_states = [(torch.zeros(d, b, self.hiddim), torch.zeros(d, b, self.hiddim)) for _ in range(self.num_layers)]
        for i, rnn in enumerate(self.rnnlist): 
            enc_x, (hn, cn) = rnn(enc_x)    # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            outs.append(enc_x)  # LSTM output
            if i < self.num_layers - 1: 
                enc_x = self.linlist[i](enc_x)
                outs.append(enc_x)  # Linear output
                enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
                # if last LSTM layer, don't pack again
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x, outs # return the last output and all intermediate outputs.
    
class VQDecoderV1(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VQDecoderV1, self).__init__()
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])

        self.act = nn.ReLU()

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

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
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
        return outputs, attention_weights
    
    
class VQDecoderV2(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    同时也输出attention_out
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VQDecoderV2, self).__init__()
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])

        self.act = nn.ReLU()

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

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            # 这里不能detach，因为training中还要做backprop
            outputs.append(dec_x)
            attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attn_outs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            # 我的理解是：虽然如果不clone，list里面的tensor也会跟着变
            # 但是因为这里记录的dec_x在后来都会进入网络中，网络是自带clone的，所以这里不clone也没问题
            attn_outs.append(dec_x.clone())
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x.clone())
            attention_weights.append(attention_weight.clone())
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attn_outs = torch.stack(attn_outs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attn_outs = attn_outs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attn_outs, attention_weights

class VQDecoderV3(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    同时也输出attention_out
    Additionally, we output all intermediate outputs. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VQDecoderV3, self).__init__()
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])

        self.act = nn.ReLU()

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

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            # 这里不能detach，因为training中还要做backprop
            outputs.append(dec_x)
            attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        b, length, _ = hid_r.size()

        dec_in_token = init_in

        outputs = []
        attn_outs = []
        first_lin_outs = []
        attention_weights = []
        rnn_layer_outs = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            first_lin_outs.append(dec_x)  # post-lin
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # add in rnn outs
            rnn_layer_outs.append(hidden[0])    # post-rnn
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            # 我的理解是：虽然如果不clone，list里面的tensor也会跟着变
            # 但是因为这里记录的dec_x在后来都会进入网络中，网络是自带clone的，所以这里不clone也没问题
            attn_outs.append(dec_x.clone())
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x.clone())
            attention_weights.append(attention_weight.clone())
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attn_outs = torch.stack(attn_outs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        first_lin_outs = torch.stack(first_lin_outs, dim=1)
        rnn_layer_outs = torch.stack(rnn_layer_outs, dim=1)
        outputs = outputs.squeeze(2)
        attn_outs = attn_outs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        first_lin_outs = first_lin_outs.squeeze(2)

        rnn_layer_outs = rnn_layer_outs.permute(0, 2, 1, 3)
        rnn_layer_outs = torch.unbind(rnn_layer_outs, 0)
        other_outs = [first_lin_outs] + list(rnn_layer_outs)
        return outputs, attn_outs, attention_weights, other_outs
    
class EncoderV1(Module): 
    """
    Linear * 1 + Unidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(EncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])
        self.rnn = nn.LSTM(input_size=size_list["lin1"], hidden_size=size_list["hid"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=False)
        # self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])
        self.act = nn.ReLU()

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x, (hn, cn)
    
    def encode(self, inputs, inputs_lens): 
        return self.forward(inputs, inputs_lens)
    
class EncoderV2(Module): 
    """
    Linear * 1 + Unidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(EncoderV2, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])

        self.rnns = nn.ModuleList()
        for i in range(num_layers): 
            # TODO: currently we don't support changing dimensions. 
            self.rnns.append(nn.LSTM(input_size=size_list["rnn_in"], hidden_size=size_list["rnn_out"], 
                                     batch_first=True, bidirectional=False))
        # self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for rnn in self.rnns: 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x = self.dropout(enc_x)
        
        return enc_x, (hn, cn)  # so obviously we are returning the last output. 
    
    def encode(self, inputs, inputs_lens): 
        outs = []
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for rnn in self.rnns: 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            outs.append(enc_x)
        return outs, (hn, cn)  # so obviously we are returning the last output. 
    

class EncoderV3(Module): 
    """
    Linear + Bidirectional LSTM + Linear (merge bidirectional output)
    注意！因爲這次是bidirectional，且用ModuleList來層曡LSTM，所以每一層都需要merge. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(EncoderV3, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["rnn_in"])

        self.rnns = nn.ModuleList()
        self.mergers = nn.ModuleList()
        for i in range(num_layers): 
            # TODO: currently we don't support changing dimensions. 
            # NOTE: now rnn_in === rnn_out
            self.rnns.append(nn.LSTM(input_size=size_list["rnn_in"], hidden_size=size_list["rnn_out"], 
                                     batch_first=True, bidirectional=True))
            self.mergers.append(nn.Linear(size_list["rnn_out"] * 2, size_list["rnn_in"]))
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for i in range(len(self.rnns)): 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = self.rnns[i](enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x = self.dropout(enc_x)
            enc_x = self.mergers[i](enc_x)
        return enc_x, (hn, cn)  # so obviously we are returning the last output. 
    
    def encode(self, inputs, inputs_lens): 
        outs = []
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for i in range(len(self.rnns)): 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = self.rnns[i](enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x = self.dropout(enc_x)
            enc_x = self.mergers[i](enc_x)
            outs.append(enc_x)
        return outs, (hn, cn)  # so obviously we are returning the last output. 
    

class GoodDecoderV2(nn.Module):
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(GoodDecoderV2, self).__init__()
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])
        self.rnn = nn.LSTM(input_size=size_list["lin1"], hidden_size=size_list["hid"], 
                            batch_first=True, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin_2 = nn.Linear(size_list["hid"], size_list["in"])
        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list["in"]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, _ = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []
        attention_outs = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, attn_out = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            attention_outs.append(attn_out)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_outs = torch.stack(attention_outs, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        attention_outs = attention_outs.squeeze(2)
        return outputs, attention_weights, attention_outs

    def forward_step(self, input_item, hidden, encoder_outputs, encoder_mask):
        output = self.lin_1(input_item)
        output = self.act(output)
        output, hidden = self.rnn(output, hidden)
        attn_out, attention_weight = self.attention(output, encoder_outputs, 
                                                  encoder_outputs, encoder_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
        output = output + attn_out  # This is mimicking Transformer
        output = self.lin_2(output)
        return output, hidden, attention_weight, attn_out

class DecoderV2(nn.Module):
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(DecoderV2, self).__init__()
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])
        self.rnn = nn.LSTM(input_size=size_list["lin1"], hidden_size=size_list["hid"], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin_2 = nn.Linear(size_list["hid"], size_list["in"])
        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list["in"]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, _ = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []
        attention_outs = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, attn_out = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            attention_outs.append(attn_out)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_outs = torch.stack(attention_outs, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        attention_outs = attention_outs.squeeze(2)
        return outputs, attention_weights, attention_outs

    def forward_step(self, input_item, hidden, encoder_outputs, encoder_mask):
        output = self.lin_1(input_item)
        output = self.act(output)
        output, hidden = self.rnn(output, hidden)
        output, attention_weight = self.attention(output, encoder_outputs, 
                                                  encoder_outputs, encoder_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
        attn_out = output
        output = self.lin_2(output)
        return output, hidden, attention_weight, attn_out
    

class VQVAEV1(Module):
    # RL + RAL(Init)
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(VQVAEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.vq_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        self.vq_embedding.weight.data.uniform_(-1.0 / embedding_dim,
                                               1.0 / embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)

        # now finding the closest vector
        # ze: [B, L, C]
        # embedding: [K, C]
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)
        # stop gradient
        dec_in = ze + (zq - ze).detach()

        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)

        return ze, zq
    

class VQPhonePredV1(Module): 
    # RL + L
    def __init__(self, in_dim, out_dim):
        # input = (batch_size, time_steps, in_size); 
        super(VQPhonePredV1, self).__init__()
        self.decoder = nn.Linear(in_dim, out_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, z):
        pred_out = self.decoder(z)
        pred_out = self.softmax(pred_out)
        return pred_out
    
    def predict_on_output(self, output): 
        preds = torch.argmax(output, dim=-1)
        return preds


############################ Normal AE with same structure as VQVAE ############################
class AEV1(Module):
    # RL + RAL(Init)
    # AEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(AEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq


############################ Normal AE + word-id with same structure as VQVAE ############################
class WIDAEV1(Module):
    # RL + RAL(Init)
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(WIDAEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.word_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        self.combine_layer = nn.Linear(2 * enc_size_list[3], enc_size_list[3])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask, word_info):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        dec_in = zq
        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask, word_info): 
        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        return ze, zq   # !!! Check the use of ze and zq in later stages. Don't mix!!!
    
class WIDAEV2(Module):
    # Fixed word embedding
    # RL + RAL(Init)
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(WIDAEV2, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.word_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        torch.nn.init.orthogonal_(self.word_embedding.weight)
        self.word_embedding.weight.requires_grad = False  # Set trainable to False
        self.combine_layer = nn.Linear(2 * enc_size_list[3], enc_size_list[3])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask, word_info):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        dec_in = zq
        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask, word_info): 
        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        return ze, zq   # !!! Check the use of ze and zq in later stages. Don't mix!!!
    

############################ Multitask Learning (20240524) ############################
class CTCDecoderV2(Module): 
    """
    注意：decoder is only for classification for CTC pred.
    No LSTM is used here, but attention is used here as well.  
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(CTCDecoderV2, self).__init__()
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin = nn.Linear(size_list["hid"], size_list["class"])
        self.softmax = nn.LogSoftmax(dim=-1)
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask):
        outputs, attention_weights = self.attention(hid_r, hid_r, hid_r, in_mask.unsqueeze(1))
        outputs = self.lin(outputs)
        outputs = self.softmax(outputs)
        return outputs, attention_weights
    
class LinAttnDecoder(Module): 
    """
    NOTE: General purpose Linear + Attention Decoder. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LinAttnDecoder, self).__init__()
        self.attention = ScaledDotProductAttention(q_in=size_list["in"], kv_in=size_list["in"], qk_out=size_list["in"], v_out=size_list["in"])
        self.lin = nn.Linear(size_list["in"], size_list["out"])
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask):
        outputs, attention_weights = self.attention(hid_r, hid_r, hid_r, in_mask.unsqueeze(1))
        outputs = self.lin(outputs)
        return outputs, attention_weights
    
class AEPPV1(Module):
    # Autoencoder + phoneme prediction
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
class AEPPV2(Module):
    # Autoencoder + phoneme prediction
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    # NOTE: test version. Only train with phone prediction
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV2, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        # self.ae_decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        # dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        # ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        # return follows: dec_out, attn_w, z
        # TODO: tomorrow just write the trining loop. 
        return (pp_dec_out, pp_dec_out), (pp_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
class AEPPV3(Module):
    # Reconstruction + phoneme prediction
    # Reconstruction uses the same linear + attention structure as PP. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV3, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = LinAttnDecoder(size_list={"in": dec_size_list[3], "out": dec_size_list[0]}, num_layers=num_layers, dropout=dropout)
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq

class AEPPV4(Module):
    # Autoencoder + phoneme prediction
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV4, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
class AEPPV5(Module):
    # Only Recon
    # use encoder hidden as decoder hidden. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV5, self).__init__()

        self.encoder = EncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = DecoderV2(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
    def attn_encode(self, inputs, input_lens, in_mask): 
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)
        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w, attn_z = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        return (ze, zq, attn_z), ae_dec_out, ae_attn_w
    
class AEPPV6(Module):
    # Only Recon
    # use encoder hidden as decoder hidden. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV6, self).__init__()

        self.encoder = EncoderV2(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = GoodDecoderV2(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        zqs = zes
        return zes, zqs
    
    def attn_encode(self, inputs, input_lens, in_mask): 
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)
        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        zqs = zes
        dec_in = zes[-1]
        ae_dec_out, ae_attn_w, attn_z = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        return (zes, zqs, attn_z), ae_dec_out, ae_attn_w
    

class AEPPV7(Module):
    # Only PP
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV7, self).__init__()
        self.encoder = EncoderV3(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)

        ze, enc_hid = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        return (pp_dec_out, pp_dec_out), (pp_attn_w, pp_attn_w), (ze, zq)
    
    def fruitfulforward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)

        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zqs = zes
        dec_in = zes[-1]    # use the last LSTM layer's hidden representation
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        return (pp_dec_out, pp_dec_out), (pp_attn_w, pp_attn_w), (zes, zqs)
    
    def encode(self, inputs, input_lens, in_mask): 
        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        zqs = zes
        return zes, zqs
    

class AEPPV8(Module):
    # 在4的基础上增加了attn_forward
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV8, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV2(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
    def attn_forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, attn_out, ae_attn_w = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, attn_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
class AEPPV9(Module):
    # 在4的基础上增加了attn_forward
    # In addition to hidrep and attnout in AEPPV8, we return all hidden layers. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV9, self).__init__()
        self.encoder = VQEncoderV3(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV3(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
    def attn_forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze, enc_hid_out_list = self.encoder.encode_and_out(inputs, input_lens)
        # always, hid_out_list = [flo, rlo1, rlo2, ..., rloN]
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, attn_out, ae_attn_w, dec_hid_out_list = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, attn_out), (ae_attn_w, pp_attn_w), (ze, zq), (enc_hid_out_list, dec_hid_out_list)
    
class AEPPV11(Module):
    # 在4的基础上增加了attn_forward
    # In addition to hidrep and attnout in AEPPV8, we return all hidden layers. 
    # 允許多層encoder和decoder
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5, num_big_layers=1): 
        # num_big_layers: the number of encoder and decoder components. 
        super(AEPPV11, self).__init__()
        self.encoder_list = nn.ModuleList([VQEncoderV3(size_list=enc_size_list, num_layers=num_layers, dropout=dropout) for i in range(num_big_layers)])
        self.ae_decoder_list = nn.ModuleList([VQDecoderV3(size_list=dec_size_list, num_layers=num_layers, dropout=dropout) for i in range(num_big_layers)])
        # self.encoder = VQEncoderV3(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        # self.ae_decoder = VQDecoderV3(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_big_layers = num_big_layers

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hids, init_ins = [], []
        for i in range(self.num_big_layers): 
            dec_hid, init_in = self.ae_decoder_list[i].inits(batch_size=batch_size, device=self.device)
            dec_hids.append(dec_hid)
            init_ins.append(init_in)
        zes = []
        ae_dec_outs, ae_attn_ws = [], []
        for i in range(self.num_big_layers): 
            ze = self.encoder_list[i](inputs, input_lens)
            zq = ze
            dec_in = ze
            ae_dec_out, ae_attn_w = self.ae_decoder_list[i](dec_in, in_mask, init_ins[i], dec_hids[i])
            zes.append(ze)
            ae_dec_outs.append(ae_dec_out)
            ae_attn_ws.append(ae_attn_w)
        
        # it returns the last layer's output as well as all the outputs. 
        return (ae_dec_outs[-1], ae_dec_outs), (ae_attn_ws[-1], ae_attn_ws), (zes[-1], zes)
    
    def encode(self, inputs, input_lens, in_mask): 
        zes = []
        for i in range(self.num_big_layers): 
            ze = self.encoder_list[i](inputs, input_lens)
            zes.append(ze)
        return zes[-1], zes
    
    def attn_forward(self, inputs, input_lens, in_mask): 
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hids, init_ins = [], []
        for i in range(self.num_big_layers): 
            dec_hid, init_in = self.ae_decoder_list[i].inits(batch_size=batch_size, device=self.device)
            dec_hids.append(dec_hid)
            init_ins.append(init_in)

        zes, ae_dec_outs, ae_attn_ws = [], [], []
        for i in range(self.num_big_layers): 
            ze = self.encoder_list[i](inputs, input_lens)
            zq = ze
            dec_in = ze
            ae_dec_out, ae_attn_w = self.ae_decoder_list[i](dec_in, in_mask, init_ins[i], dec_hids[i])
            zes.append(ze)
            ae_dec_outs.append(ae_dec_out)
            ae_attn_ws.append(ae_attn_w)

        ze, enc_hid_out_list = self.encoder.encode_and_out(inputs, input_lens)
        # always, hid_out_list = [flo, rlo1, rlo2, ..., rloN]
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, attn_out, ae_attn_w, dec_hid_out_list = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, attn_out), (ae_attn_w, pp_attn_w), (ze, zq), (enc_hid_out_list, dec_hid_out_list)
    
class AEPPV10(Module):
    # 在4的基础上增加了attn_forward
    # In addition to hidrep and attnout in AEPPV8, we return all hidden layers. 
    # Added to each LSTM layer a linear layer to combine the hidden representation (encoder only). 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV10, self).__init__()
        self.encoder = VQEncoderV5(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV3(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
    def attn_forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze, enc_hid_out_list = self.encoder.encode_and_out(inputs, input_lens)
        # always, hid_out_list = [flo, rlo1, rlo2, ..., rloN]
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, attn_out, ae_attn_w, dec_hid_out_list = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, attn_out), (ae_attn_w, pp_attn_w), (ze, zq), (enc_hid_out_list, dec_hid_out_list)

################################ Chung's AE Replication ################################
class AEEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(AEEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["out_lin1"])
        self.rnn = nn.LSTM(input_size=size_list["out_lin1"], hidden_size=size_list["out_rnn"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list["out_rnn"] * 2, size_list["out_lin2"])

        self.act = nn.ReLU()

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
class AEDecoderV1(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(AEDecoderV1, self).__init__()
        self.lin_1 = nn.Linear(size_list["in"], size_list["out_lin1"])
        self.rnn = nn.LSTM(input_size=size_list["out_lin1"], hidden_size=size_list["out_rnn"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=False)
        self.lin_2 = nn.Linear(size_list["out_rnn"], size_list["out_lin2"])
        # self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list["out_rnn"]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list["out_rnn"]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list["in"]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x)
            # attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        # attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        # attention_weights = attention_weights.squeeze(2)
        attention_weights = None
        return outputs, attention_weights


class AE_Chung(Module):
    # Reconstruction + phoneme prediction
    # Reconstruction uses the same linear + attention structure as PP. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AE_Chung, self).__init__()

        self.encoder = AEEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = LinAttnDecoder(size_list={"in": dec_size_list[3], "out": dec_size_list[0]}, num_layers=num_layers, dropout=dropout)
        # self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq

############################ CTC Predictor [20240223] ############################
class CTCEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(CTCEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["hid"])
        self.rnn = nn.LSTM(input_size=size_list["hid"], hidden_size=size_list["hid"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list["hid"] * 2, size_list["hid"])

        self.act = nn.ReLU()

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
class CTCDecoderV1(Module): 
    """
    注意：decoder is only for classification for CTC pred. 
    This time we use dictionary to pass by the configurations. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(CTCDecoderV1, self).__init__()
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin_2 = nn.Linear(size_list["hid"], size_list["class"])
        self.softmax = nn.LogSoftmax(dim=-1)
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask):
        outputs, attention_weights = self.attention(hid_r, hid_r, hid_r, in_mask.unsqueeze(1))
        outputs = self.lin_2(outputs)
        outputs = self.softmax(outputs)
        return outputs, attention_weights

class CTCPredNetV1(Module):
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(CTCPredNetV1, self).__init__()

        self.encoder = CTCEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = CTCDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.vq_embedding = nn.Embedding(embedding_dim, enc_size_list["hid"])
        self.vq_embedding.weight.data.uniform_(-1.0 / embedding_dim,
                                               1.0 / embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        ze = self.encoder(inputs, input_lens)
        # now finding the closest vector
        # ze: [B, L, C]
        # embedding: [K, C]
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)
        # stop gradient
        dec_in = ze + (zq - ze).detach()

        dec_out, attn_w = self.decoder(dec_in, in_mask)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)

        return ze, zq
    
    def predict_on_output(self, output): 
        # output = nn.Softmax(dim=-1)(output)
        preds = torch.argmax(output, dim=-1)
        return preds
    



#########################################Transformer-like Multi-block Model#########################################
class DirectPass(Module): 
    def __init__(self):
        super(DirectPass, self).__init__()
        # used in alternation to add (residual connection)
    def forward(self, input_a, input_b): 
        return input_a

class AddPass(Module): 
    def __init__(self):
        super(AddPass, self).__init__()
        # add pass for residual connection
    def forward(self, input_a, input_b): 
        return input_a + input_b
    
class EncoderSingleV1(Module): 
    """
    20241023
    Bidirectional LSTM + Linear (merge bidirectional output)
    We use ModuleList to stack LSTM layers and allow outputting all intermediate outputs. 
    This serves as a building block of the multi-block model. 
    """
    def __init__(self, indim, outdim, num_layers=1, dropout=0.5, add_linear=False):
        # Input: (B, L, H0)
        # Output: (B, L, H0)
        super(EncoderSingleV1, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        # LSTM
        self.rnnlist = nn.ModuleList(
            [nn.LSTM(input_size=indim, hidden_size=indim,
                        batch_first=True, bidirectional=True)]
        )
        for _ in range(1, num_layers): 
            self.rnnlist.append(
                nn.LSTM(input_size=indim * 2, hidden_size=indim,
                        batch_first=True, bidirectional=True)
            )
        self.dropout = nn.Dropout(dropout)
        # final linear layer to merge bidirectional outputs; if we pass in different indim and outdim, this will be the only layer to process it.  
        self.lin = nn.Linear(indim * 2, outdim)
        # we include two passers in case we want to set one to be identity while keeping the another in the future. 
        self.passer_1 = AddPass() if add_linear else DirectPass()

    def forward(self, inputs, inputs_lens): 
        enc_x = pack_padded_sequence(inputs, inputs_lens, batch_first=True, enforce_sorted=False)
        for i, rnn in enumerate(self.rnnlist): 
            enc_x, (hn, cn) = rnn(enc_x)    # (B, L, H0) -> (B, L, H0)
            if i < self.num_layers - 1:  # No need to apply dropout after the last layer
                enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
                enc_x = self.dropout(enc_x)
                enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x_lstm_out, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x_lin_out = self.lin(enc_x_lstm_out)   # this merges the bidirectional into one.
        enc_x_pass_1 = self.passer_1(enc_x_lin_out, inputs)  # residual connection 2
        return enc_x_pass_1
    
    # this is only to be called during inference, so dropout is not applied. 
    def encode_and_out(self, inputs, inputs_lens): 
        lstm_outs_f, lstm_outs_b = [], []
        enc_x = pack_padded_sequence(inputs, inputs_lens, batch_first=True, enforce_sorted=False)
        for i, rnn in enumerate(self.rnnlist): 
            enc_x, (hn, cn) = rnn(enc_x)    # (B, L, I1) -> (B, L, I2)
            enc_x_lstm_out, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x_lstm_out_f, enc_x_lstm_out_b = enc_x_lstm_out[:, :, :self.indim], enc_x_lstm_out[:, :, self.indim:]
            lstm_outs_f.append(enc_x_lstm_out_f)
            lstm_outs_b.append(enc_x_lstm_out_b)

        enc_x_lin_out = self.lin(enc_x_lstm_out)   # this merges the bidirectional into one.
        enc_x_pass_1 = self.passer_1(enc_x_lin_out, inputs)  # residual connection 2
        return enc_x_pass_1, lstm_outs_f, lstm_outs_b # return the last output and all intermediate outputs.

class EncoderBlockV1(Module): 
    """
    20241023
    Linear + Blocked EncoderSingles
    """
    def __init__(self, indim, outdim, num_layers=1, dropout=0.5, add_linear=False, num_blocks=1):
        # Input: (B, L, I0)
        # Output: (B, L, H0)
        super(EncoderBlockV1, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.num_blocks = num_blocks

        self.lin = nn.Linear(indim, outdim)
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList(
            [EncoderSingleV1(indim=outdim, outdim=outdim, num_layers=num_layers, dropout=dropout, add_linear=add_linear) for _ in range(num_blocks)]
        )

    def forward(self, inputs, inputs_lens): 
        enc_outs = []
        enc_x = self.lin(inputs) # (B, L, I0) -> (B, L, H0)
        enc_x = self.act(enc_x)
        for block in self.blocks: 
            enc_x = block(enc_x, inputs_lens)   # NOTE: each block will take the output from the previous block. 
            enc_outs.append(enc_x)
        return enc_outs # [num_blocks, (B, L, H0)]
    
    # this is only to be called during inference, so dropout is not applied. 
    def encode_and_out(self, inputs, inputs_lens): 
        enc_hidrep_outs, enc_lstm_outs_f, enc_lstm_outs_b = [], [], []
        enc_x = self.lin(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for block in self.blocks:
            enc_x, lstm_outs_f, lstm_outs_b = block.encode_and_out(enc_x, inputs_lens)
            enc_hidrep_outs.append(enc_x)
            enc_lstm_outs_f.append(lstm_outs_f)
            enc_lstm_outs_b.append(lstm_outs_b)

        # enc_hidrep_outs = [num_blocks, (B, L, H0)]
        # enc_lstm_outs_f/b = [num_blocks, num_layers, (B, L, H0)]
        return enc_hidrep_outs, enc_lstm_outs_f, enc_lstm_outs_b # return the last output and all intermediate outputs.
    
class DecoderBlockV1(Module):
    """
    注意：decoder是自回归的，因而无需bidirectional
    同时也输出attention_out
    Additionally, we output all intermediate outputs. 
    For decoder there is no Single, directly it should be Block. 
    """
    def __init__(self, hiddim, outdim, num_layers=1, dropout=0.5, add_linear=False, num_blocks=1):
        super(DecoderBlockV1, self).__init__()
        self.rnnBlocks = nn.ModuleList([nn.LSTM(input_size=hiddim, hidden_size=hiddim,
                                                num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False) for _ in range(num_blocks)])
        self.attentionBlocks = nn.ModuleList([ScaledDotProductAttention(q_in=hiddim, kv_in=hiddim, qk_out=hiddim, v_out=hiddim) for _ in range(num_blocks)])
        self.linBlocks = nn.ModuleList([nn.Linear(hiddim, hiddim) for _ in range(num_blocks)])
        
        # non-block components
        self.lin_1 = nn.Linear(outdim, hiddim)
        self.lin_2 = nn.Linear(hiddim, outdim)
        self.act = nn.ReLU()
        self.passer_1 = AddPass() if add_linear else DirectPass()
        self.passer_2 = AddPass() if add_linear else DirectPass()
        self.passer_3 = AddPass() if add_linear else DirectPass()

        # vars
        self.hiddim = hiddim
        self.outdim = outdim
        self.num_layers = num_layers
        self.num_blocks = num_blocks

    def inits(self, batch_size, device): 
        hiddens = []
        for i in range(self.num_blocks): 
            h0 = torch.zeros((self.num_layers, batch_size, self.hiddim), dtype=torch.float, device=device, requires_grad=False)
            c0 = torch.zeros((self.num_layers, batch_size, self.hiddim), dtype=torch.float, device=device, requires_grad=False)
            hidden = (h0, c0)
            hiddens.append(hidden)
        dec_in_token = torch.zeros((batch_size, 1, self.outdim), dtype=torch.float, device=device, requires_grad=False)
        # NOTE: each LSTM needs its own hidden state, but all blocks except the first one takes in output from previous block as input. 
        return hiddens, dec_in_token

    def forward(self, hid_rs, in_mask, init_in, hiddens):
        length = hid_rs[0].size(1) # get length, all hidreps should have the same length
        dec_in_token = init_in  # (B, 1, I/O)

        final_output_frames = []    # this stores the final output from decoder
        for t in range(length): 
            # all things are processed timestep-by-timestep
            dec_x_lin1 = self.lin_1(dec_in_token)  # (B, 1, I/O) -> (B, 1, H)
            dec_x_lin1 = self.act(dec_x_lin1)
            for i in range(len(self.rnnBlocks)):
                hidrep = hid_rs[i]
                dec_inblock_in = dec_x_lin1
                dec_inblock_lstm, hiddens[i] = self.rnnBlocks[i](dec_inblock_in, hiddens[i])    # update hidden state
                dec_inblock_passer_1 = self.passer_1(dec_inblock_lstm, dec_inblock_in)  # residual connection
                dec_inblock_attention, attention_weight = self.attentionBlocks[i](dec_inblock_passer_1, hidrep, hidrep, in_mask.unsqueeze(1))
                dec_inblock_passer_2 = self.passer_2(dec_inblock_attention, dec_inblock_passer_1)  # residual connection
                dec_inblock_lin = self.linBlocks[i](dec_inblock_passer_2)
                dec_inblock_passer_3 = self.passer_3(dec_inblock_lin, dec_inblock_passer_2)  # residual connection
            dec_x_lin2 = self.lin_2(dec_inblock_passer_3)
            final_output_frames.append(dec_x_lin2)
            dec_in_token = dec_x_lin2

        final_output = torch.stack(final_output_frames, dim=1)   # stack along length dim
        final_output = final_output.squeeze(2)  # (B, L, O)
        return final_output
    
    def decode_and_out(self, hid_rs, in_mask, init_in, hiddens):
        length = hid_rs[0].size(1) # get length, all hidreps should have the same length
        dec_in_token = init_in  # (B, 1, I/O)

        final_output_frames = []    # this stores the final output from decoder
        lstm_outputs, attention_outputs, block_outputs = [[]] * self.num_blocks, [[]] * self.num_blocks, [[]] * self.num_blocks
        attention_weights = [[]] * self.num_blocks
        for t in range(length): 
            # all things are processed timestep-by-timestep
            dec_x_lin1 = self.lin_1(dec_in_token)  # (B, 1, I/O) -> (B, 1, H)
            dec_x_lin1 = self.act(dec_x_lin1)
            for i in range(len(self.rnnBlocks)):
                hidrep = hid_rs[i]
                dec_inblock_in = dec_x_lin1
                dec_inblock_lstm, hiddens[i] = self.rnnBlocks[i](dec_inblock_in, hiddens[i])    # update hidden state
                lstm_outputs[i].append(hiddens[i][0])
                dec_inblock_passer_1 = self.passer_1(dec_inblock_lstm, dec_inblock_in)  # residual connection
                dec_inblock_attention, attention_weight = self.attentionBlocks[i](dec_inblock_passer_1, hidrep, hidrep, in_mask.unsqueeze(1))
                attention_outputs[i].append(dec_inblock_attention)
                dec_inblock_passer_2 = self.passer_2(dec_inblock_attention, dec_inblock_passer_1)  # residual connection
                dec_inblock_lin = self.linBlocks[i](dec_inblock_passer_2)
                dec_inblock_passer_3 = self.passer_3(dec_inblock_lin, dec_inblock_passer_2)  # residual connection
                block_outputs[i].append(dec_inblock_passer_3)
                attention_weights[i].append(attention_weight)
            dec_x_lin2 = self.lin_2(dec_inblock_passer_3)
            final_output_frames.append(dec_x_lin2)
            dec_in_token = dec_x_lin2

        final_output = torch.stack(final_output_frames, dim=1).squeeze(-2)   # stack along length dim
        lstm_outputs_org = [torch.unbind(torch.stack(lstm_outputs[i], dim=2), dim=0) for i in range(self.num_blocks)]   # [num_blocks, num_layers, (B, L, H)]
        attention_outputs_org = [torch.stack(attention_outputs[i], dim=1).squeeze(-2) for i in range(self.num_blocks)]   # [num_blocks, (B, L, H)]
        block_outputs_org = [torch.stack(block_outputs[i], dim=1).squeeze(-2) for i in range(self.num_blocks)]   # [num_blocks, (B, L, H)]
        attention_weights_org = [torch.stack(attention_weights[i], dim=1).squeeze(-2) for i in range(self.num_blocks)]   # [num_blocks, (B, L, H)]
        return final_output, lstm_outputs_org, attention_outputs_org, block_outputs_org, attention_weights_org
    

class MultiBlockV1(Module):
    # Multi-block model with residual connections
    def __init__(self, enc_size_list, dec_size_list, num_layers=2, num_blocks=1, dropout=0.5, residual=False):
        super(MultiBlockV1, self).__init__()
        self.encoder = EncoderBlockV1(indim=enc_size_list["in"], outdim=enc_size_list["hid"], 
                                      num_layers=num_layers, num_blocks=num_blocks, 
                                      add_linear=residual, dropout=dropout)
        self.decoder = DecoderBlockV1(hiddim=dec_size_list["hid"], outdim=dec_size_list["out"], 
                                      num_layers=num_layers, num_blocks=num_blocks, 
                                      add_linear=residual, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hids, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)
        enc_outs = self.encoder(inputs, input_lens)
        dec_out = self.decoder(enc_outs, in_mask, init_in, dec_hids)
        return dec_out
    
    def run_and_out(self, inputs, input_lens, in_mask):
        batch_size = inputs.size(0)
        dec_hids, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_hidreps, enc_lstm_outs_f, enc_lstm_outs_b = self.encoder.encode_and_out(inputs, input_lens)
        dec_out, dec_lstm_outs, dec_attn_outs, dec_block_outs, attn_ws = self.decoder.decode_and_out(enc_hidreps, in_mask, init_in, dec_hids)

        # enc_hidreps = [num_blocks, (B, L, H0)]
        # enc_lstm_outs_f/b = [num_blocks, num_layers, (B, L, H0)]
        # dec_out = (B, L, O)
        # dec_lstm_outs = [num_blocks, num_layers, (B, L, H)]
        # dec_attn_outs = [num_blocks, (B, L, H)]
        # dec_block_outs = [num_blocks, (B, L, H)]
        # attn_ws = [num_blocks, (B, L, H)]
        hidlayer_outs = {}
        # Now we start to name all the outputs and put them in dict
        for idx, out in enumerate(enc_hidreps): 
            hidlayer_outs[f"hidrep-{idx+1}"] = out
        
        for idx, outs in enumerate(enc_lstm_outs_f): 
            for jdx, out in enumerate(outs): 
                hidlayer_outs[f"encrnn-{idx+1}-{jdx+1}-f"] = out

        for idx, outs in enumerate(enc_lstm_outs_b):
            for jdx, out in enumerate(outs): 
                hidlayer_outs[f"encrnn-{idx+1}-{jdx+1}-b"] = out

        for idx, outs in enumerate(dec_lstm_outs):
            for jdx, out in enumerate(outs): 
                hidlayer_outs[f"decrnn-{idx+1}-{jdx+1}-f"] = out
        
        for idx, out in enumerate(dec_attn_outs):
            hidlayer_outs[f"attnout-{idx+1}"] = out

        for idx, out in enumerate(dec_block_outs):
            hidlayer_outs[f"decrep-{idx+1}"] = out

        return dec_out, attn_ws, hidlayer_outs
    