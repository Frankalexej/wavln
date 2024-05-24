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
        # TODO: tomorrow just write the trining loop. 
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask, word_info): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq   # !!! Check the use of ze and zq in later stages. Don't mix!!!

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