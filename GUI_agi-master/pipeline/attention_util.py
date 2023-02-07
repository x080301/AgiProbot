# from ast import Mult
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):     #[bs,4,4096,16]
        attn = q @ k.transpose(-1, -2)      #[bs,4,4096,4096]
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        output = attn @ v       #[bs,4,4096,16]

        return output, attn
        
        

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_q, d_model_kv, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model_q, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_kv, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_kv, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_q, bias=False)

        self.attention = Attention()

        self.layer_norm1 = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model_q, eps=1e-6)
        self.bn=nn.BatchNorm1d(64)


    def forward(self, q, k, v):     #[bs,n_points,features]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head      #d_k dimention of every key     d_v: dimention of every value   
        b_size, n_q, n_k = q.size(0), q.size(1), k.size(1)      #n_q  target features dimention     n_k:source features dimention

        residual = q

        q = self.w_qs(q).view(-1, n_q, n_head, d_k)     #[bs,4096,4,16]
        k = self.w_ks(k).view(-1, n_k, n_head, d_k)     #[bs,4096,4,16]
        v = self.w_vs(v).view(-1, n_k, n_head, d_v)     #[bs,4096,4,16]
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)       #[bs,4,4096,16] [bs,4,4096,16]  [bs,4,4096,16]

        # get b x n x k x dv
        q, _ = self.attention(q, k, v)   #[bs,4,4096,16]
        
        # b x k x ndv
        q = q.transpose(1, 2).contiguous().view(b_size, n_q, -1)                #[bs,4096,64]
        s = self.layer_norm1(q)                                                 #[bs,4096,64]
        res = self.layer_norm2(residual + self.fc(s))                           #[bs,4096,features]

        return res



class SA_Layer_Single_Head(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_Single_Head, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x=x.permute(0,2,1)
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q@x_k# b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-6 + attention.sum(dim=1, keepdims=True))
        x_r =x@attention# b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        x=x.permute(0,2,1)
        return x



class SA_Layer_Multi_Head(nn.Module):
    def __init__(self,args,num_features):          #input [bs,n_points,num_features]
        super(SA_Layer_Multi_Head, self).__init__()
        self.num_heads=args.num_heads
        self.num_hidden_features=args.self_encoder_latent_features
        self.num_features=num_features
        
        self.w_qs = nn.Linear(self.num_features, self.num_heads * int(self.num_hidden_features/self.num_heads), bias=False)
        self.w_ks = nn.Linear(self.num_features, self.num_heads * int(self.num_hidden_features/self.num_heads), bias=False)
        self.w_vs = nn.Linear(self.num_features, self.num_heads * int(self.num_hidden_features/self.num_heads), bias=False)
        self.attention=Attention()
        self.norm1 = nn.LayerNorm(self.num_hidden_features)
        self.trans = nn.Linear(self.num_hidden_features,self.num_features)
        self.norm2=nn.LayerNorm(self.num_features)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b_s,n_points,_=x.size()
        original=x
        q = self.w_qs(x).view(-1, n_points, self.num_heads, int(self.num_hidden_features/self.num_heads))     #[bs,4096,4,32]
        k = self.w_ks(x).view(-1, n_points, self.num_heads, int(self.num_hidden_features/self.num_heads))     #[bs,4096,4,32]
        v = self.w_vs(x).view(-1, n_points, self.num_heads, int(self.num_hidden_features/self.num_heads))     #[bs,4096,4,32]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)           #[bs,4,4096,32] [bs,4,4096,32]  [bs,4,4096,32]
        q, attn = self.attention(q, k, v)   #[bs,4,4096,32]
        q = q.transpose(1, 2).contiguous().view(b_s, n_points, -1)        #[bs,4096,128]
        q=self.norm1(q)
        ######################
        #x=self.norm2(self.trans(q)+original)
        ###########################
        residual = self.act(self.norm2(original-self.trans(q)))
        x =original+residual
        return x



class SA_Layers(nn.Module):
    def __init__(self, n_layers,encoder_layer):
        super(SA_Layers, self).__init__()
        self.num_layers=n_layers
        self.encoder_layer=encoder_layer
        self.layers = _get_clones(self.encoder_layer, self.num_layers)
    def forward(self, x):
        for i in range(self.num_layers):
            x=self.layers[i](x)
        return x



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))



def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)



def create_conv1d_serials(channel_list):
    conv1d_serials = nn.ModuleList(
        [
            nn.Conv1d(
                in_channels=channel_list[i],
                out_channels=channel_list[i + 1],
                kernel_size=1,
            )
            for i in range(len(channel_list) - 1)
        ]
    )

    return conv1d_serials



def create_conv3d_serials(channel_list, num_points, dim):
    conv3d_serials = nn.ModuleList(
        [
            nn.Conv3d(
                in_channels=channel_list[i],
                out_channels=channel_list[i + 1],
                kernel_size=(1, 1, 1),
            )
            for i in range(len(channel_list) - 1)
        ]
    )
    conv3d_serials.insert(
        0,
        nn.Conv3d(
            in_channels=1,
            out_channels=channel_list[0],
            kernel_size=(1, num_points, dim),
        ),
    )

    return conv3d_serials



def create_rFF(channel_list, input_dim):
    rFF = nn.ModuleList([nn.Conv2d(in_channels=channel_list[i], 
                                   out_channels=channel_list[i+1],
                                   kernel_size=(1,1)) for i in range(len(channel_list) - 1)])
    rFF.insert(0, nn.Conv2d(in_channels=1, 
                            out_channels=channel_list[0], 
                            kernel_size=(input_dim,1)))

    return rFF



def create_rFF3d(channel_list, num_points, dim):
    rFF = nn.ModuleList([nn.Conv3d(in_channels=channel_list[i], 
                                   out_channels=channel_list[i+1],
                                   kernel_size=(1,1,1)) for i in range(len(channel_list) - 1)])
    rFF.insert(0, nn.Conv3d(in_channels=1, 
                            out_channels=channel_list[0], 
                            kernel_size=(1, num_points, dim)))

    return rFF



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, last_dim=256, dropout=0.1, activation=F.relu):
        super(PTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, last_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        return tgt



class PTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, last_layer, norm=None):
        super(PTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)    #repeat the decoder layers
        self.last_layer = last_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm:
            output = self.norm(output)

        output = self.last_layer(output, memory)

        return output
