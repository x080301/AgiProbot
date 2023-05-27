import math

from torch import nn


def split_heads(x, heads, depth):
    x = x.view(x.shape[0], heads, depth, x.shape[2])  # _  (B,C,N) -> (B,H,D,N)
    return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_heads=1):
        super(SelfAttentionLayer, self).__init__()
        self.q_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.trans_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.after_norm = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.feed_forward_cov1 = nn.Conv1d(128, 512, 1)
        self.relu = nn.ReLU()
        self.feed_forward_cov2 = nn.Conv1d(512, out_channels, 1)
        self.feed_forward_bn = nn.BatchNorm1d(out_channels)

        self.num_heads = num_heads
        assert in_channels % num_heads == 0, 'in_channels should be divisible by num_heads.'
        self.q_depth = in_channels // num_heads
        self.k_depth = in_channels // num_heads
        self.v_depth = in_channels // num_heads

        self.feed_forward = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(512, out_channels, 1, bias=False))

    def forward(self, x):
        # _                                                                         input (8,N,128)

        x = x.permute(0, 2, 1)  # _                                                 (B,N,128) -> (B,128,N)

        x_q = self.q_conv(x)
        x_q = split_heads(x_q, self.num_heads, self.q_depth)  # _                   (B,128,N) -> (B,H,D,N)
        x_q = x_q.permute(0, 1, 3, 2).contiguous()  # _                             (B,H,D,N) -> (B,H,N,D)

        x_k = self.k_conv(x)  # _                                                   (B,128,N) -> (B,32,N)
        x_k = split_heads(x_k, self.num_heads, self.k_depth).contiguous()  # _      (B,128,N) -> (B,H,D,N)

        x_v = self.v_conv(x)  # _                                                   (B,128,N) -> (B,128,N)
        x_v = split_heads(x_v, self.num_heads, self.v_depth).contiguous()  # _      (B,128,N) -> (B,H,D,N)

        energy = x_q @ x_k  # _                                                     (B,H,N,D) @ (B,H,D,N)  -> (B,H,N,N)

        scale_factor = math.sqrt(x_q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # _                        (B,H,N,N) -> (B,H,N,N)

        residual = x
        x = x_v @ attention  # _                                                    (B,H,D,N) @ (B,H,N,N) -> (B,H,D,N)
        x = x.reshape(x.shape[0], -1, x.shape[3])  # _                              (B,H,D,N) -> (B,128,N)

        x = self.after_norm(x + residual)  # _                                      (B,128,N) + (B,128,N) -> (B,128,N)

        # feed forward
        residual = x
        x = self.feed_forward(x)  # _                                               (B,128,N) -> (B,512,N) -> (B,128,N)
        x = self.feed_forward_bn(x + residual)  # _                                 (B,128,N) + (B,128,N) -> (B,128,N)

        x = x.permute(0, 2, 1)  # _                                                 (B,128,N) -> (B,N,128)

        return x


class SALayerSingleHead(nn.Module):
    def __init__(self, channels):
        super(SALayerSingleHead, self).__init__()
        self.self_attention_single_head = SelfAttentionLayer(channels, out_channels=128, num_heads=1)

    def forward(self, x):
        return self.self_attention_single_head(x)


class SALayerSingleHeadArchive(nn.Module):
    def __init__(self, channels):
        super(SALayerSingleHeadArchive, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.feed_forward_cov1 = nn.Conv1d(128, 512, 1)
        self.relu = nn.ReLU()
        self.feed_forward_cov2 = nn.Conv1d(512, 128, 1)
        self.feed_forward_bn = nn.BatchNorm1d(128)

    def forward(self, x):
        # _                                                                         input (8,N,128)

        x = x.permute(0, 2, 1)  # _                                                 (B,N,128) -> (B,128,N)
        x_q = self.q_conv(x).permute(0, 2, 1)  # _                                  (B,128,N) -> (B,N,32)
        x_k = self.k_conv(x)  # _                                                   (B,128,N) -> (B,32,N)
        x_v = self.v_conv(x)  # _                                                   (B,128,N) -> (B,128,N)
        energy = x_q @ x_k  # _                                                     (B,N,32) @ (B,32,N) -> (B,N,N)

        scale_factor = math.sqrt(x_v.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # _                                                                         (B,N,N) -> (B,N,N)

        # attention = attention / (1e-6 + attention.sum(dim=1, keepdims=True))  # _   (B,N,N) -> (B,N,N)

        x_r = x_v @ attention  # _                                                  (B,128,N) @ (B,N,N) -> (B,128,N)

        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # _              (B,128,N) -> (B,128,N)

        x = x + x_r  # _                                                            (B,128,N) + (B,128,N) -> (B,128,N)

        # feed forward
        residual = x
        x = self.relu(self.feed_forward_cov1(x))
        # _                                                                         (B,128,N) -> (B,512,N)
        x = self.feed_forward_cov2(x)  # _                                          (B,512,N) -> (B,128,N)
        x = self.feed_forward_bn(residual + x)  # _                                 (B,128,N) + (B,128,N) -> (B,128,N)

        x = x.permute(0, 2, 1)  # _                                                 (B,128,N) -> (B,N,128)

        return x
