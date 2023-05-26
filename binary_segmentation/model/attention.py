import math

from torch import nn


class SALayerSingleHead(nn.Module):
    def __init__(self, channels):
        super(SALayerSingleHead, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # TODO
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

        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # TODO  # _              (B,128,N) -> (B,128,N)

        x = x + x_r  # _                                                            (B,128,N) + (B,128,N) -> (B,128,N)

        # feed forward
        residual = x
        x = self.relu(self.feed_forward_cov1(x))
        # _                                                                         (B,128,N) -> (B,512,N)
        x = self.feed_forward_cov2(x)  # _                                          (B,512,N) -> (B,128,N)
        x = self.feed_forward_bn(residual + x)  # _                                 (B,128,N) + (B,128,N) -> (B,128,N)

        x = x.permute(0, 2, 1)  # _                                                 (B,128,N) -> (B,N,128)

        return x
