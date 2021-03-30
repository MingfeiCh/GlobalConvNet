import torch.nn as nn

class GCN(nn.Module):
    """
    Global Convolution Network which can be regarded as Spatial-wise attention.
    """
    def __init__(self, in_channels, out_channels, k=3, padding=1):
        super(GCN, self).__init__()

        self.conv_l1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, 1), padding=(padding, 0))
        self.conv_l2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, k), padding=(0, padding))

        self.conv_r1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), padding=(0, padding))
        self.conv_r2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(k, 1), padding=(padding, 0))

    def forward(self, x):
        x1 = self.conv_l1(x)
        x1 = self.conv_l2(x1)

        x2 = self.conv_r1(x)
        x2 = self.conv_r2(x2)

        out = x1 + x2

        return out

class ChannelAttention(nn.Module):
    """
    Channel-wise attention module, implemented of CBAM
    """
    def __init__(self, channel, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return out

if __name__ == '__main__':
    import torch
    in_x = torch.rand([2, 32, 10, 10], dtype=torch.float)
    gcn = GCN(32, 1, 9, 4)
    ca = ChannelAttention(32, 2)
    gcn_x = gcn(in_x)
    print(gcn_x.shape)
    ca_x = ca(in_x)
    print(ca_x.shape)
