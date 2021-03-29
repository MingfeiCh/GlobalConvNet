import torch.nn as nn

class GCN(nn.Module):
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

if __name__ == '__main__':
    pass