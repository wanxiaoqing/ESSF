import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class MSFFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels // 4
        self.spectral3 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral5 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 5), padding=(0, 0, 2), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral7 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 7), padding=(0, 0, 3), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral9 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 9), padding=(0, 0, 4), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )

    def forward(self, x):
        out3 = self.spectral3(x)
        out5 = self.spectral5(x)
        out7 = self.spectral7(x)
        out9 = self.spectral9(x)
        return torch.cat((out3, out5, out7, out9), dim=1)


class CMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SSAAMM(nn.Module):
    def __init__(self, channels, factor=16):
        super(SSAAMM, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, "The number of channels must be divisible by the packet factor."

        self.softmax = nn.Softmax(dim=-1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)

        x_reduced = self.conv1x1(x)
        x_reduced = F.relu(self.gn(x_reduced))

        x_enhanced = self.conv3x3(x_reduced)
        x_enhanced = self.bn(x_enhanced)

        weights = self.softmax(self.adaptive_pool(x_enhanced))

        x_weighted = weights * x

        out = x_weighted + residual
        return out


class EASSFEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EASSFEM, self).__init__()
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ssaaem = SSAAMM(in_channels)
        self.cmlp = CMLP(in_features=in_channels)

    def forward(self, x):
        residual1 = x
        out = self.pw_conv(x)
        out += residual1

        residual2 = out
        out = self.BN(out)
        out = self.ssaaem(out)
        out += residual2

        residual3 = out
        out = self.BN(out)
        out = self.cmlp(out)  # 加mlp 60% IP
        out += residual3
        return out


class CFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFN, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gelu1 = nn.GELU()
        self.pw_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # padding=1保持特征图大小不变
        self.gelu2 = nn.GELU()
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1_1(x)
        x = self.gelu1(x)
        x = self.pw_conv(x)
        x = self.gelu2(x)
        x = self.conv1x1_2(x)
        return x


class ESSF(nn.Module):
    def __init__(self, channels=1, num_classes=16, drop=0.1):
        super(ESSF, self).__init__()
        self.stem_3D = nn.Sequential(
            nn.Conv3d(channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.msffm = MSFFM(32)

        self.stem_2D = nn.Sequential(
            nn.Conv2d(in_channels=32 * 28, out_channels=64, kernel_size=(3, 3)),  # 32 * 28
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.eassfem = EASSFEM(in_channels=64, out_channels=64)
        self.cfn = CFN(in_channels=64, out_channels=64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem_3D(x)
        x = self.msffm(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.stem_2D(x)
        x = self.eassfem(x)
        x = self.cfn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = ESSF(channels=1, num_classes=9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input = torch.randn(64, 1, 30, 13, 13).to(device)
    y = model(input)
    print(y.size())
