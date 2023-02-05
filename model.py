import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 p_dropout=0,
                 use_bn=True,
                 bn_affine=True):
        super().__init__()

        use_bias = not (use_bn and bn_affine)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=use_bias)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(
            p_dropout) if p_dropout > 0 else nn.Identity()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channels, affine=bn_affine)

        torch.nn.init.kaiming_normal_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class Wav2Letter(nn.Module):

    def __init__(self, num_classes: int = 29,
                 num_features: int = 1,
                 p=0.1):
        super(Wav2Letter, self).__init__()

        waveform_model = Conv1dBlock(
            in_channels=num_features, out_channels=250, 
            kernel_size=250, stride=160, padding=45, 
            use_bn=False, p_dropout=0)

        acoustic_model = nn.Sequential(
            Conv1dBlock(in_channels=250, out_channels=250,
                        kernel_size=48, stride=2, padding=23, p_dropout=p),

            *(Conv1dBlock(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3, p_dropout=p)
              for _ in range(7)),

            Conv1dBlock(in_channels=250, out_channels=2000,
                        kernel_size=32, stride=1, padding=16, p_dropout=p),
            Conv1dBlock(in_channels=2000, out_channels=2000,
                        kernel_size=1, stride=1, padding=0, p_dropout=p),
            Conv1dBlock(in_channels=2000, out_channels=num_classes,
                        kernel_size=1, stride=1, padding=0, use_bn=False),
        )

        self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)

    def forward(self, x):

        x = self.acoustic_model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
