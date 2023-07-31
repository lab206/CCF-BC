import torch
import torch.nn as nn
import torch._utils
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.DANet import DAModule, PositionAttentionModule, ChannelAttentionModule
from fightingcv_attention.attention.ExternalAttention import ExternalAttention

from isegm.model.ops import ScaleLayer


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ClickPixelCognitionFusionModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, norm_layer=None, relu_inplace=True):
        super(ClickPixelCognitionFusionModule, self).__init__()
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=relu_inplace)
        )

        self.induct_gate = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=relu_inplace)
        )

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            ScaleLayer(init_value=0.05, lr_mult=1)
        )

        self.padding = (kernel_size - 1) // 2
        ratio = 4
        self.conv_q_channel = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_channel = nn.Conv2d(
            self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_channel = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_spatial = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                        bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_m_spatial = nn.MaxPool1d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)
        self.conv_v_spatial = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                        bias=False)  # theta
        self.softmax_spatial = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_channel, mode='fan_in')
        kaiming_init(self.conv_v_channel, mode='fan_in')
        kaiming_init(self.conv_q_spatial, mode='fan_in')
        kaiming_init(self.conv_v_spatial, mode='fan_in')

        self.conv_q_channel.inited = True
        self.conv_v_channel.inited = True
        self.conv_q_spatial.inited = True
        self.conv_v_spatial.inited = True

    def channel_perception_node(self, x):
        p_x = self.conv_v_channel(x)
        B, C, H, W = p_x.size()
        p_x = p_x.view(B, C, H * W)
        q_mask = self.conv_q_channel(x)
        q_mask = q_mask.view(B, 1, H * W)
        q_mask = self.softmax_channel(q_mask)
        p_x_mask = torch.matmul(p_x, q_mask.transpose(1, 2))
        p_x_mask = p_x_mask.unsqueeze(-1)
        p_x_mask = self.conv_up(p_x_mask)
        mask_ch = self.sigmoid(p_x_mask)
        ca_x = x * mask_ch
        return ca_x

    def spatial_perception_node(self, x):
        ca_x = self.conv_q_spatial(x)
        B, C, H, W = ca_x.size()
        avg_x = self.avg_pool(ca_x)
        avg_x_res = self.conv_m_spatial(avg_x.flatten(1).unsqueeze(1))
        avg_x_res = avg_x_res.permute(0, 2, 1).unsqueeze(-1)
        avg_x = avg_x + avg_x_res
        B, C, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(B, C, avg_x_h * avg_x_w).permute(0, 2, 1)
        v_x = self.conv_v_spatial(x).view(B, self.inter_planes, H * W)
        v_x = self.softmax_spatial(v_x)
        avg_v_x = torch.matmul(avg_x, v_x)
        avg_v_x = avg_v_x.view(B, 1, H, W)
        mask_sp = self.sigmoid(avg_v_x)
        s_x = x * mask_sp
        return s_x

    def forward(self, x, additional_features):
        x = self.conv_block(x)
        additional_features = self.conv_stem(additional_features)
        coarse = x + additional_features
        context_channel = self.channel_perception_node(coarse)
        # context_spatial = context_spatial + additional_features
        context_spatial = self.spatial_perception_node(context_channel)
        # x = x + context_channel
        return self.induct_gate(context_spatial)
