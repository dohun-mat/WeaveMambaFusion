import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from lib_mamba.vmambanew import SS2D
from functools import partial

class CPM3_BN(nn.Module):
    def __init__(self, in_channels):
        super(CPM3_BN, self).__init__()

        self.in_channels = in_channels

        self.res_branch1 = nn.Sequential(
            DeformableConv2d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels // 2),
            # nn.ReLU(True),
        )

        self.res_branch2 = nn.Sequential(
            DeformableConv2d(in_channels // 2, in_channels // 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels // 4),
            # nn.ReLU(True),
            DeformableConv2d(in_channels // 4, in_channels // 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels // 4),
            # nn.ReLU(True),
        )

        self.res_branch3 = nn.Sequential(
            DeformableConv2d(in_channels // 4, in_channels // 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels // 4),
            # nn.ReLU(True),
            DeformableConv2d(in_channels // 4, in_channels // 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels // 4),
            # nn.ReLU(True),
        )

    def forward(self, x):
        res1 = self.res_branch1(x)
        res2 = self.res_branch2(res1)
        res3 = self.res_branch3(res2)
        out = torch.cat([res1, res2, res3], dim=1)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

class WeaveMambaFusion(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        assert dim % n_div == 0
        self.dim_partial   = dim // n_div
        self.dim_untouched = dim - self.dim_partial

        # ★ Fix: 전체 채널(dim) 대상의 LayerNorm
        self.norm_full = nn.LayerNorm(dim)
        
        self.mamba = SS2D(
            d_model=self.dim_partial,
            d_state=16, ssm_ratio=1,
            initialize="v2", forward_type="v052d",
            channel_first=True, k_group=2,
        )
        
        self.in_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
    
        # 1) W축 interleave
        x_inter = torch.empty((B, C, H, W * 2), device=x1.device, dtype=x1.dtype)
        x_inter[..., 0::2] = x1
        x_inter[..., 1::2] = x2

        # 2) 전체 텐서에 LayerNorm
        x_inter_n = x_inter.permute(0, 2, 3, 1).contiguous()
        x_inter_n = self.norm_full(x_inter_n)
        x_inter_n = x_inter_n.permute(0, 3, 1, 2).contiguous()

        # 3) PConv-style split (두 branch 모두 같은 scale)
        x_p, x_u = torch.split(x_inter_n, [self.dim_partial, self.dim_untouched], dim=1)

        # 4) SS2D는 partial에만
        x_p = self.mamba(x_p)

        # 5) Concat
        x_cat = torch.cat([x_p, x_u], dim=1)

        # 6) De-interleave
        x1_out = x_cat[..., 0::2]
        x2_out = x_cat[..., 1::2]

        # 7) Fusion
        x_fused = torch.cat([x1_out, x2_out], dim=1)
        out = self.act(self.in_proj(x_fused))

        return out

    
class Weave_BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(Weave_BiFPN, self).__init__()
        self.epsilon = epsilon

        self.conv4_up = WeaveMambaFusion(num_channels)
        self.conv3_up = WeaveMambaFusion(num_channels)
        
        self.conv4_down = WeaveMambaFusion(num_channels)
        self.conv5_down = WeaveMambaFusion(num_channels)
        
        # Feature scaling layers
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish =  Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
       
        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0
            p3_in, p4_in, p5_in = inputs

        
        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        
        p5_upsamp_p4 = F.interpolate(p5_in,size=[p4_in.size(2), p4_in.size(3)], mode="bilinear", align_corners=False)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in), self.swish(weight[1] * p5_upsamp_p4))
        
        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p4_upsamp_p3 = F.interpolate(p4_up,size=[p3_in.size(2), p3_in.size(3)], mode="bilinear", align_corners=False)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in), self.swish(weight[1] * p4_upsamp_p3))
        
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_up), self.swish(weight[1] * self.p4_downsample(p3_out)))

       
        # Weights for Pt_0 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0 and P4_2 to P5_2
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in) , self.swish(weight[1] * self.p5_downsample(p4_out)))

        return p3_out, p4_out, p5_out

from ops_dcnv3 import modules as dcnv3   # 경로는 환경에 맞게

class DeformableConv2d(nn.Module):
    """
    nn.Conv2d 호환 인터페이스의 DCNv3 wrapper.
    - 입출력: (B, C, H, W) NCHW
    - 내부: DCNv3 (NHWC, 3×3 deformable, 채널 보존) -> 1×1 conv (in→out 변환)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        # ↓ DCNv3 추가 옵션 (필요시 override)
        group=None,
        dilation=1,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        dw_kernel_size=None,
        center_feature_scale=False,
        core_op_name='DCNv3',
    ):
        super().__init__()

        # group 미지정 시 자동: 그룹당 채널 ≈ 16 (InternImage 표준)
        if group is None:
            group = max(1, in_channels // 16)

        assert in_channels % group == 0, (
            f"in_channels({in_channels}) must be divisible by group({group})"
        )

        # (1) DCNv3: 3×3 deformable, 채널 보존
        core_op = getattr(dcnv3, core_op_name)
        self.dcn = core_op(
            channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad=padding,
            dilation=dilation,
            group=group,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size,
            center_feature_scale=center_feature_scale,
        )

        # (2) 1×1 conv: in_channels -> out_channels
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # NCHW -> NHWC (DCNv3 입력 포맷)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.dcn(x)
        # NHWC -> NCHW (출력 및 1×1 conv를 위해)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x


def channel_shuffle2(x, groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle2(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self, channels, groups):
        super(ChannelShuffle2, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle2(x, self.groups)

    
class ConvBlock(nn.Module):
    """
    EResFD의 기본 구성 블록 (수정 없음)
    """
    def __init__(self, in_channel, depth, stride=1):
        super(ConvBlock, self).__init__()
        intermediate_depth = depth
        if in_channel != depth or stride == 2:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        else:
            self.shortcut_layer = None

        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channel, intermediate_depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(intermediate_depth),
            nn.ReLU(False),
            nn.Conv2d(intermediate_depth, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = x
        x = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(shortcut)
        x += shortcut
        return x


CM_IN_FPN = 1

class DetEncoder(nn.Module):
    """
    수정됨: FPN 및 FEM 모듈 제거.
    순수하게 Feature Map들을 추출하여 리스트로 반환하는 역할만 수행.
    """
    def __init__(self, num_features=256):
        super(DetEncoder, self).__init__()
        self.num_features = num_features

        # Stride 4 (Input) -> Stride 8
        self.layer1 = nn.Sequential(
            ConvBlock(self.num_features, int(self.num_features * CM_IN_FPN), stride=2),
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
        )
        # Stride 8 -> Stride 16
        self.layer2 = nn.Sequential(
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN), stride=2),
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
        )
        # Stride 16 -> Stride 32
        self.layer3 = nn.Sequential(
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN), stride=2),
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
            ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
        )
        # Stride 32 -> Stride 64
        # self.layer4 = nn.Sequential(
        #     ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN), stride=2),
        #     ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
        # )
        # # Stride 64 -> Stride 128
        # self.layer5 = nn.Sequential(
        #     ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN), stride=2),
        #     ConvBlock(int(self.num_features * CM_IN_FPN), int(self.num_features * CM_IN_FPN)),
        # )

    def forward(self, inp) -> List[torch.Tensor]:
        # inp: Stride 4 feature map from EResNet stem
        conv0 = inp                         # Stride 4 (P2)
        conv1 = self.layer1(conv0)          # Stride 8 (P3)
        conv2 = self.layer2(conv1)          # Stride 16 (P4)
        conv3 = self.layer3(conv2)          # Stride 32 (P5)
        # conv4 = self.layer4(conv3)          # Stride 64 (P6)
        # conv5 = self.layer5(conv4)          # Stride 128 (P7)

        # BiFPN에 필요한 Multi-scale feature map 리스트 반환
        # 보통 BiFPN은 P3~P7을 사용하므로 상황에 맞춰 슬라이싱해서 쓰면 됩니다.
        # return [conv0, conv1, conv2, conv3, conv4, conv5]
        return [conv0, conv1, conv2, conv3]


class EResNet(nn.Module):
    def __init__(self, width_mult=0.25):
        super(EResNet, self).__init__()

        # Base Stem part (Input -> Stride 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(128 * width_mult), kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(int(128 * width_mult)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(128 * width_mult), int(128 * width_mult), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 * width_mult)),
            nn.ReLU(False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(128 * width_mult), int(256 * width_mult), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(256 * width_mult)),
            nn.ReLU(False),
        )
        self.conv4 = ConvBlock(int(256 * width_mult), int(256 * width_mult))

        # Encoder stacking part
        self.encoder = DetEncoder(int(256 * width_mult))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        # Stem Forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # 여기까지 오면 이미지는 1/4 크기로 줄어듦

        # Encoder Forward (Multi-scale features 추출)
        features = self.encoder(x)
        
        return features

def _get_backbone(width_mult=0.25):
    return EResNet(width_mult)

