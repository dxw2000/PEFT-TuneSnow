import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# from exp_SAM_desnow.base_net_snow import *
# from exp_SAM_desnow.LoRa_segment_anything.modeling.common import LayerNorm2d
# from exp_SAM_desnow.LoRa_segment_anything import sam_model_registry
from ..base_net_snow import *
from ..LoRa_segment_anything.modeling.common import LayerNorm2d
from ..LoRa_segment_anything import sam_model_registry


import torch.nn.functional as F


# 插入到SAM-image-encoder中间的Prompt
# class PromptGen(nn.Module):
#     def __init__(self, blk, reduction=4, cls_token=False, reshape=False, seq_size=None) -> None:
#         super(PromptGen, self).__init__()
#         self.block = blk
#         dim = blk.attn.qkv.in_features
#         prompt_dim = dim // reduction
#         self.prompt_learn = nn.Sequential(
#             # nn.Linear(dim, 32),
#             # nn.GELU(),
#             # nn.Linear(32, dim),
#             # nn.GELU()
#             nn.Conv2d(dim, prompt_dim, 1, 1),
#             LayerNorm2d(prompt_dim),
#             nn.GELU(),
#             nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
#             LayerNorm2d(prompt_dim),
#             nn.GELU(),
#             nn.Conv2d(prompt_dim, dim, 1, 1),
#             LayerNorm2d(dim),
#             nn.GELU()
#         )
#         self.cls_token = cls_token
#         self.reshape = reshape
#         self.seq_size = seq_size
#
#     def forward(self, x):
#         if self.cls_token:
#             tokens = x[:, 1:]
#             bs, seq_len, dim = tokens.size()
#             if self.reshape:
#                 tokens = tokens.reshape(-1, self.seq_size, self.seq_size, dim).permute(0, 3, 1, 2)
#             prompt = self.prompt_learn(tokens)
#             promped = tokens + prompt
#             promped = promped.reshape(bs, dim, seq_len).transpose(1, 2)
#             promped = torch.cat([x[:, 0].unsqueeze(1), promped], dim=1)
#         else:
#             prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
#             # prompt = self.prompt_learn(x)
#             promped = x + prompt
#         net = self.block(promped)
#         return net

# 插入到SAM-image-encoder中间的Prompt
class No_PromptGen(nn.Module):
    def __init__(self, blk, reduction=4, cls_token=False, reshape=False, seq_size=None) -> None:
        super(No_PromptGen, self).__init__()
        self.block = blk

    def forward(self, x):
        net = self.block(x)
        return net


# sam, _ = sam_model_registry[model_name](image_size=128, num_classes=8, checkpoint=checkpoint)

# def SAM_Image_Encoder(model_name, image_size, num_classes, checkpoint):


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class SAM_Image_Encoder(nn.Module):
    def __init__(self, model_name, image_size, checkpoint, num_classes=2):
        super(SAM_Image_Encoder, self).__init__()
        sam, _ = sam_model_registry[model_name](image_size=image_size, num_classes=num_classes, checkpoint=checkpoint)
        # 摘出SAM-image_encoder
        del sam.prompt_encoder
        del sam.mask_decoder
        out_dim = sam.image_encoder.neck[0].out_channels
        # 冻结image_encoder
        # for param in sam.image_encoder.parameters():
        #     param.requires_grad = False
        for n, value in sam.image_encoder.named_parameters():

            # 微调Adapter层和相对位置编码层
            if "Adapter" not in n and "rel_pos" not in n:
                value.requires_grad = False

        # #  添加LoRA_QKV分支
        self.lora_layer = list(
            range(len(sam.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, 4, bias=False)
            w_b_linear_q = nn.Linear(4, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, 4, bias=False)
            w_b_linear_v = nn.Linear(4, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()

        img_size = sam.image_encoder.img_size
        blocks = []
        for block in sam.image_encoder.blocks:
            blocks.append(
                No_PromptGen(block, reduction=4)
            )
        sam.image_encoder.blocks = nn.Sequential(*blocks)
        # self.sam_image_encoder = sam.image_encoder.blocks
        self.sam_image_encoder_patch_embed = sam.image_encoder.patch_embed
        self.sam_image_encoder_0_1 = sam.image_encoder.blocks[0:2]
        self.sam_image_encoder_2_3_4 = sam.image_encoder.blocks[2:5]
        self.sam_image_encoder_5_6_7 = sam.image_encoder.blocks[5:8]
        self.sam_image_encoder_8_9_10_11 = sam.image_encoder.blocks[8:12]
        self.sam_image_encoder_neck = sam.image_encoder.neck

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):

        # print("sam_image_encoder:", self.sam_image_encoder)
        # print("ok")
        # x = self.sam_image_encoder(x)
        x = self.sam_image_encoder_patch_embed(x)
        x_0_1 = self.sam_image_encoder_0_1(x)
        x_2_3_4 = self.sam_image_encoder_2_3_4(x_0_1)
        x_5_6_7 = self.sam_image_encoder_5_6_7(x_2_3_4)
        x_8_9_10_11 = self.sam_image_encoder_8_9_10_11(x_5_6_7)

        x_out = self.sam_image_encoder_neck(x_8_9_10_11.permute(0, 3, 1, 2))

        return x_0_1, x_2_3_4, x_5_6_7, x_8_9_10_11, x_out


# model_name="vit_b"
# checkpoint="ckpt/sam_vit_b_01ec64.pth"
# SAM_IE = SAM_Image_Encoder(model_name=model_name, image_size=128, checkpoint=checkpoint)
# image = torch.rand((1, 3, 128, 128))
# out = SAM_IE(image)
# print("out:", out.shape)
# print("ok")


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)
    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class channel_shuffle(nn.Module):
    def __init__(self, groups=3):
        super(channel_shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.groups == 0
        C_per_group = C // self.groups
        x = x.view(B, self.groups, C_per_group, H, W)
        x = x.transpose(1, 2).contiguous()

        x = x.view(B, C, H, W)
        return x


class overlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, dim=768):
        super(overlapPatchEmbed, self).__init__()

        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."

        self.dim = dim
        self.num_heads = num_head
        head_dim = dim // num_head
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        B, N, C = x.shape
        x_conv = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x.transpose(1, 2).reshape(B, C, H, W))
        x = self.proj_drop(x)
        x = x + x_conv
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class MFFN(nn.Module):
    def __init__(self, dim, FFN_expand=2, norm_layer='WithBias'):
        super(MFFN, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim * FFN_expand, 1)
        self.conv33 = nn.Conv2d(dim * FFN_expand, dim * FFN_expand, 3, 1, 1, groups=dim * FFN_expand)
        self.conv55 = nn.Conv2d(dim * FFN_expand, dim * FFN_expand, 5, 1, 2, groups=dim * FFN_expand)
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        x33 = self.conv33(x1)
        x55 = self.conv55(x1)
        x = x1 + x33 + x55
        x = self.sg(x)
        x = self.conv4(x)
        return x


class Scale_aware_Query(nn.Module):
    def __init__(self,
                 dim,
                 out_channel,
                 window_size,
                 num_heads):
        super().__init__()
        self.dim = dim
        self.out_channel = out_channel
        self.window_size = window_size
        self.conv = nn.Conv2d(dim, out_channel, 1, 1, 0)

        layers = []
        for i in range(3):
            layers.append(CALayer(out_channel, 4))
            layers.append(SALayer(out_channel, 4))
        self.globalgen = nn.Sequential(*layers)

        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = out_channel // self.num_heads

    def forward(self, x):
        x = self.conv(x)
        x = F.upsample(x, (self.window_size, self.window_size), mode="bicubic")
        x = self.globalgen(x)
        B = x.shape[0]
        ide = x.reshape(B, 1, self.N, self.num_heads, self.dim_head)
        # print("ide:", ide.shape)
        # print("ok")
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x


class LocalContext_Interaction(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalContext_Interaction(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_ // B, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, C // self.num_heads)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Context_Interaction_Block(nn.Module):

    def __init__(self,
                 latent_dim,
                 dim,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=LocalContext_Interaction,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False

        self.gamma1 = 1.0
        self.gamma2 = 1.0

    def forward(self, x, q_global):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class Context_Interaction_layer(nn.Module):
    def __init__(self, n, latent_dim, in_channel, head, window_size, globalatten=False):
        super(Context_Interaction_layer, self).__init__()

        # layers=[]
        self.globalatten = globalatten
        self.model = nn.ModuleList([
            Context_Interaction_Block(
                latent_dim,
                in_channel,
                num_heads=head,
                window_size=window_size,
                attention=GlobalContext_Interaction if i % 2 == 1 and self.globalatten == True else LocalContext_Interaction,
            )
            for i in range(n)])

        if self.globalatten == True:
            self.gen = Scale_aware_Query(latent_dim, in_channel, window_size=8, num_heads=head)

    def forward(self, x, latent):
        if self.globalatten == True:
            q_global = self.gen(latent)
            x = _to_channel_last(x)
            for model in self.model:
                x = model(x, q_global)
        else:
            x = _to_channel_last(x)
            for model in self.model:
                x = model(x, 1)
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.reduction = reduction
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class Refine_Block(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Refine_Block, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class Refine(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(Refine, self).__init__()
        modules_body = []
        modules_body = [Refine_Block(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# 投影头的注意力模块

class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(in_channels, num_experts)
        )

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)


class DynamicCondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2, rooting_channels=512):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicCondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(rooting_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs_q):
        inputs = inputs_q[0]
        kernel_conditions = inputs_q[1]
        b, _, _, _ = inputs.size()
        res = []
        for i, input in enumerate(inputs):
            input = input.unsqueeze(0)
            routing_weights = self._routing_fn(kernel_conditions[i])
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_dynamic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = CondConv2D(dim, dim, 5, 1, 2, 1, dim)
        self.act1 = nn.GELU()
        self.conv_spatial = CondConv2D(dim, dim, 7, 1, 9, 3, dim)
        self.act2 = nn.GELU()
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.act1(attn)
        attn = self.conv_spatial(attn)
        attn = self.act2(attn)
        attn = self.conv1(attn)

        return u * attn


class LKAAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv_epsa(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()

        self.conv_1 = conv_epsa(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                                stride=stride, groups=conv_groups[0])
        self.conv_2 = conv_epsa(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                                stride=stride, groups=conv_groups[1])
        self.conv_3 = conv_epsa(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                                stride=stride, groups=conv_groups[2])
        self.conv_4 = conv_epsa(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                                stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSA_BLOCK(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EPSA_BLOCK, self).__init__()
        self.conv11_in = nn.Conv2d(in_channel, 64, 1)
        self.PASM = PSAModule(64, 64)
        self.conv11_out = nn.Conv2d(64, out_channel, 1)

    def forward(self, x):
        x = self.conv11_in(x)
        x = self.PASM(x)
        x = self.conv11_out(x)

        return x


# class xw_refine(nn.Module):
#     def __init__(self, dim):
#         super(xw_refine, self).__init__()
#
#         self.up1 = UpSample()
#         self.up2 =
#         self.up3 =
#
#
#
#
#     def forward(self, x):


# class HFPH_esa(nn.Module):
#     def __init__(self, n_feat, fusion_dim, kernel_size, reduction, act, bias, num_cab):
#         super(HFPH_esa, self).__init__()
#         self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
#         self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
#         self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
#         self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
#
#         self.up_1 = UpSample(n_feat[1], fusion_dim, s_factor=2)
#         self.up_2 = UpSample(n_feat[2], fusion_dim, s_factor=4)
#         self.up_3 = UpSample(n_feat[3], fusion_dim, s_factor=8)
#
#         layer0 = []
#         for i in range(2):
#             layer0.append(CALayer(fusion_dim, 16))
#             layer0.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
#         self.conv_enc0 = nn.Sequential(*layer0)
#
#         layer1 = []
#         for i in range(2):
#             layer1.append(CALayer(fusion_dim, 16))
#             layer1.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
#         self.conv_enc1 = nn.Sequential(*layer1)
#
#         layer2 = []
#         for i in range(2):
#             layer2.append(CALayer(fusion_dim, 16))
#             layer2.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
#         self.conv_enc2 = nn.Sequential(*layer2)
#
#         layer3 = []
#         for i in range(2):
#             layer3.append(CALayer(fusion_dim, 16))
#             layer3.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
#         self.conv_enc3 = nn.Sequential(*layer3)
#
#     def forward(self, x, encoder_outs):
#         x = x + self.conv_enc0(encoder_outs[0])
#         x = self.refine0(x)
#
#         x = x + self.conv_enc1(self.up_1(encoder_outs[1]))
#         x = self.refine1(x)
#
#         x = x + self.conv_enc2(self.up_2(encoder_outs[2]))
#         x = self.refine2(x)
#
#         x = x + self.conv_enc3(self.up_3(encoder_outs[3]))
#         x = self.refine3(x)
#
#         return x


class HFPH_LKA(nn.Module):
    def __init__(self, n_feat, fusion_dim, kernel_size, reduction, act, bias, num_cab):
        super(HFPH_LKA, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1], fusion_dim, s_factor=2)
        self.up_dec1 = UpSample(n_feat[1], fusion_dim, s_factor=2)

        self.up_enc2 = UpSample(n_feat[2], fusion_dim, s_factor=4)
        self.up_dec2 = UpSample(n_feat[2], fusion_dim, s_factor=4)

        self.up_enc3 = UpSample(n_feat[3], fusion_dim, s_factor=8)
        self.up_dec3 = UpSample(n_feat[3], fusion_dim, s_factor=8)

        layer0 = []
        for i in range(2):
            layer0.append(CALayer(fusion_dim, 16))
            layer0.append(LKAAttention(fusion_dim))
        self.conv_enc0 = nn.Sequential(*layer0)

        layer1 = []
        for i in range(2):
            layer1.append(CALayer(fusion_dim, 16))
            layer1.append(LKAAttention(fusion_dim))
        self.conv_enc1 = nn.Sequential(*layer1)

        layer2 = []
        for i in range(2):
            layer2.append(CALayer(fusion_dim, 16))
            layer2.append(LKAAttention(fusion_dim))
        self.conv_enc2 = nn.Sequential(*layer2)

        layer3 = []
        for i in range(2):
            layer3.append(CALayer(fusion_dim, 16))
            layer3.append(LKAAttention(fusion_dim))
        self.conv_enc3 = nn.Sequential(*layer3)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0] + decoder_outs[0])
        x = self.refine0(x)

        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]) + self.up_dec1(decoder_outs[1]))
        x = self.refine1(x)

        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]) + self.up_dec2(decoder_outs[2]))
        x = self.refine2(x)

        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]) + self.up_dec3(decoder_outs[3]))
        x = self.refine3(x)

        return x


class HFPH_EPSA(nn.Module):
    def __init__(self, n_feat, fusion_dim, kernel_size, reduction, act, bias, num_cab):
        super(HFPH_EPSA, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1], fusion_dim, s_factor=2)
        self.up_dec1 = UpSample(n_feat[1], fusion_dim, s_factor=2)

        self.up_enc2 = UpSample(n_feat[2], fusion_dim, s_factor=4)
        self.up_dec2 = UpSample(n_feat[2], fusion_dim, s_factor=4)

        self.up_enc3 = UpSample(n_feat[3], fusion_dim, s_factor=8)
        self.up_dec3 = UpSample(n_feat[3], fusion_dim, s_factor=8)

        layer0 = []
        for i in range(2):
            layer0.append(CALayer(fusion_dim, 16))
            layer0.append(EPSA_BLOCK(fusion_dim, 16))
        self.conv_enc0 = nn.Sequential(*layer0)

        layer1 = []
        for i in range(2):
            layer1.append(CALayer(fusion_dim, 16))
            layer1.append(EPSA_BLOCK(fusion_dim, 16))
        self.conv_enc1 = nn.Sequential(*layer1)

        layer2 = []
        for i in range(2):
            layer2.append(CALayer(fusion_dim, 16))
            layer2.append(EPSA_BLOCK(fusion_dim, 16))
        self.conv_enc2 = nn.Sequential(*layer2)

        layer3 = []
        for i in range(2):
            layer3.append(CALayer(fusion_dim, 16))
            layer3.append(EPSA_BLOCK(fusion_dim, 16))
        self.conv_enc3 = nn.Sequential(*layer3)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0] + decoder_outs[0])
        x = self.refine0(x)

        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]) + self.up_dec1(decoder_outs[1]))
        x = self.refine1(x)

        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]) + self.up_dec2(decoder_outs[2]))
        x = self.refine2(x)

        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]) + self.up_dec3(decoder_outs[3]))
        x = self.refine3(x)

        return x


class HFPH_ESA(nn.Module):
    def __init__(self, n_feat, fusion_dim, kernel_size, reduction, act, bias, num_cab):
        super(HFPH_ESA, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1], fusion_dim, s_factor=2)
        self.up_dec1 = UpSample(n_feat[1], fusion_dim, s_factor=2)

        self.up_enc2 = UpSample(n_feat[2], fusion_dim, s_factor=4)
        self.up_dec2 = UpSample(n_feat[2], fusion_dim, s_factor=4)

        self.up_enc3 = UpSample(n_feat[3], fusion_dim, s_factor=8)
        self.up_dec3 = UpSample(n_feat[3], fusion_dim, s_factor=8)

        layer0 = []
        for i in range(2):
            layer0.append(CALayer(fusion_dim, 16))
            layer0.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
        self.conv_enc0 = nn.Sequential(*layer0)

        layer1 = []
        for i in range(2):
            layer1.append(CALayer(fusion_dim, 16))
            layer1.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
        self.conv_enc1 = nn.Sequential(*layer1)

        layer2 = []
        for i in range(2):
            layer2.append(CALayer(fusion_dim, 16))
            layer2.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
        self.conv_enc2 = nn.Sequential(*layer2)

        layer3 = []
        for i in range(2):
            layer3.append(CALayer(fusion_dim, 16))
            layer3.append(ESA(n_feats=fusion_dim, conv=nn.Conv2d))
        self.conv_enc3 = nn.Sequential(*layer3)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0] + decoder_outs[0])
        x = self.refine0(x)

        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]) + self.up_dec1(decoder_outs[1]))
        x = self.refine1(x)

        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]) + self.up_dec2(decoder_outs[2]))
        x = self.refine2(x)

        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]) + self.up_dec3(decoder_outs[3]))
        x = self.refine3(x)

        return x


class CNN_Transformer_out_Fusion(nn.Module):

    def __init__(self, dim, factor=4):
        super(CNN_Transformer_out_Fusion, self).__init__()

        self.ca_layer = CALayer(dim * 2)
        self.conv11 = nn.Conv2d(dim * 2, dim, 1)
        self.dwconv33 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.R_conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.R_conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.R_conv1 = nn.Conv2d(dim, dim, 1)

        self.L_map = nn.Sequential(
            nn.Conv2d(dim, dim // factor, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim // factor, dim, 1, 1, 0),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, CNN_input, Transformer_input):
        C = CNN_input.clone()
        T = Transformer_input.clone()
        F_cat = torch.cat([CNN_input, Transformer_input], dim=1)
        F_cat = self.ca_layer(F_cat)
        F = self.conv11(F_cat)
        F = self.dwconv33(F)

        R_out = self.R_conv0(F)
        R_out = self.R_conv_spatial(R_out)
        R_out = self.R_conv1(R_out)
        C_out = R_out * C

        L_out = self.L_map(F)
        T_out = L_out * T

        cat_out = torch.cat([C_out, T_out], dim=1)
        out = cat_out + F_cat
        out = self.out_proj(out)

        return out


class HFPH_CTOF(nn.Module):
    def __init__(self, n_feat, fusion_dim, kernel_size, reduction, act, bias, num_cab):
        super(HFPH_CTOF, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1], fusion_dim, s_factor=2)
        self.up_dec1 = UpSample(n_feat[1], fusion_dim, s_factor=2)

        self.up_enc2 = UpSample(n_feat[2], fusion_dim, s_factor=4)
        self.up_dec2 = UpSample(n_feat[2], fusion_dim, s_factor=4)

        self.up_enc3 = UpSample(n_feat[3], fusion_dim, s_factor=8)
        self.up_dec3 = UpSample(n_feat[3], fusion_dim, s_factor=8)

        # layer0 = []
        # for i in range(2):
        #     layer0.append(CALayer(fusion_dim, 16))
        #     layer0.append(SALayer(fusion_dim, 16))
        # self.conv_enc0 = nn.Sequential(*layer0)
        self.conv_enc0 = CNN_Transformer_out_Fusion(dim=fusion_dim)

        # layer1 = []
        # for i in range(2):
        #     layer1.append(CALayer(fusion_dim, 16))
        #     layer1.append(SALayer(fusion_dim, 16))
        # self.conv_enc1 = nn.Sequential(*layer1)
        self.conv_enc1 = CNN_Transformer_out_Fusion(dim=fusion_dim)

        # layer2 = []
        # for i in range(2):
        #     layer2.append(CALayer(fusion_dim, 16))
        #     layer2.append(SALayer(fusion_dim, 16))
        # self.conv_enc2 = nn.Sequential(*layer2)
        self.conv_enc2 = CNN_Transformer_out_Fusion(dim=fusion_dim)

        # layer3 = []
        # for i in range(2):
        #     layer3.append(CALayer(fusion_dim, 16))
        #     layer3.append(SALayer(fusion_dim, 16))
        # self.conv_enc3 = nn.Sequential(*layer3)
        self.conv_enc3 = CNN_Transformer_out_Fusion(dim=fusion_dim)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0], decoder_outs[0])
        x = self.refine0(x)

        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]), self.up_dec1(decoder_outs[1]))
        x = self.refine1(x)

        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]), self.up_dec2(decoder_outs[2]))
        x = self.refine2(x)

        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]), self.up_dec3(decoder_outs[3]))
        x = self.refine3(x)

        return x


class HFPH(nn.Module):
    def __init__(self, n_feat, fusion_dim, kernel_size, reduction, act, bias, num_cab):
        super(HFPH, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1], fusion_dim, s_factor=2)
        self.up_dec1 = UpSample(n_feat[1], fusion_dim, s_factor=2)

        self.up_enc2 = UpSample(n_feat[2], fusion_dim, s_factor=4)
        self.up_dec2 = UpSample(n_feat[2], fusion_dim, s_factor=4)

        self.up_enc3 = UpSample(n_feat[3], fusion_dim, s_factor=8)
        self.up_dec3 = UpSample(n_feat[3], fusion_dim, s_factor=8)

        layer0 = []
        for i in range(2):
            layer0.append(CALayer(fusion_dim, 16))
            layer0.append(SALayer(fusion_dim, 16))
        self.conv_enc0 = nn.Sequential(*layer0)

        layer1 = []
        for i in range(2):
            layer1.append(CALayer(fusion_dim, 16))
            layer1.append(SALayer(fusion_dim, 16))
        self.conv_enc1 = nn.Sequential(*layer1)

        layer2 = []
        for i in range(2):
            layer2.append(CALayer(fusion_dim, 16))
            layer2.append(SALayer(fusion_dim, 16))
        self.conv_enc2 = nn.Sequential(*layer2)

        layer3 = []
        for i in range(2):
            layer3.append(CALayer(fusion_dim, 16))
            layer3.append(SALayer(fusion_dim, 16))
        self.conv_enc3 = nn.Sequential(*layer3)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0] + decoder_outs[0])
        x = self.refine0(x)

        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]) + self.up_dec1(decoder_outs[1]))
        x = self.refine1(x)

        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]) + self.up_dec2(decoder_outs[2]))
        x = self.refine2(x)

        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]) + self.up_dec3(decoder_outs[3]))
        x = self.refine3(x)

        return x


class Transformer_block(nn.Module):
    def __init__(self, dim, num_head=8, groups=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 FFN_expand=2, norm_layer='WithBias', act_layer=nn.GELU, l_drop=0., mlp_ratio=2, drop_path=0., sr=1):
        super(Transformer_block, self).__init__()
        self.dim = dim * 2
        self.num_head = num_head

        self.norm1 = LayerNorm(self.dim // 2, norm_layer)
        self.norm3 = LayerNorm(self.dim // 2, norm_layer)

        self.attn_nn = Attention(dim=self.dim // 2, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr)

        self.ffn = MFFN(self.dim // 2, FFN_expand=2, norm_layer=nn.LayerNorm)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        ind = x
        b, c, h, w = x.shape
        b, c, h, w = x.shape
        x = self.attn_nn(self.norm1(x).reshape(b, c, h * w).transpose(1, 2), h, w)
        b, c, h, w = x.shape
        x = self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        return x


class GLFusion(nn.Module):
    def __init__(self, dim):
        super(GLFusion, self).__init__()
        self.mix = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, Local_feat, Global_feat):
        Global_weight = torch.sigmoid(Global_feat)
        Global_feat = Global_weight * Global_feat
        Local_weight = torch.sigmoid(Local_feat)
        Local_feat = Local_weight * Local_weight

        Local2Global = torch.sigmoid(Global_feat)
        Global2Local = torch.sigmoid(Local_feat)

        Local_feat = Local2Global * Local_feat
        Global_feat = Global2Local * Global2Local

        return self.mix(Local_feat * Global_feat)


class Baseline_Finetune(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_cahnnels=3,
                 transformer_blocks=8,
                 dim=[16, 32, 64, 128, 256],
                 window_size=[8, 8, 8, 8],
                 patch_size=64,
                 swin_num=[4, 6, 7, 8],
                 head=[1, 2, 4, 8, 16],
                 FFN_expand_=2,
                 qkv_bias_=False,
                 qk_sacle_=None,
                 attn_drop_=0.,
                 proj_drop_=0.,
                 norm_layer_='WithBias',
                 act_layer_=nn.GELU,
                 l_drop_=0.,
                 drop_path_=0.,
                 sr=1,
                 wavevit_num_heads=[2, 4, 8, 8],
                 wavevit_mlp_ratios=[8, 8, 4, 4],
                 wavevit_sr_ratio=[4, 2, 1, 1],
                 wavevit_depths=[2, 2, 2, 2],
                 conv_num=[4, 6, 7, 8],
                 expand_ratio=[1, 2, 2, 2],
                 VAN=False,
                 dynamic_e=False,
                 dynamic_d=False,
                 global_atten=True,
                 model_name="vit_b",  # "vit_b"  "vit_l"  "vit_h"
                 checkpoint="/mnt/Temp_new/dxw/code/SAM_desnow/exp_SAM_desnow/SAM_ckpt/sam_vit_b_01ec64.pth",
                 # "sam_vit_b_01ec64.pth" "sam_vit_l_0b3195.pth" "sam_vit_h_4b8939.pth"
                 SAM_image_size = 256,
                 ):
        super(Baseline_Finetune, self).__init__()
        self.patch_size = patch_size
        # self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.SAM_Image_Encoder = SAM_Image_Encoder(model_name=model_name, image_size=SAM_image_size,
                                                   checkpoint=checkpoint)

        self.embed = Down(in_channels, dim[0], 3, 1, 1)
        # self.wavevit_embed = Down(in_channels, dim[0], 3, 1, 1)
        # self.conv0 = nn.Conv2d(dim[0], dim[4], 1)
        self.encoder_level0 = nn.Sequential(
            Conv_block(conv_num[0], dim[0], dim[0], expand_ratio=expand_ratio[0], VAN=VAN, dynamic=dynamic_e))
        #self.sam_conv0 = nn.Conv2d(768, dim[0], 1)
        # self.wavevit_encoder_lever0 = nn.Sequential(
        #     *[superwavevit_xw_Block(dim=dim[0], num_heads=wavevit_num_heads[0], mlp_ratio=wavevit_mlp_ratios[0], sr_ratio=wavevit_sr_ratio[0])
        #       for i in range(wavevit_depths[0])])
        # self.wave_conv0 = nn.Conv2d(dim[0], dim[4], 1)
        # self.fusion_level0_reduceC = nn.Conv2d(dim[0] * 2, dim[0], 1)
        # self.GL_fusion_0 = GLFusion(dim=dim[0])

        self.down0 = Down(dim[0], dim[1], 3, 2, 1)  ## H//2,W//2
        # self.wavevit_down0 = Down(dim[0], dim[1], 3, 2, 1)
        # self.conv1 = nn.Conv2d(dim[1], dim[4], 1)
        self.encoder_level1 = nn.Sequential(
            Conv_block(conv_num[1], dim[1], dim[1], expand_ratio=expand_ratio[1], VAN=VAN, dynamic=dynamic_e))
        #self.sam_conv1 = nn.Conv2d(768, dim[1], 1)
        # self.wavevit_encoder_lever1 = nn.Sequential(
        #     *[superwavevit_xw_Block(dim=dim[1], num_heads=wavevit_num_heads[1], mlp_ratio=wavevit_mlp_ratios[1], sr_ratio=wavevit_sr_ratio[1])
        #       for i in range(wavevit_depths[1])])
        # self.wave_conv1 = nn.Conv2d(dim[1], dim[4], 1)
        # self.fusion_level1_reduceC = nn.Conv2d(dim[1] * 2, dim[1], 1)
        # self.GL_fusion_1 = GLFusion(dim=dim[1])

        self.down1 = Down(dim[1], dim[2], 3, 2, 1)  ## H//4,W//4
        # self.wavevit_down1 = Down(dim[1], dim[2], 3, 2, 1)
        # self.conv2 = nn.Conv2d(dim[2], dim[4], 1)
        self.encoder_level2 = nn.Sequential(
            Conv_block(conv_num[2], dim[2], dim[2], expand_ratio=expand_ratio[2], VAN=VAN, dynamic=dynamic_e))
        #self.sam_conv2 = nn.Conv2d(768, dim[2], 1)
        # self.wavevit_encoder_lever2 = nn.Sequential(
        #     *[superwavevit_xw_Block(dim=dim[2], num_heads=wavevit_num_heads[2], mlp_ratio=wavevit_mlp_ratios[2],
        #                     sr_ratio=wavevit_sr_ratio[2])
        #       for i in range(wavevit_depths[2])])
        # self.wave_conv2 = nn.Conv2d(dim[2], dim[4], 1)
        # self.fusion_level2_reduce2 = nn.Conv2d(dim[2] * 2, dim[2], 1)
        # self.GL_fusion_2 = GLFusion(dim=dim[2])

        self.down2 = Down(dim[2], dim[3], 3, 2, 1)  ## H//8,W//8
        # self.wavevit_down2 = Down(dim[2], dim[3], 3, 2, 1)
        # self.conv3 = nn.Conv2d(dim[3], dim[4],1)
        self.encoder_level3 = nn.Sequential(
            Conv_block(conv_num[3], dim[3], dim[3], expand_ratio=expand_ratio[3], VAN=VAN, dynamic=dynamic_e))
        #self.sam_conv3 = nn.Conv2d(768, dim[3], 1)
        # self.wavevit_encoder_lever3 = nn.Sequential(
        #     *[superwavevit_xw_Block(dim=dim[3], num_heads=wavevit_num_heads[3], mlp_ratio=wavevit_mlp_ratios[3],
        #                     sr_ratio=wavevit_sr_ratio[3])
        #       for i in range(wavevit_depths[3])])
        # self.wave_conv3 = nn.Conv2d(dim[3], dim[4], 1)
        # self.fusion_level3_reduceC = nn.Conv2d(dim[3] * 2, dim[3], 1)
        # self.GL_fusion_3 = GLFusion(dim=dim[3])

        self.down3 = Down(dim[3], dim[4], 3, 2, 1)  ## H//16,W//16
        # self.wavevit_down3 = Down(dim[3], dim[4], 3, 2, 1) ## H//16,W//16
        # self.bottle_neck_ca = CALayer(dim[4]*2, reduction=4)

        # self.CNN_SAM_Cross = Cross_Transformer_Layer(n=4, dim=dim[4], Query_dim=dim[4])
        # self.CQ_SAM_CNN = CQ_Restormer_Layer(n=4, dim=dim[4], query=dim[4],
        #                                                  num_heads=wavevit_num_heads[3],
        #                                                  ffn_expansion_factor=2.66, bias=True,
        #                                                  LayerNorm_type='WithBias')

        self.reduce_chan_bottle_neck = nn.Conv2d(dim[4] * 2, dim[4], kernel_size=1, bias=False)

        self.latent = nn.Sequential(*[
            Transformer_block(dim=(dim[4]), num_head=head[4], qkv_bias=qkv_bias_, qk_scale=qk_sacle_,
                              attn_drop=attn_drop_, proj_drop=proj_drop_, FFN_expand=FFN_expand_,
                              norm_layer=norm_layer_, act_layer=act_layer_, l_drop=l_drop_, drop_path=drop_path_, sr=sr)
            for i in range(transformer_blocks)])

        self.up3 = Up((dim[4]), dim[3], 4, 2, 1)
        self.ca3 = CALayer(dim[3] * 2, reduction=4)
        self.reduce_chan_level3 = nn.Conv2d(dim[3] * 2, dim[3], kernel_size=1, bias=False)
        self.decoder_level3 = Context_Interaction_layer(n=swin_num[3], latent_dim=dim[4], in_channel=dim[3],
                                                        head=head[3], window_size=window_size[3],
                                                        globalatten=global_atten)

        self.up2 = Up(dim[3], dim[2], 4, 2, 1)
        self.ca2 = CALayer(dim[2] * 2, reduction=4)
        self.reduce_chan_level2 = nn.Conv2d(dim[2] * 2, dim[2], kernel_size=1, bias=False)
        self.decoder_level2 = Context_Interaction_layer(n=swin_num[2], latent_dim=dim[4], in_channel=dim[2],
                                                        head=head[2], window_size=window_size[2],
                                                        globalatten=global_atten)

        self.up1 = Up(dim[2], dim[1], 4, 2, 1)
        self.ca1 = CALayer(dim[1] * 2, reduction=4)
        self.reduce_chan_level1 = nn.Conv2d(dim[1] * 2, dim[1], kernel_size=1, bias=False)
        self.decoder_level1 = Context_Interaction_layer(n=swin_num[1], latent_dim=dim[4], in_channel=dim[1],
                                                        head=head[1], window_size=window_size[1],
                                                        globalatten=global_atten)

        self.up0 = Up(dim[1], dim[0], 4, 2, 1)
        self.ca0 = CALayer(dim[0] * 2, reduction=4)
        self.reduce_chan_level0 = nn.Conv2d(dim[0] * 2, dim[0], kernel_size=1, bias=False)
        self.decoder_level0 = Context_Interaction_layer(n=swin_num[0], latent_dim=dim[4], in_channel=dim[0],
                                                        head=head[0], window_size=window_size[0],
                                                        globalatten=global_atten)

        # self.refinement = HFPH(n_feat=dim, fusion_dim=dim[0], kernel_size=3, reduction=4, act=nn.GELU(), bias=True,
        #                        num_cab=6)
        # self.refinement = HFPH_EPSA(n_feat=dim, fusion_dim=dim[0], kernel_size=3, reduction=4, act=nn.GELU(), bias=True,
        #                        num_cab=6)
        # self.refinement = HFPH_ESA(n_feat=dim, fusion_dim=dim[0], kernel_size=3, reduction=4, act=nn.GELU(), bias=True,
        #                        num_cab=6)
        # self.refinement = HFPH_LKA(n_feat=dim, fusion_dim=dim[0], kernel_size=3, reduction=4, act=nn.GELU(), bias=True,
        #                            num_cab=6)
        # self.refinement = HFPH_CTOF(n_feat=dim, fusion_dim=dim[0], kernel_size=3, reduction=4, act=nn.GELU(), bias=True,
        #                             num_cab=6)
        # self.refinement = HFPH_ThreeOTF(n_feat=dim, fusion_dim=dim[0], kernel_size=3, reduction=4, act=nn.GELU(), bias=True,
        #                            num_cab=6)
        self.out2 = nn.Conv2d(dim[0], out_cahnnels, kernel_size=3, stride=1, padding=1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        x = self.check_image_size(x)
        sam_x = x

        # cnn_encoder_item = []
        # sam_encoder_item = []

        # x_0_1, x_2_3_4, x_5_6_7, x_8_9_10_11, x_out = self.SAM_Image_Encoder(sam_x)
        # x_0_1 = x_0_1.permute(0, 3, 1, 2)
        # x_2_3_4 = x_2_3_4.permute(0, 3, 1, 2)
        # x_5_6_7 = x_5_6_7.permute(0, 3, 1, 2)
        # x_8_9_10_11 = x_8_9_10_11.permute(0, 3, 1, 2)

        _, _, _, _, x_out = self.SAM_Image_Encoder(sam_x)


        inp_enc_level0 = self.embed(x)
        inp_enc_level0 = self.encoder_level0(inp_enc_level0)
        # SAM_enc_level0 = F.upsample(x_0_1, (inp_enc_level0.shape[2], inp_enc_level0.shape[3]), mode="bicubic")
        # SAM_enc_level0 = self.sam_conv0(SAM_enc_level0)
        # cnn_encoder_item.append(inp_enc_level0)
        # sam_encoder_item.append(SAM_enc_level0)

        inp_enc_level1 = self.down0(inp_enc_level0)
        inp_enc_level1 = self.encoder_level1(inp_enc_level1)
        # SAM_enc_level1 = F.upsample(x_2_3_4, (inp_enc_level1.shape[2], inp_enc_level1.shape[3]), mode="bicubic")
        # SAM_enc_level1 = self.sam_conv1(SAM_enc_level1)
        # cnn_encoder_item.append(inp_enc_level1)
        # sam_encoder_item.append(SAM_enc_level1)

        inp_enc_level2 = self.down1(inp_enc_level1)
        inp_enc_level2 = self.encoder_level2(inp_enc_level2)
        # SAM_enc_level2 = F.upsample(x_5_6_7, (inp_enc_level2.shape[2], inp_enc_level2.shape[3]), mode="bicubic")
        # SAM_enc_level2 = self.sam_conv2(SAM_enc_level2)
        # cnn_encoder_item.append(inp_enc_level2)
        # sam_encoder_item.append(SAM_enc_level2)

        inp_enc_level3 = self.down2(inp_enc_level2)
        inp_enc_level3 = self.encoder_level3(inp_enc_level3)
        # SAM_enc_level3 = F.upsample(x_8_9_10_11, (inp_enc_level3.shape[2], inp_enc_level3.shape[3]), mode="bicubic")
        # SAM_enc_level3 = self.sam_conv3(SAM_enc_level3)
        # cnn_encoder_item.append(inp_enc_level3)
        # sam_encoder_item.append(SAM_enc_level3)

        out_enc_level4 = self.down3(inp_enc_level3)

        # SAM_out = self.CNN_SAM_Cross(SAM_out, out_enc_level4)

        latent = torch.cat([out_enc_level4, x_out], 1)

        latent = self.reduce_chan_bottle_neck(latent)
        latent = self.latent(latent)

        inp_dec_level3 = self.up3(latent)
        inp_dec_level3 = F.upsample(inp_dec_level3, (inp_enc_level3.shape[2], inp_enc_level3.shape[3]), mode="bicubic")
        inp_dec_level3 = torch.cat([inp_dec_level3, inp_enc_level3], 1)
        inp_dec_level3 = self.ca3(inp_dec_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, latent)
        out_dec_level3 = _to_channel_first(out_dec_level3)

        inp_dec_level2 = self.up2(out_dec_level3)
        inp_dec_level2 = F.upsample(inp_dec_level2, (inp_enc_level2.shape[2], inp_enc_level2.shape[3]), mode="bicubic")
        inp_dec_level2 = torch.cat([inp_dec_level2, inp_enc_level2], 1)
        inp_dec_level2 = self.ca2(inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, latent)
        out_dec_level2 = _to_channel_first(out_dec_level2)

        inp_dec_level1 = self.up1(out_dec_level2)
        inp_dec_level1 = F.upsample(inp_dec_level1, (inp_enc_level1.shape[2], inp_enc_level1.shape[3]), mode="bicubic")
        inp_dec_level1 = torch.cat([inp_dec_level1, inp_enc_level1], 1)
        inp_dec_level1 = self.ca1(inp_dec_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, latent)
        out_dec_level1 = _to_channel_first(out_dec_level1)

        inp_dec_level0 = self.up0(out_dec_level1)
        inp_dec_level0 = F.upsample(inp_dec_level0, (inp_enc_level0.shape[2], inp_enc_level0.shape[3]), mode="bicubic")
        inp_dec_level0 = torch.cat([inp_dec_level0, inp_enc_level0], 1)
        inp_dec_level0 = self.ca0(inp_dec_level0)
        inp_dec_level0 = self.reduce_chan_level0(inp_dec_level0)
        out_dec_level0 = self.decoder_level0(inp_dec_level0, latent)
        out_dec_level0 = _to_channel_first(out_dec_level0)

        # out_dec_level0_refine = self.refinement(out_dec_level0, cnn_encoder_item, sam_encoder_item)
        # out_dec_level0_refine = self.refinement(out_dec_level0, item)
        out_dec_level1 = self.out2(out_dec_level0) + x

        return out_dec_level1


from ptflops import get_model_complexity_info

# model = Baseline_Finetune().cuda()
# input_img = torch.rand((1, 3, 128, 128)).cuda()
# output_img = model(input_img)
# print("output_img", output_img.shape)
# #
# from ptflops import get_model_complexity_info
#
# model = Baseline_Finetune().cuda()
# H, W = 256, 256
# flops_t, params_t = get_model_complexity_info(model, (3, H, W), as_strings=True, print_per_layer_stat=True)
# print(f"net flops:{flops_t} parameters:{params_t}")
#
# # 参数量计算方式2
# total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total Trainable Parameters: {total_trainable_params / 1000000} M")
#
# total_no_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
# print(f"No Trainable Parameters: {total_no_trainable_params / 1000000} M")
#
# Total_Params = (total_trainable_params + total_no_trainable_params)
# print(f"Total Parameters: {Total_Params / 1000000} M")


# model = Desnow_Baseline_SIE_LoRa_cat_SkipCupsample().cuda()
# H, W = 128, 128
# flops_t, params_t = get_model_complexity_info(model, (3, H, W), as_strings=True, print_per_layer_stat=True)
#
# print(f"net flops:{flops_t} parameters:{params_t}")
# model = nn.DataParallel(model)
# x = torch.ones([1,3,H,W]).cuda()
# b = model(x)
# steps=25   #25
# # print(b)
# time_avgs=[]
# with torch.no_grad():
#     for step in range(steps):
#
#         torch.cuda.synchronize()
#         start = time.time()
#         result = model(x)
#         torch.cuda.synchronize()
#         time_interval = time.time() - start
#         if step>5:
#             time_avgs.append(time_interval)
#         print('run time:',time_interval)
# print('avg time:', np.mean(time_avgs), 'fps:', (1/np.mean(time_avgs)),' size:',H,W)