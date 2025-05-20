import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from mamba_ssm import Mamba


class Hamburger(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Hamburger, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool_w = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_d = nn.AdaptiveAvgPool3d((None, None, 1))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(8, mip)
        self.gn2 = nn.GroupNorm(8, mip)
        self.gn3 = nn.GroupNorm(8, mip)

        self.act = nn.LeakyReLU(0.2)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w, d = x.size()
        x_h = self.pool_h(x)
        # print(x_h.shape)
        x_w = self.pool_w(x).permute(0, 1, 3, 2, 4)
        # print(x_w.shape)
        x_d = self.pool_d(x).permute(0, 1, 4, 2, 3)
        # print(x_d.shape)
        y_hwd = torch.cat([x_h, x_w, x_d], dim=2)
        # y_hd = torch.cat([x_h, x_d], dim=2)
        # y_dw = torch.cat([x_d, x_w], dim=2)
        y_hwd = self.conv1(y_hwd)
        # y_hd = self.conv2(y_hd)
        # y_dw = self.conv3(y_dw)
        y_hwd = self.gn1(y_hwd)
        # y_hd = self.gn2(y_hd)
        # y_dw = self.gn3(y_dw)
        y_hwd = self.act(y_hwd)
        # y_hd = self.act(y_hd)
        # y_dw = self.act(y_dw)
        # print(y_hwd.shape)
        x_h, x_w, x_d = torch.split(y_hwd, [1, 1, 1], dim=2)
        x_w = x_w
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_d = x_d.permute(0, 1, 3, 4, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_d = self.conv_d(x_d).sigmoid()
        a_hw = a_w * a_h
        out = a_hw * a_d
        return out + x


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, input_channels=4, base_channels=16, feature_dim=512):
        super(ResNet3D, self).__init__()
        self.in_channels = base_channels

        self.conv1 = nn.Conv3d(input_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 每个layer的通道数是基于 base_channels 乘以扩展因子
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, feature_dim)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: (B, C, D, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # -> (B, C, D/4, H/4, W/4)
        x = self.layer2(x)  # -> (B, 2C, D/8, H/8, W/8)
        x = self.layer3(x)  # -> (B, 4C, D/16, H/16, W/16)
        x = self.layer4(x)  # -> (B, 8C, D/32, H/32, W/32)

        x = self.avgpool(x)  # -> (B, 8C, 1, 1, 1)
        x = torch.flatten(x, 1)  # -> (B, 8C)
        x = self.fc(x)  # -> (B, feature_dim)

        return x

def ResNet3D34(input_channels=4, base_channels=16, feature_dim=512):
    """Constructs a ResNet-34 3D model."""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3],
                   input_channels=input_channels,
                   base_channels=base_channels,
                   feature_dim=feature_dim)

def ResNet3D50(input_channels=4, base_channels=16, feature_dim=512):
    """Constructs a ResNet-50 3D model."""
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3],
                   input_channels=input_channels,
                   base_channels=base_channels,
                   feature_dim=feature_dim)


class DenseLayer3D(nn.Module):
    """DenseNet3D 的基本层，包括批归一化、激活和卷积操作"""
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.drop_rate > 0:
            out = F.dropout3d(out, p=self.drop_rate, training=self.training)
        
        # 将输入和输出在通道维度上拼接
        out = torch.cat([x, out], 1)
        return out


class DenseBlock3D(nn.Module):
    """由多个 DenseLayer3D 组成的 DenseBlock"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock3D, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer3D(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            ))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Transition3D(nn.Module):
    """用于减少特征图的尺寸和通道数的过渡层"""
    def __init__(self, in_channels, out_channels):
        super(Transition3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out


class DenseNet3D(nn.Module):
    """DenseNet3D 模型"""
    def __init__(self, input_channels=1, base_channels=64, growth_rate=32, block_layers=[3, 6, 12, 8],
                 bn_size=4, drop_rate=0.0, feature_dim=1024):
        super(DenseNet3D, self).__init__()
        self.growth_rate = growth_rate
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            )
        
        # Dense Blocks 和 Transition Layers
        num_features = base_channels
        self.block_layers = []
        self.num_blocks = len(block_layers)
        self.dense_blocks = nn.ModuleList()
        self.trans_blocks = nn.ModuleList()
        
        for i, num_layers in enumerate(block_layers):
            dense_block = DenseBlock3D(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.dense_blocks.append(dense_block)
            num_features = num_features + num_layers * growth_rate
            
            if i != self.num_blocks - 1:
                trans_block = Transition3D(
                    in_channels=num_features,
                    out_channels=num_features // 2
                )
                self.trans_blocks.append(trans_block)
                num_features = num_features // 2

        # 最后一个 batch norm
        self.bn_final = nn.BatchNorm3d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, feature_dim)
        )
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        
        for i in range(self.num_blocks):
            out = self.dense_blocks[i](out)
            if i < self.num_blocks - 1:
                out = self.trans_blocks[i](out)
        
        out = self.bn_final(out)
        out = self.relu_final(out)
        
        out = self.global_pool(out)  # (B, C, 1, 1, 1)
        out = out.view(out.size(0), -1)  # (B, C)
        out = self.classifier(out)  # (B, feature_dim)
        return out

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DepthwiseConv3D(nn.Module):
    """深度可分离卷积（3D）"""
    def __init__(self, dim, kernel_size=7, padding=3):
        super(DepthwiseConv3D, self).__init__()
        self.dw_conv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)

    def forward(self, x):
        return self.dw_conv(x)


class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt 基本块（3D）"""
    def __init__(self, dim, drop_path=0.0):
        super(ConvNeXtBlock3D, self).__init__()
        self.depthwise_conv = DepthwiseConv3D(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

    def forward(self, x):
        # 输入x形状: (B, C, D, H, W)
        residual = x
        x = self.depthwise_conv(x)
        # 转换为 (B, D, H, W, C) 以应用 LayerNorm 和 Linear
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.drop_path(x)
        # 转换回 (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        x = residual + x  # 残差连接
        return x


class DownSample3D(nn.Module):
    """下采样层（3D）"""
    def __init__(self, in_channels, out_channels):
        super(DownSample3D, self).__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 输入x形状: (B, C, D, H, W)
        # 需要在 LayerNorm 之前转换为 (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer[0](x)  # LayerNorm
        x = x.permute(0, 4, 1, 2, 3)  # 转回 (B, C, D, H, W)
        x = self.layer[1](x)  # Conv3d 下采样
        return x


class ConvNeXt3D(nn.Module):
    """ConvNeXt 模型（3D）"""
    def __init__(self, input_channels=3, base_channels=96, feature_dim=1024,
                 depths=[3, 3, 9, 3], drop_path_rate=0.1, layer_scale_init_value=1e-6):
        super(ConvNeXt3D, self).__init__()
        self.num_stages = len(depths)
        self.drop_path_rate = drop_path_rate

        # 计算每个阶段的通道数
        self.dims = [base_channels * (2 ** i) for i in range(self.num_stages)]

        # Stem 层：卷积下采样
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, self.dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(self.dims[0], eps=1e-6)
        )

        # 将每个阶段的块和下采样层组合
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        total_blocks = sum(depths)
        # 为 DropPath 计算每个块的丢弃概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        block_idx = 0
        for i in range(self.num_stages):
            # ConvNeXt 块
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(dim=self.dims[i], drop_path=dpr[block_idx + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            block_idx += depths[i]
            # 下采样层（除最后一个阶段外）
            if i < self.num_stages - 1:
                self.downsamples.append(DownSample3D(self.dims[i], self.dims[i+1]))

        # 全局池化和分类头
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.norm_head = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.dims[-1], feature_dim)

    def forward(self, x):
        # Stem 层
        x = self.stem(x)

        # 各个阶段
        for i in range(self.num_stages):
            x = self.stages[i](x)
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        # 全局池化
        x = self.global_pool(x)  # (B, C, 1, 1, 1)
        x = x.view(x.shape[0], x.shape[1])  # (B, C)
        x = self.norm_head(x)  # LayerNorm
        x = self.flatten(x)  # (B, C)
        x = self.fc(x)  # (B, feature_dim)
        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding Layer"""
    def __init__(self, input_channels=4, embed_dim=16, patch_size=(4, 8, 8)):
        super(PatchEmbed3D, self).__init__()
        self.patch_size = patch_size  # (D, H, W)
        self.proj = nn.Conv3d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # (B, D', H', W', C)
        x = x.reshape(B, D * H * W, C)  # (B, N, C), N = D'*H'*W' = (64/4)*(64/8)*(64/8)=16*8*8=1024
        x = self.norm(x)
        return x  # (B, N, C)


class TransformerEncoderLayer3D(nn.Module):
    """Standard Transformer Encoder Layer for 3D ViT"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super(TransformerEncoderLayer3D, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # x: (B, N, C)
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)  # attn_output: (B, N, C)
        x = x + self.drop1(attn_output)
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x  # (B, N, C)


class TransformerEncoder3D(nn.Module):
    """Transformer Encoder consisting of multiple TransformerEncoderLayer3D"""
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super(TransformerEncoder3D, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer3D(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x  # (B, N, C)


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer (ViT)"""
    def __init__(self, input_channels=4, base_channels=16, feature_dim=512,
                 patch_size=(8, 8, 8), depth=4, num_heads=4,
                 mlp_ratio=4.0, drop_rate=0.1, attn_drop_rate=0.1):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = PatchEmbed3D(input_channels, base_channels, patch_size)
        # Calculate number of patches
        D_patch, H_patch, W_patch = patch_size
        self.num_patches = (64 // D_patch) * (64 // H_patch) * (64 // W_patch)  # 8*8*8=512

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, base_channels))
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, base_channels))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder
        self.encoder = TransformerEncoder3D(depth, base_channels, num_heads, mlp_ratio, drop_rate, attn_drop_rate)

        # Classification head
        self.norm_head = nn.LayerNorm(base_channels)
        self.fc = nn.Linear(base_channels, feature_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        # Expand CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, C)
        x = x + self.pos_embed  # (B, N+1, C)
        x = self.pos_drop(x)

        x = self.encoder(x)  # (B, N+1, C)

        cls_token_final = x[:, 0]  # (B, C)
        cls_token_final = self.norm_head(cls_token_final)  # (B, C)
        features = self.fc(cls_token_final)  # (B, feature_dim)
        return features  # (B, feature_dim)


class SwinPatchEmbed3D(nn.Module):
    """将3D输入分割成patches并进行嵌入"""

    def __init__(self, patch_size=(4, 4, 4), in_channels=4, embed_dim=16):
        super(SwinPatchEmbed3D, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, D/p, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        D_p, H_p, W_p = x.shape[1] // (H // self.patch_size[1] * W // self.patch_size[2]), H // self.patch_size[1], W // self.patch_size[2]
        return x, (D_p, H_p, W_p)


class WindowAttention3D(nn.Module):
    """3D窗口多头自注意力"""

    def __init__(self, dim, window_size=(7, 7, 7), num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention3D, self).__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )  # 每个相对位置一个bias

        # 生成相对位置的index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords += torch.tensor(self.window_size) - 1  # shift to start from 0
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, N, C)
        mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        # 添加相对位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        out = (attn @ v)  # (B_, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer Block"""

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super(SwinTransformerBlock3D, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size  # Wd_shift, Wh_shift, Ww_shift
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, qkv_bias, attn_drop, drop)

        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    @staticmethod
    def window_partition(x, window_size):
        """
        将输入x (B, D, H, W, C) 分割成窗口
        返回: windows (num_windows*B, window_size D, window_size H, window_size W, C)
        """
        B, D, H, W, C = x.shape
        x = x.view(
            B,
            D // window_size[0], window_size[0],
            H // window_size[1], window_size[1],
            W // window_size[2], window_size[2],
            C
        )
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, *window_size, C)
        return windows

    @staticmethod
    def window_reverse(windows, window_size, B, D, H, W):
        """
        将窗口逆操作合并回原始特征图
        Input:
            windows: (num_windows*B, window_size D, window_size H, window_size W, C)
            window_size: tuple (Wd, Wh, Ww)
            B, D, H, W: 原始体数据的维度
        Output:
            x: (B, D, H, W, C)
        """
        C = windows.shape[-1]
        x = windows.view(
            B,
            D // window_size[0],
            H // window_size[1],
            W // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            C
        )
        x = x.permute(0, 1, 2, 3, 4, 5, 6, 7).contiguous().view(B, D, H, W, C)
        return x
    
    def create_attn_mask(self, input_resolution):
        """创建注意力掩码，用于处理移动窗口边界"""
        D, H, W = input_resolution
        img_mask = torch.zeros((1, D, H, W, 1), device=self.attn.relative_position_bias_table.device)  # (1, D, H, W, 1)
        Wd, Wh, Ww = self.window_size
        sd, sh, sw = self.shift_size

        # 计算每个区块的位置标记
        cnt = 0
        for d in (slice(0, -Wd), slice(-Wd, -sd) if sd > 0 else slice(0, None), slice(-sd, None) if sd > 0 else slice(0, None)):
            for h in (slice(0, -Wh), slice(-Wh, -sh) if sh > 0 else slice(0, None), slice(-sh, None) if sh > 0 else slice(0, None)):
                for w in (slice(0, -Ww), slice(-Ww, -sw) if sw > 0 else slice(0, None), slice(-sw, None) if sw > 0 else slice(0, None)):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        # 分割为窗口
        mask_windows = self.window_partition(img_mask, self.window_size)  # (num_windows, Wd, Wh, Ww, 1)
        mask_windows = mask_windows.view(-1, Wd * Wh * Ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, input_resolution):
        """
        x: (B, N, C), N = D*H*W
        input_resolution: tuple (D, H, W)
        """
        D, H, W = input_resolution
        B, N, C = x.shape
        assert N == D * H * W, "Input has incorrect size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # 1. Shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # 2. Window partition
        x_windows = self.window_partition(shifted_x, self.window_size)  # (num_windows*B, Wd, Wh, Ww, C)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # (num_windows*B, Nw, C)

        # 3. Window Multi-Head Self-Attention
        if any(s > 0 for s in self.shift_size):
            attn_mask = self.create_attn_mask(input_resolution)
        else:
            attn_mask = None

        attn_windows = self.attn(x_windows, mask=attn_mask)  # (num_windows*B, Nw, C)

        # 4. Merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, B, D, H, W)  # (B, D, H, W, C)

        # 5. Reverse shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)

        # 6. MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging3D(nn.Module):
    """3D Patch Merging"""

    def __init__(self, dim):
        super(PatchMerging3D, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim * 8, 2 * dim, bias=False)  # 2x reduction in resolution
        self.norm = nn.LayerNorm(dim * 8)

    def forward(self, x, input_resolution):
        # x: (B, N, C), N = D*H*W
        B, N, C = x.shape
        D, H, W = input_resolution
        assert N == D * H * W, "Input has incorrect size"

        x = x.view(B, D, H, W, C)

        # 2x2x2_merge for D, H, W
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, D/2, H/2, W/2, 8*C)
        x = x.view(B, -1, 8 * C)  # (B, D/2 * H/2 * W/2, 8*C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, D/2 * H/2 * W/2, 2*C)

        return x


class SwinTransformerStage3D(nn.Module):
    """一个Swin Transformer阶段，包括多个Block和Patch Merging"""

    def __init__(self, dim, depth, num_heads, window_size=(7, 7, 7), mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=None, downsample=True):
        super(SwinTransformerStage3D, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = (
                window_size[0] // 2 if (i % 2 == 1 and window_size[0] > 1) else 0,
                window_size[1] // 2 if (i % 2 == 1 and window_size[1] > 1) else 0,
                window_size[2] // 2 if (i % 2 == 1 and window_size[2] > 1) else 0,
            )
            block = SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            self.blocks.append(block)

        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x, input_resolution):
        """
        x: (B, N, C)
        input_resolution: tuple (D, H, W)
        """
        for blk in self.blocks:
            x = blk(x, input_resolution)
        if self.downsample:
            x = self.downsample(x, input_resolution)
            D, H, W = input_resolution
            input_resolution = (D // 2, H // 2, W // 2)
        return x, input_resolution


class SwinTransformer3D(nn.Module):
    """3D Swin Transformer模型"""

    def __init__(self, input_channels=4, base_channels=16, feature_dim=512,
                 patch_size=(4, 4, 4),
                 depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16],
                 window_size=(2, 2, 2), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super(SwinTransformer3D, self).__init__()
        self.num_layers = len(depths)
        self.embed_dim = base_channels
        self.patch_size = patch_size
        self.patch_embed = SwinPatchEmbed3D(patch_size, in_channels=input_channels, embed_dim=self.embed_dim)

        # 计算drop path率
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.layers = nn.ModuleList()
        current_depth = 0
        for i_layer in range(self.num_layers):
            layer = SwinTransformerStage3D(
                dim=base_channels * 2**i_layer,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[current_depth:current_depth + depths[i_layer]],
                downsample=(i_layer < self.num_layers - 1)
            )
            self.layers.append(layer)
            current_depth += depths[i_layer]

        self.norm = nn.LayerNorm(base_channels * 2**(self.num_layers - 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base_channels * 2**(self.num_layers - 1), feature_dim)

    def forward_features(self, x):
        """
        x: (B, C, D, H, W)
        """
        x, input_resolution = self.patch_embed(x)  # (B, num_patches, embed_dim), (D_p, H_p, W_p)
        for layer in self.layers:
            x, input_resolution = layer(x, input_resolution)
        x = self.norm(x)  # (B, num_patches, dim)
        return x

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        x = self.forward_features(x)  # (B, num_patches, dim)
        x = x.transpose(1, 2)  # (B, dim, num_patches)
        x = self.avgpool(x).squeeze(-1)  # (B, dim)
        x = self.head(x)  # (B, feature_dim)
        return x


class VoxPeptide(nn.Module):
    def __init__(self, v_encoder='resnet34', classes=6, channels=16, in_channels=4):
        super().__init__()
        self.classes = classes

        if v_encoder == 'resnet34':
            self.v_encoder = ResNet3D34(input_channels=in_channels, base_channels=channels, feature_dim=512)
        elif v_encoder == 'resnet50':
            self.v_encoder = ResNet3D50(input_channels=in_channels, base_channels=channels, feature_dim=512)
        elif v_encoder == 'densenet':
            self.v_encoder = DenseNet3D(input_channels=in_channels, base_channels=channels, growth_rate=16 if channels < 48 else 32, feature_dim=512)
        elif v_encoder == 'convnext':
            self.v_encoder = ConvNeXt3D(input_channels=in_channels, base_channels=channels, feature_dim=512)
        elif v_encoder == 'vit':
            self.v_encoder = VisionTransformer3D(input_channels=in_channels, base_channels=channels, feature_dim=512)
        elif v_encoder == 'swintf':
            self.v_encoder = SwinTransformer3D(input_channels=in_channels, base_channels=channels, feature_dim=512)
        else:
            raise NotImplementedError(f'\'{v_encoder}\' not implemented')

        self.vox_fc = nn.Linear(512, classes)

    def forward(self, x):
        vox, seq = x
        seq_emb = self.v_encoder(vox)
        pred = self.vox_fc(seq_emb)
        return pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.FloatTensor([10000.0])) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, nheads, d_model, num_layers, out_dim, max_length=50):
        super(TransformerModel, self).__init__()
        
        # 嵌入层，将输入从 (B, N) 转换到 (B, N, embed_dim)
        self.embedding = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # Transformer 编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, 
                                                    nhead=nheads, 
                                                    activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 全局池化（可以根据任务选择不同的聚合方式）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc = nn.Linear(d_model, out_dim)
        
    def forward(self, src):
        """
        src: (B, N)
        """
        # 嵌入
        embedded = self.embedding(src.unsqueeze(-1))  # (B, N, embed_dim)
        embedded = self.pos_encoder(embedded)  # 添加位置编码
        
        # 转置以适应 Transformer (N, B, embed_dim)
        embedded = embedded.permute(1, 0, 2)
        
        # Transformer 编码
        transformer_out = self.transformer_encoder(embedded)  # (N, B, embed_dim)
        
        # 转置回 (B, N, embed_dim)
        transformer_out = transformer_out.permute(1, 0, 2)
        
        # 全局池化，将 (B, N, embed_dim) 转换为 (B, embed_dim)
        pooled = self.global_pool(transformer_out.permute(0, 2, 1)).squeeze(-1)
        
        # 输出层
        output = self.fc(pooled)  # (B, output_dim)
        
        return output


class MambaModel(nn.Module):
    def __init__(self, d_model, out_dim, max_length=30):
        super(MambaModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        self.mamba = Mamba(d_model=d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model * 2, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.pos_encoder(self.linear(x.unsqueeze(-1)))
        y = self.mamba(x)
        y_flip = self.mamba(x.flip([-2])).flip([-2])
        y = torch.cat((y, y_flip), dim=-1)
        y = self.fc(self.global_pool(y.permute(0, 2, 1)).squeeze(-1))
        return y


class SEQ(nn.Module):
    def __init__(self, seq_type='mlp', input_dim=21, hidden_dim=128, out_dim=128, num_layers=2, max_length=30):
        super(SEQ, self).__init__()
        self.seq_type = seq_type
        if seq_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,  # input & output will take batch size as 1 dim (batch, time_step, input_size)
                bidirectional=True
            )
        elif seq_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,  # input & output will take batch size as 1 dim (batch, time_step, input_size)
                bidirectional=True
            )
        elif seq_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,  # input & output will take batch size as 1 dim (batch, time_step, input_size)
                bidirectional=True
            )
        elif seq_type == 'tf':
            self.transformer = TransformerModel(nheads=4, d_model=hidden_dim, num_layers=2, out_dim=out_dim, max_length=max_length)
        elif seq_type == 'mamba':
            self.mamba = MambaModel(d_model=hidden_dim, out_dim=out_dim, max_length=max_length)
        else:
            # nn.Linear(50, 50, bias=False), nn.ReLU(),
            self.rnn = nn.Sequential(nn.Linear(max_length, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, out_dim))
        self.rnn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, seq):
        if self.seq_type == 'mlp':
            return self.rnn(seq.squeeze(1))
        elif self.seq_type == 'tf':
            return self.transformer(seq)
        elif self.seq_type == 'mamba':
            return self.mamba(seq)
        else:
            one_hot_seq = F.one_hot(seq.to(torch.int64), num_classes=21).float()
            r_out = self.rnn(one_hot_seq, None)[0]  # None represents zero initial hidden state
            out = self.rnn_fc(r_out[:, -1, :])
            return out
    # def forward(self, x, seq_lengths):
    #


class SEQPeptide(nn.Module):
    def __init__(self, v_encoder='resnet26', q_encoder='mlp', fusion='mlp', classes=6, attention=None, max_length=30):
        super().__init__()
        self.classes = classes
        # q_encoder could be mlp, gru, rnn, lstm, transformer
        self.q_encoder = SEQ(seq_type=q_encoder, max_length=max_length)

        self.seq_fc = nn.Linear(128, classes)

    def forward(self, x, seq_lengths=None):
        vox, seq = x
        seq_emb = self.q_encoder(seq)
        pred = self.seq_fc(seq_emb)
        return pred


class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7, 128)
        # self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.shape[0], -1)
        return self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x


class ConvNet2D(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(ConvNet2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=2, stride=2)
        # self.pool = nn.AdaptiveAvgPool2d(32)
        self.fc1 = nn.Linear(32 * 3 * 9, 128)
        # self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # x = self.pool(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        return self.fc1(x)


# convnet = ConvNet()
# print(convnet)
class MMPeptide(nn.Module):
    def __init__(self, v_encoder='resnet26', q_encoder='mlp', fusion='mlp', classes=6, attention=None, max_length=30):
        super().__init__()
        if attention == 'hamburger':
            self.attention = Hamburger(2048, 2048)
        else:
            self.attention = None
        # v_encoder could be resnet26 or resnet50
        if v_encoder == 'resnet26':
            self.v_encoder = ResNet3D(Bottleneck3D, [1, 2, 4, 1], self.attention)
            # self.v_encoder = SwinUNETR(img_size=(64, 64, 64), in_channels=3, out_channels=1)
        elif v_encoder == 'resnet50':
            self.v_encoder = ResNet3D(Bottleneck3D, [3, 4, 6, 3], self.attention)
        else:
            raise NotImplementedError

        # q_encoder could be mlp, gru, rnn, lstm, transformer
        self.q_encoder = SEQ(seq_type=q_encoder, max_length=max_length)
        # self.ss_encoder = SEQ(seq_type=q_encoder)
        if fusion == 'mlp':
            self.fusion = nn.Linear(512 * 4 + 256, 256)
            # self.fusion = nn.Linear(192 + 256, classes)
        elif fusion == 'att':
            self.fusion = nn.Linear(512 * 4 + 256, 256)
        else:
            raise NotImplementedError

        # self.vox_fc = nn.Linear(2048, classes)
        # self.seq_fc = nn.Linear(256, classes)
        self.out = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256, classes))
        self.classes = classes

    def forward(self, x, seq_lengths=None):
        vox, seq = x
        # print(vox.shape)
        # print(seq.shape)
        vox_emb = self.v_encoder(vox)
        # print(vox_emb.shape)
        seq_emb = self.q_encoder(seq, seq_lengths)
        # print(seq_emb.shape)
        # ss_emb = self.ss_encoder(second_s)
        fused_feature = torch.cat((seq_emb, vox_emb), dim=1)
        pred = self.fusion(fused_feature)
        pred = self.out(pred)
        # pred1 = self.vox_fc(vox_emb)
        # pred2 = self.seq_fc(seq_emb)
        # return pred, fused_feature
        return pred


class SMPeptide(nn.Module):
    def __init__(self, v_encoder='resnet26', q_encoder='mlp', fusion='mlp', classes=6, attention=None, hidden_dim=256, max_length=30):
        super().__init__()
        self.siamese_encoder1 = MMPeptide(v_encoder, q_encoder, fusion, classes, attention, max_length)
        # self.siamese_encoder2 = MMPeptide(v_encoder, q_encoder, fusion, classes, attention)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, seq_lengths=None):
        f_mutated = self.siamese_encoder1(x[0])
        f_wide_type = self.siamese_encoder1(x[1])
        return self.fc(torch.cat((f_mutated, f_wide_type), dim=1))


class MMFPeptide(nn.Module):
    def __init__(self, v_encoder='resnet26', q_encoder='mlp', fusion='mlp', classes=6, attention=None, max_length=30):
        super().__init__()
        if attention == 'hamburger':
            self.attention = Hamburger(2048, 2048)
        else:
            self.attention = None
        # v_encoder could be resnet26 or resnet50
        if v_encoder == 'resnet26':
            self.v_encoder = ResNet3D(Bottleneck3D, [1, 2, 4, 1], self.attention)
            # self.v_encoder = ResNet3DFusion(Bottleneck, [1, 2, 4, 1], self.attention)
        elif v_encoder == 'resnet50':
            self.v_encoder = ResNet3D(Bottleneck3D, [3, 4, 6, 3], self.attention)
        else:
            raise NotImplementedError

        # q_encoder could be mlp, gru, rnn, lstm, transformer
        self.q_encoder = SEQ(seq_type=q_encoder, max_length=max_length)

        if fusion == 'mlp':
            self.fusion = nn.Linear(512 * 4 + 256, classes)
        elif fusion == 'att':
            self.fusion = nn.Linear(512 * 4 + 256, classes)
        else:
            raise NotImplementedError

        self.vox_fc = nn.Linear(2048, classes)
        self.seq_fc = nn.Linear(256, classes)

    def forward(self, x, seq_lengths=None):
        vox, seq = x
        # print(vox.shape)
        # print(seq.shape)
        seq_emb = self.q_encoder(seq, seq_lengths)

        vox_emb = self.v_encoder(vox, seq_emb)
        # print(vox_emb.shape)
        # print(seq_emb.shape)
        fused_feature = torch.cat((seq_emb, vox_emb), dim=1)
        pred = self.fusion(fused_feature)
        # pred1 = self.vox_fc(vox_emb)
        # pred2 = self.seq_fc(seq_emb)
        return pred


if __name__ == "__main__":
    # model = MMFPeptide()
    # voxel = torch.zeros((4, 3, 64, 64, 64))
    # # # h_in = torch.zeros((2, 2048, 2, 2, 2))
    # # # h = Hamburger(2048, 2048)
    # # # h(h_in)
    # seq = torch.ones((4, 50))
    # res = model.forward((voxel, seq))
    # out = model((voxel, seq))
    # print(out.shape)
    # model = ConvNet2D()
    input_seq = torch.ones((4, 1, 30))
    # model(input_seq)
    transformer = TransformerModel(nhead=4, d_model=32, num_layers=2)
    print(transformer(input_seq).shape)
