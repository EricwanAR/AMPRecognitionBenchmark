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




class FusionPeptide(nn.Module):
    def __init__(self, v_encoder='resnet34', q_encoder='lstm', g_encoder='mlp', mode='111', classes=6, channels=16):
        super().__init__()
        if mode == '000':
            raise KeyError('None of the module acitvated')
        self.classes = classes
        self.mode = [False, False, False]
        final_dim = 0
        
        if mode[0] == '1':
            final_dim += 128
            self.mode[0] = True
            if q_encoder == 'lstm':
                self.q_encoder = nn.LSTM(
                    input_size=21,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,  # input & output will take batch size as 1 dim (batch, time_step, input_size)
                    bidirectional=True
                )
                self.q_fc = nn.Linear(256, 128)
            else:
                raise NotImplementedError
        
        if mode[1] == '1':
            final_dim += 512
            self.mode[1] = True
            if v_encoder == 'resnet34':
                self.v_encoder = ResNet3D34(input_channels=4, base_channels=channels, feature_dim=512)
            elif v_encoder == 'resnet50':
                self.v_encoder = ResNet3D50(input_channels=4, base_channels=channels, feature_dim=512)
            elif v_encoder == 'densenet':
                self.v_encoder = DenseNet3D(input_channels=4, base_channels=channels, growth_rate=16 if channels < 48 else 32, feature_dim=512)
            else:
                raise NotImplementedError(f'\'{v_encoder}\' not implemented')
        
        if mode[2] == '1':
            final_dim += 128
            self.mode[2] = True
            if g_encoder == 'mlp':
                self.g_encoder = MLP(10, 128, 128, 3, 0.3)
            else:
                raise NotImplementedError

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128), nn.LeakyReLU(0.1), nn.Dropout(0.3), 
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.3), 
            nn.Linear(64, self.classes))

    def forward(self, x):
        vox, seq, globf = x
        fusion = []
        if self.mode[0]:
            fusion.append(self.q_fc(self.q_encoder(seq)[0][:, -1, :]))
        if self.mode[1]:
            fusion.append(self.v_encoder(vox))
        if self.mode[2]:
            fusion.append(self.g_encoder(globf))
        fusion = torch.cat(fusion, dim=-1)
        pred = self.fc(fusion)
        return pred


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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
