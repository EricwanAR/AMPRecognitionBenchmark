import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

        self.stride = stride

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, attention_module):
        super().__init__()
        if attention_module is not None:
            self.attention_flag = True
            self.attention_module = attention_module
        else:
            self.attention_flag = False

        self.in_channels = 64

        self.conv1 = nn.Conv3d(4, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if debug:
            print("shape1:", out.shape)
        out = self.max_pool(out)
        if debug:
            print("shape2:", out.shape)
        out = self.layer1(out)
        if debug:
            print("shape3:", out.shape)
        out = self.layer2(out)
        if debug:
            print("shape4:", out.shape)
        out = self.layer3(out)
        if debug:
            print("shape5:", out.shape)
        out = self.layer4(out)
        if self.attention_flag:
            out = self.attention_module(out)
        if debug:
            print("shape6:", out.shape)
        out = self.avg_pool(out)
        if debug:
            print("shape7:", out.shape)

        out = out.view(out.size(0), -1)
        return out


class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_value = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1, stride=1),
        )

    def forward(self, x, x_seq):
        x = self.f_value(x)
        print(x.shape)


class ResNet3DFusion(nn.Module):
    def __init__(self, block, layers, attention_module):
        super().__init__()
        if attention_module is not None:
            self.attention_flag = True
            self.attention_module = attention_module
        else:
            self.attention_flag = False

        self.in_channels = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 1)
        self.fc4 = nn.Linear(256, 1)
        self.fc5 = nn.Linear(256, 1)
        self.fc6 = nn.Linear(256, 1)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x, x_seq, debug=False):
        seq1 = self.fc1(x_seq)
        seq2 = self.fc2(x_seq)
        seq3 = self.fc3(x_seq)
        seq4 = self.fc4(x_seq)
        seq5 = self.fc5(x_seq)
        out = self.conv1(x)

        out = out * (seq1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1)

        out = self.bn1(out)
        out = self.relu(out)
        if debug:
            print("shape1:", out.shape)
        out = self.max_pool(out)
        if debug:
            print("shape2:", out.shape)
        out = self.layer1(out)
        out = out * (seq2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1)

        if debug:
            print("shape3:", out.shape)
        out = self.layer2(out)
        out = out * (seq3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1)

        if debug:
            print("shape4:", out.shape)
        out = self.layer3(out)
        out = out * (seq4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1)

        if debug:
            print("shape5:", out.shape)
        out = self.layer4(out)
        out = out * (seq5.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1)

        if self.attention_flag:
            out = self.attention_module(out)
        if debug:
            print("shape6:", out.shape)
        out = self.avg_pool(out)
        if debug:
            print("shape7:", out.shape)

        out = out.view(out.size(0), -1)
        return out


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
        elif self.seq_type == 'mlp':
            self.rnn = MLP(30, hidden_dim, out_dim, 3, 0.3)
        else:
            raise NotImplementedError(f'\'{seq_type}\' not implemented')

        self.rnn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, seq):
        if self.seq_type == 'mlp':
            return self.rnn(seq)
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


class VoxPeptide(nn.Module):
    def __init__(self, v_encoder='resnet26', q_encoder='mlp', fusion='mlp', classes=6, attention=None):
        super().__init__()

        if attention == 'hamburger':
            self.attention = Hamburger(2048, 2048)
        else:
            self.attention = None
        # v_encoder could be resnet26 or resnet50
        if v_encoder == 'resnet26':
            self.v_encoder = ResNet3D(Bottleneck, [1, 2, 4, 1], self.attention)

        if fusion == 'mlp':
            self.fusion = nn.Linear(512 * 4 + 256, classes)
        elif fusion == 'att':
            self.fusion = nn.Linear(512 * 4 + 256, classes)
        else:
            raise NotImplementedError
        self.vox_fc = nn.Linear(2048, classes)

    def forward(self, x, seq_lengths=None):
        vox, seq = x
        seq_emb = self.v_encoder(vox)
        pred = self.vox_fc(seq_emb)
        return pred


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
    
class GLOBPeptide(nn.Module):
    def __init__(self, v_encoder='resnet26', q_encoder='mlp', fusion='mlp', classes=6, attention=None, max_length=30):
        super().__init__()
        self.classes = classes
        # q_encoder could be mlp, gru, rnn, lstm, transformer
        self.q_encoder = MLP(input_dim=10, hidden_dim=128, output_dim=128, num_layers=3, dropout_rate=0.3)

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
            self.v_encoder = ResNet3D(Bottleneck, [1, 2, 4, 1], self.attention)
            # self.v_encoder = SwinUNETR(img_size=(64, 64, 64), in_channels=3, out_channels=1)
        elif v_encoder == 'resnet50':
            self.v_encoder = ResNet3D(Bottleneck, [3, 4, 6, 3], self.attention)
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
            self.v_encoder = ResNet3D(Bottleneck, [1, 2, 4, 1], self.attention)
            # self.v_encoder = ResNet3DFusion(Bottleneck, [1, 2, 4, 1], self.attention)
        elif v_encoder == 'resnet50':
            self.v_encoder = ResNet3D(Bottleneck, [3, 4, 6, 3], self.attention)
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
