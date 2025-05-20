import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GraphConv, GINConv, GATConv, SAGEConv, GPSConv, GINEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, MulAggregation
from torch_geometric.nn import aggr
from torch_geometric.nn import GraphNorm


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=1):
        super(SimpleSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Assuming key_channels = value_channels = embedding_dim
        self.key_channels = self.embedding_dim
        self.value_channels = self.embedding_dim

        # Linear layers for queries, keys, and values
        self.query = nn.Linear(embedding_dim, self.key_channels * num_heads)
        self.key = nn.Linear(embedding_dim, self.key_channels * num_heads)
        self.value = nn.Linear(embedding_dim, self.value_channels * num_heads)

        # Output projection layer
        self.proj = nn.Linear(self.value_channels * num_heads, embedding_dim)

        # Scaling for dot-product attention
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.key_channels // num_heads])))

    def forward(self, x1, x2, x3):
        # x1, x2, x3 shapes: [Batch_size, Embedding_dim]
        # Stack the inputs along a new dimension (sequence dimension)
        batch_size = x1.shape[0]
        x = torch.stack((x1, x2, x3), dim=1)  # [Batch_size, 3, Embedding_dim]

        # Compute queries, keys, values for all three inputs
        Q = self.query(x)  # [Batch_size, 3, num_heads * embedding_dim]
        K = self.key(x)    # [Batch_size, 3, num_heads * embedding_dim]
        V = self.value(x)  # [Batch_size, 3, num_heads * embedding_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.embedding_dim).transpose(1, 2)  # [Batch_size, num_heads, 3, embedding_dim]
        K = K.view(batch_size, -1, self.num_heads, self.embedding_dim).transpose(1, 2)  # [Batch_size, num_heads, 3, embedding_dim]
        V = V.view(batch_size, -1, self.num_heads, self.embedding_dim).transpose(1, 2)  # [Batch_size, num_heads, 3, embedding_dim]

        # Calculate dot product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention_scores, dim=-1)

        # Apply attention to V
        x = torch.matmul(attention, V)  # [Batch_size, num_heads, 3, embedding_dim]

        # Concatenate heads and put through final linear layer
        x = x.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.embedding_dim)
        x = self.proj(x)  # [Batch_size, 3, embedding_dim]

        # Sum the outputs from the three inputs
        out = x.sum(dim=1)  # [Batch_size, embedding_dim]
        return out

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_t, weights=None):
        loss = (y - y_t) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        return torch.mean(loss)


class GNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # self.fc2 = nn.Linear(200, 200)
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            in_dim = input_dim if layer == 0 else emb_dim
            if gnn_type == "gin":
                # self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                #                                        nn.Linear(emb_dim, emb_dim))))
                self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), GraphNorm(emb_dim), nn.ReLU(),
                                                       nn.Linear(emb_dim, emb_dim), nn.ReLU())))
            elif gnn_type == "gps":
                nn_ = Sequential(
                    Linear(in_dim, emb_dim),
                    ReLU(),
                    Linear(emb_dim, emb_dim),
                )
                conv = GPSConv(emb_dim, GINEConv(nn_), heads=4)
                self.gnns.append(conv)
            elif gnn_type == "gcn":
                self.gnns.append(GraphConv(in_dim, emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_dim, emb_dim))
            elif gnn_type == "gatv2":
                self.gnns.append(GATv2Conv(in_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(in_dim, emb_dim))
            else:
                raise ValueError("Invalid GNN type.")

    def forward(self, x, edge_index, edge_attr=None):
        h_list = [x]
        mut_site = []
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # if layer == self.num_layer - 1:
            #     # remove relu from the last layer
            #     h = F.dropout(h, self.drop_ratio, training=self.training)
            # else:
            #     h = F.dropout(F.relu(h), self.drop_ratio, training=self.training) # F.relu()
            h_list.append(h)
            # if len(h_list) == 2:
            #     previous_mut_site_feature = h_list[-2][mut_res_idx]
            #     current_mut_site_feature = h_list[-1][mut_res_idx]
            #     # print(previous_mut_site_feature.shape, current_mut_site_feature.shape)
            #     h_feature = self.global_encoder(previous_mut_site_feature)
            #     h_list[-1][mut_res_idx] = h_feature + current_mut_site_feature
            # if len(h_list) == 3:
            #     previous_mut_site_feature = h_list[-2][mut_res_idx].squeeze(0)
            #     current_mut_site_feature = h_list[-1][mut_res_idx].squeeze(0)

            #     h_feature = self.fc2(previous_mut_site_feature) + current_mut_site_feature
            #     h_list[-1][mut_res_idx] = h_feature.unsqueeze(0)
            # mut_site.append()
        # print(len(h_list))
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)
        # print('node_rep', node_representation.shape)
        return h_list[-1]


# orthogonal initialization
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)


def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    # positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4: l // 2].data.fill_(1.0)
        hh_b[l // 4: l // 2].data.fill_(1.0)


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


class FusionGraphOld(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0.5, graph_pooling="attention",
                 gnn_type="gat", concat_type=None, fds=False, feature_level='both', contrast_curri=False, aux_mode='11') -> object:
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.concat_type = concat_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri
        self.mode = [False, False]
        final_dim = emb_dim

        if aux_mode[0] == '1':
            final_dim += 128
            self.mode[0] = True
            self.q_encoder = nn.LSTM(
                input_size=21,
                hidden_size=128,
                num_layers=2,
                batch_first=True,  # input & output will take batch size as 1 dim (batch, time_step, input_size)
                bidirectional=True
            )
            self.q_fc = nn.Linear(256, 128)
        
        if aux_mode[1] == '1':
            final_dim += 128
            self.mode[1] = True
            self.g_encoder = MLP(10, 128, 128, 3, 0.3)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio), 
            nn.Linear(self.emb_dim, self.emb_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
            nn.Linear(self.emb_dim // 2, self.out_dim))

        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "mul":
            self.pool = MulAggregation()
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        elif graph_pooling == "lstm":
            self.pool = aggr.LSTMAggregation(emb_dim, emb_dim)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward_once(self, x, edge_index, batch):
        node_representation = self.gnn(x, edge_index)
        graph_rep = self.pool(node_representation, batch)
        return graph_rep

    def forward(self, data):
        fusion = [self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch)]
        seq, globf = data.seq, data.global_f
        device = self.fc[0].bias.device
        if self.mode[0]:
            seq = torch.tensor(np.asarray(seq, dtype=np.float32), device=device)
            fusion.append(self.q_fc(self.q_encoder(seq)[0][:, -1, :]))
        if self.mode[1]:
            globf = torch.tensor(np.asarray(globf, dtype=np.float32), device=device)
            fusion.append(self.g_encoder(globf))
        fusion = torch.cat(fusion, dim=-1)
        x = self.fc(fusion)
        return x


class FusionGraph(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0.5, graph_pooling="attention",
                 gnn_type="gat", fusion_type="concat", fds=False, feature_level='both', contrast_curri=False, aux_mode='11') -> object:
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.fusion_type = fusion_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri
        self.mode = [False, False]
        
        # 各模态特征维度
        self.modal_dims = {'graph': emb_dim}
        
        if aux_mode[0] == '1':
            self.mode[0] = True
            self.modal_dims['seq'] = 128
            self.q_encoder = nn.LSTM(
                input_size=21,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            self.q_fc = nn.Linear(256, 128)
        
        if aux_mode[1] == '1':
            self.mode[1] = True
            self.modal_dims['glob'] = 128
            self.g_encoder = MLP(10, 128, 128, 3, 0.3)

        # 根据融合方式设置最终维度
        if fusion_type == "concat":
            final_dim = sum(self.modal_dims.values())
        elif fusion_type in ["weighted", "attention"]:
            final_dim = emb_dim  # 使用统一维度
            # 为seq和glob特征添加投影层,使维度统一为emb_dim
            if self.mode[0]:
                self.seq_proj = nn.Linear(128, emb_dim)
            if self.mode[1]:
                self.glob_proj = nn.Linear(128, emb_dim)
            
            if fusion_type == "weighted":
                # 可学习权重
                self.modal_weights = nn.Parameter(torch.ones(len(self.modal_dims)))
                self.softmax = nn.Softmax(dim=0)
            else:  # attention
                # 多头注意力
                self.multihead_attn = nn.MultiheadAttention(
                    embed_dim=emb_dim,
                    num_heads=4,
                    batch_first=True
                )
                self.layer_norm = nn.LayerNorm(emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio), 
            nn.Linear(self.emb_dim, self.emb_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
            nn.Linear(self.emb_dim // 2, self.out_dim))

        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "mul":
            self.pool = MulAggregation()
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        elif graph_pooling == "lstm":
            self.pool = aggr.LSTMAggregation(emb_dim, emb_dim)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward_once(self, x, edge_index, batch):
        node_representation = self.gnn(x, edge_index)
        graph_rep = self.pool(node_representation, batch)
        return graph_rep

    def fuse_features(self, features):
        if self.fusion_type == "concat":
            return torch.cat(features, dim=-1)
        
        # 将所有特征投影到相同维度
        if self.mode[0]:
            features[1] = self.seq_proj(features[1])
        if self.mode[1] and len(features) > 2:
            features[2] = self.glob_proj(features[2])
            
        if self.fusion_type == "weighted":
            # 使用可学习权重加权
            weights = self.softmax(self.modal_weights)
            fused = sum(w * f for w, f in zip(weights, features))
            return fused
        
        elif self.fusion_type == "attention":
            # 将特征堆叠成序列
            features = torch.stack(features, dim=1)  # [batch_size, num_modalities, emb_dim]
            
            # 多头注意力
            attn_output, _ = self.multihead_attn(features, features, features)
            attn_output = self.layer_norm(attn_output + features)
            
            # 平均池化得到最终特征
            fused = torch.mean(attn_output, dim=1)  # [batch_size, emb_dim]
            return fused

    def forward(self, data):
        features = [self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch)]
        seq, globf = data.seq, data.global_f
        device = self.fc[0].bias.device
        
        if self.mode[0]:
            seq = torch.tensor(np.asarray(seq, dtype=np.float32), device=device)
            features.append(self.q_fc(self.q_encoder(seq)[0][:, -1, :]))
            
        if self.mode[1]:
            globf = torch.tensor(np.asarray(globf, dtype=np.float32), device=device)
            features.append(self.g_encoder(globf))
            
        fusion = self.fuse_features(features)
        x = self.fc(fusion)
        return x


class MMGraph(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0.5, graph_pooling="attention",
                 gnn_type="gat", concat_type=None, fds=False, feature_level='both', contrast_curri=False, max_length=50) -> object:
        super(MMGraph, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.concat_type = concat_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri

        self.graph_pool = nn.Linear(self.emb_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
            nn.Linear(self.emb_dim, self.out_dim))

        if fds:
            self.dir = True
        else:
            self.dir = False
        self.global_encoder = nn.Sequential(nn.Linear(10, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),)
        self.seq_encoder = nn.Sequential(
            nn.Linear(max_length, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
            )
        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "mul":
            self.pool = MulAggregation()
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        elif graph_pooling == "lstm":
            self.pool = aggr.LSTMAggregation(emb_dim, emb_dim)
        else:
            raise ValueError("Invalid graph pooling type.")
        self.att = SimpleSelfAttention(emb_dim, num_heads=4)

    def forward_once(self, x, edge_index, batch):
        node_representation = self.gnn(x, edge_index)
        graph_rep = self.pool(node_representation, batch)
        return graph_rep

    def forward(self, data):
        seq1, global_1 = data.seq, data.global_f
        device = self.graph_pool.bias.device
        seq1 = torch.tensor(seq1, dtype=torch.float, device=device)
        global_1 = torch.tensor(global_1, dtype=torch.float, device=device)

        graph_rep_be = self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch)
        seq1_rep_be = self.seq_encoder(seq1)
        global1 = self.global_encoder(global_1)

        a1 = self.att(graph_rep_be, seq1_rep_be, global1)
        return self.fc(a1)



class PMMGraph(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0.5, graph_pooling="attention",
                 gnn_type="gat", concat_type=None, fds=False, feature_level='both', contrast_curri=False):
        super(PMMGraph, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.concat_type = concat_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri

        # Define the learnable prompt token
        self.prompt_token = nn.Parameter(torch.randn(1, 10))

        self.graph_pool = nn.Linear(self.emb_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim + 10, self.emb_dim),  # Adjust input size to include the prompt token
            nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
            nn.Linear(self.emb_dim, self.out_dim))

        self.dir = fds
        self.global_encoder = nn.Sequential(nn.Linear(10, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio))
        self.seq_encoder = nn.Sequential(
            nn.Linear(30, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
        )
        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Initialize pooling based on the specified type
        if graph_pooling in ["sum", "mean", "max", "mul", "attention", "set2set", "lstm"]:
            pooling_classes = {
                "sum": global_add_pool,
                "mean": global_mean_pool,
                "max": global_max_pool,
                "mul": aggr.MulAggregation(),
                "attention": GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1)),
                "set2set": Set2Set(emb_dim, processing_steps=2),
                "lstm": aggr.LSTMAggregation(emb_dim, emb_dim)
            }
            self.pool = pooling_classes[graph_pooling]
        else:
            raise ValueError("Invalid graph pooling type.")

        self.att = SimpleSelfAttention(emb_dim + 10, num_heads=4)  # Adjust for prompt dimension

    def forward_once(self, x, edge_index, batch):
        node_representation = self.gnn(x, edge_index)
        graph_rep = self.pool(node_representation, batch)
        # Concatenate the prompt token
        graph_rep = torch.cat([graph_rep, self.prompt_token.expand(graph_rep.size(0), -1)], dim=1)
        return graph_rep

    def forward(self, data):
        seq1 = torch.tensor(np.array(data.seq, dtype=np.float32)).to(device='cuda')
        global_1 = torch.tensor(np.array(data.global_f, dtype=np.float32)).to(device='cuda')

        graph_rep_be = self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch)
        seq1_rep_be = self.seq_encoder(seq1)
        global1_rep = self.global_encoder(global_1)
        # Concatenate the prompt token to other representations as well
        seq1_rep_be = torch.cat([seq1_rep_be, self.prompt_token.expand(seq1_rep_be.size(0), -1)], dim=1)
        global1_rep = torch.cat([global1_rep, self.prompt_token.expand(global1_rep.size(0), -1)], dim=1)

        # Process combined representations
        a1 = self.att(graph_rep_be, seq1_rep_be, global1_rep)
        return self.fc(a1)
    

