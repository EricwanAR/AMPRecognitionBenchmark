import os
import networkx as nx
import torch
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_one
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Align import PairwiseAligner
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from Bio.SeqUtils.ProtParam import ProteinAnalysis


aligner = PairwiseAligner()
# Define the scoring system
aligner.mode = 'global'  # Global alignment
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.1


def calculate_similarity(seq_gen, seq_tem):

    # Perform the alignment
    score = aligner.score(seq_gen, seq_tem)
    return score / len(seq_gen)


def parse_ll37_testset(path='metadata/LL37_v0.csv'):
    files = open(path, 'r').readlines()
    seqs = []
    for file in files:
        seqs.append(str(file.strip().split(',')[-1].strip().upper()))
    return seqs


def gmean(labels):
    # Assuming 'labels' is a NumPy array of shape (n, m)
    # Calculate the geometric mean across axis 1

    # Take the product of elements along axis=1
    product = np.prod(labels, axis=1)

    # nth root of product, where n is the number of elements along axis=1
    geometric_means = product ** (1.0 / labels.shape[1])
    return geometric_means


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


ATOMS = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 30}
ATOMS_R = {'H': 1, 'C': 1.5, 'N': 1.5, 'O': 1.5, 'S': 2}
AMINO_ACID_WATER = {'A': 255, 'V': 255, 'P': 255, 'F': 255, 'W': 255, 'I': 255, 'L': 255, 'G': 155, 'M': 155,
                    'Y': 55, 'S': 55, 'T': 55, 'C': 55, 'N': 55, 'Q': 55, 'D': 55, 'E': 55, 'K': 55, 'R': 55, 'H': 55}
AMINO_ACID_CHARGE = {'D': 55, 'E': 55, 'A': 155, 'V': 155, 'P': 155, 'F': 155, 'W': 155, 'I': 155, 'L': 155, 'G': 155,
                     'M': 155, 'Y': 155, 'S': 155, 'T': 155, 'C': 155, 'N': 155, 'Q': 155, 'K': 255, 'R': 255, 'H': 255}
# AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
#         'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19}
AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
        'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21}


def pdb_parser(structure):
    """

    """
    voxel = np.zeros((4, 64, 64, 64), dtype=np.int8)
    id = ''
    seq_str = ''
    for i in structure[0]:
        id = i.id
    chain = structure[0][id]
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            resname = res.get_resname()
            amino = three_to_one(resname)
            seq_str += str(amino)
            ATOM_WATER = AMINO_ACID_WATER[amino]
            ATOM_CHARGE = AMINO_ACID_CHARGE[amino]
            ATOM_CATEGORY = AMAs[amino] * 20

            for i in res:
                if i.id not in ATOMS.keys():
                    continue
                x, y, z = i.get_coord()
                if abs(x) > 32:
                    x = clamp(x, -31, 31)
                if abs(y) > 32:
                    y = clamp(x, -31, 31)
                if abs(z) > 32:
                    z = clamp(x, -31, 31)
                x_i, y_i, z_i = int(x) + 32, int(y) + 32, int(z) + 32
                ATOM_WEIGHT = ATOMS[i.id]
                ATOM_R = ATOMS_R[i.id]

                if ATOM_R <= 1.5:
                    voxel[0, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WEIGHT
                    voxel[1, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WATER
                    voxel[2, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CHARGE
                    voxel[3, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CATEGORY
                else:
                    voxel[0, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WEIGHT
                    voxel[1, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WATER
                    voxel[2, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_CHARGE
                    voxel[3, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_CATEGORY
    return voxel


def read_scoring_functions(pdb):
    scoring = False
    profile = []
    for line in open(pdb):
        if line.startswith("#END_POSE_ENERGIES_TABLE"):
            scoring = False
        if scoring:
            data = [float(v) for v in line.split()[1:]]
            profile.append(data)
        if line.startswith("pose"):
            scoring = True
    return profile


def load_aa_features(feature_path):
    aa_features = {}
    for line in open(feature_path):
        line = line.strip().split()
        aa, features = line[0], [float(feature) for feature in line[1:]]
        aa_features[aa] = features
    return aa_features


class PairData(Data):
    def __init__(self, edge_index_s, x_s):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s

    def __inc__(self, key, value, *args):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'wide_nodes':
            return self.x_s.num_nodes
        else:
            return super().__inc__(key, value, *args)
        

def calculate_property(seq):
    # print(seq)
    analysed_seq = ProteinAnalysis(seq)
    aa_counts = analysed_seq.count_amino_acids()
    aliphatic_index = ((aa_counts['A'] + 2.9 * aa_counts['V'] + 3.9 * (aa_counts['I'] + aa_counts['L'])) / len(seq))

    charged_amino_acids = ['D', 'E', 'R', 'K', 'H']
    total_charged = sum(aa_counts[aa] for aa in charged_amino_acids)
    charge_density = total_charged / len(seq)
    alpha_helix, beta_helix, turn_helix = analysed_seq.secondary_structure_fraction()

    return list(
        [round(analysed_seq.gravy(), 3) * 10, round(aliphatic_index, 3) * 10, round(analysed_seq.aromaticity(), 3) * 10,
         round(analysed_seq.instability_index(), 3), round(alpha_helix * 10, 3), round(beta_helix * 10, 3),
         round(turn_helix * 10, 3), round(analysed_seq.charge_at_pH(7), 3), round(analysed_seq.isoelectric_point(), 3),
         round(charge_density, 3) * 10])


def construct_graph(structure, seq, gt, pdb_path, threshold, max_length):
    # Initialize an empty graph
    G = nx.Graph()

    # Check if chain 'A' is present in the structure, return None if not
    if 'A' not in structure[0]:
        return None
    chain = structure[0]['A']

    # Add nodes to the graph for each amino acid in the chain with their 3D positions
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            resname = res.get_resname()
            alpha_c_atom = res["CA"]
            x, y, z = alpha_c_atom.get_coord()  # Get 3D coordinates
            G.add_node(res.id[1], name=resname, position=(x, y, z))

    # Add edges between nodes if the distance between alpha carbons is <= 5Å
    for i, m in enumerate(G.nodes):
        for n in list(G.nodes)[i + 1:]:
            distance = chain[m]["CA"] - chain[n]["CA"]
            if distance <= 5:
                G.add_edge(m, n, weight=5 / distance)

    # Load scoring functions and features
    scoring = read_scoring_functions(pdb_path)
    # print(pdb_path)
    aa_features = load_aa_features('metadata/features.txt')

    # Assign features to nodes in the graph, including the 3D position
    for node_id, node_data in G.nodes(data=True):
        res = chain[node_id]
        aa_feature = aa_features[three_to_one(res.get_resname())]
        energy_feature = scoring[node_id - 1]
        position_feature = list(node_data['position'])  # Convert position tuple to list
        node_data['x'] = aa_feature + energy_feature + position_feature
        # node_data['x'][20:40]=[0 for i in range(20)]
        # node_data['x'][40:43]=[0 for i in range(3)]

    # Convert graph to data format used in ML models
    G = nx.convert_node_labels_to_integers(G)
    data_wt = from_networkx(G)
    data_graph = PairData(data_wt.edge_index, data_wt.x)
    data_graph.gt = (gt < threshold).astype(int)
    global_properties = calculate_property(seq)  

    # Encode the sequence and pad it to a fixed length
    seq_emb = [AMAs[char] for char in seq] + [0] * (max_length - len(seq))

    # print(seq_emb)
    data_graph.seq = seq_emb
    data_graph.global_f = global_properties
    return data_graph


def remove_features(x, features_to_remove):
    # 创建一个掩码来保留特征
    num_features = x.size(1)
    mask = torch.ones(num_features, dtype=torch.bool)
    mask[features_to_remove] = False
    
    # 更新节点特征矩阵
    x = x[:, mask]
    
    return x


class MDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, threshold=32, mode='train', max_length=50, pdb_src='af', data_ver='0922', exclude_feature=None):
        self.num_classes = 6
        self.num_features = 43
        exclude_flag = True
        if exclude_feature == 0:
            self.num_features = 23
            exclude_index = torch.arange(0, 20)
        elif exclude_feature == 1:
            self.num_features = 23
            exclude_index = torch.arange(20, 40)
        elif exclude_feature == 2:
            self.num_features = 40
            exclude_index = torch.arange(40, 43)
        else:
            exclude_flag = False
        exclude_list = pd.read_csv('metadata/data_simi.csv', encoding="unicode_escape")['Seq'].str.upper().str.strip().tolist()
        exclude_filter = False

        p = PDBParser(QUIET=True)

        if mode == 'train':
            all_data = pd.read_csv(f'metadata/data_{data_ver}_i.csv', encoding="unicode_escape").values
            exclude_filter = True
        elif mode == 'qlx':
            all_data = pd.read_csv('metadata/data_qlx.csv', encoding="unicode_escape").values
        elif mode == 'saap':
            all_data = pd.read_csv('metadata/data_saap.csv', encoding="unicode_escape").values
        else:
            raise NotImplementedError
        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], np.concatenate((all_data[:, 4:9], all_data[:, 10:11]), axis=1)

        filter_idx_list = []
        seq_new_list = []
        label_list = []
        for idx in range(len(idx_list)):
            seq = seq_list[idx].upper().strip()
            if 'X' in seq or 'B' in seq or 'J' in seq or 'Z' in seq or 'U' in seq or 'O' in seq or len(seq) > max_length or len(seq) < 6:
                continue
            if exclude_filter:
                if seq in exclude_list:
                    continue

            filter_idx_list.append(idx)
            seq_new_list.append(seq)
            label_list.append(labels[idx])

        if pdb_src == 'af':
            pdb_root = './pdb/pdb_af/' 
        elif pdb_src == 'hf':
            pdb_root = './pdb/pdb_dbassp/'
        else:
            raise NotImplementedError
        self.data_list = []
        for i in tqdm(range(len(filter_idx_list))):
                idx = filter_idx_list[i]
                seq = seq_new_list[i]
                label = label_list[i]

                # if os.path.exists("./pdb/pdb_dbassp/" + seq + ".pdb"):
                #     pdb_path = "./pdb/pdb_dbassp/" + seq + ".pdb"
                # elif os.path.exists("./pdb/pdb_gen/" + seq + ".pdb"):
                #     pdb_path = "./pdb/pdb_gen/" + seq + ".pdb"
                if os.path.exists(pdb_root + seq + ".pdb"):
                    pdb_path = pdb_root + seq + ".pdb"
                else:
                    continue
                    # raise FileNotFoundError

                structure = p.get_structure(idx, pdb_path)
                voxel1 = construct_graph(structure, seq, label, pdb_path, threshold, max_length)
                if exclude_flag:
                    voxel1.x_s = remove_features(voxel1.x_s, exclude_index)
                self.data_list.append(voxel1)

    def __getitem__(self, idx):
        voxel1 = self.data_list[idx]

        return voxel1

    def __len__(self):
        return len(self.data_list)
    

# class SDataset(Dataset):
#     '''
#     x: Features.
#     y: Targets, if none, do prediction.
#     '''

#     def __init__(self, threshold=32, mode='train', task='0', max_length=50, pdb_src='af', data_ver='0922'):
#         self.num_classes = 1
#         self.task = int(task)
#         exclude_list = pd.read_csv('metadata/data_simi.csv', encoding="unicode_escape")['Seq'].str.upper().str.strip().tolist()
#         exclude_filter = False

#         p = PDBParser(QUIET=True)

#         if mode == 'train':
#             all_data = pd.read_csv(f'metadata/data_{data_ver}_i.csv', encoding="unicode_escape").values
#             exclude_filter = True
#         elif mode == 'qlx':
#             all_data = pd.read_csv('metadata/data_qlx.csv', encoding="unicode_escape").values
#         elif mode == 'saap':
#             all_data = pd.read_csv('metadata/data_saap.csv', encoding="unicode_escape").values
#         else:
#             raise NotImplementedError
#         idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], np.concatenate((all_data[:, 4:9], all_data[:, 10:11]), axis=1)
#         labels = labels[:, self.task:self.task+1]

#         filter_idx_list = []
#         seq_new_list = []
#         label_list = []
#         for idx in range(len(idx_list)):
#             seq = seq_list[idx].upper().strip()
#             if 'X' in seq or 'B' in seq or 'J' in seq or 'Z' in seq or 'U' in seq or 'O' in seq or len(seq) > max_length or len(seq) < 6:
#                 continue
#             if exclude_filter:
#                 if seq in exclude_list:
#                     continue

#             filter_idx_list.append(idx)
#             seq_new_list.append(seq)
#             label_list.append(labels[idx])

#         if pdb_src == 'af':
#             pdb_root = './pdb/pdb_af/' 
#         elif pdb_src == 'hf':
#             pdb_root = './pdb/pdb_dbassp/'
#         else:
#             raise NotImplementedError
#         self.data_list = []
#         for i in tqdm(range(len(filter_idx_list))):
#                 idx = filter_idx_list[i]
#                 seq = seq_new_list[i]
#                 label = label_list[i]

#                 # if os.path.exists("./pdb/pdb_dbassp/" + seq + ".pdb"):
#                 #     pdb_path = "./pdb/pdb_dbassp/" + seq + ".pdb"
#                 # elif os.path.exists("./pdb/pdb_gen/" + seq + ".pdb"):
#                 #     pdb_path = "./pdb/pdb_gen/" + seq + ".pdb"
#                 if os.path.exists(pdb_root + seq + ".pdb"):
#                     pdb_path = pdb_root + seq + ".pdb"
#                 else:
#                     continue
#                     # raise FileNotFoundError

#                 structure = p.get_structure(idx, pdb_path)
#                 voxel1 = construct_graph(structure, seq, label, pdb_path, threshold, max_length)
#                 self.data_list.append(voxel1)

#     def __getitem__(self, idx):
#         voxel1 = self.data_list[idx]

#         return voxel1

#     def __len__(self):
#         return len(self.data_list)
    
