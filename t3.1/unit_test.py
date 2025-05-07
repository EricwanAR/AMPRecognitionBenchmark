import torch
import torch.nn as nn
import torch.nn.functional as F


SEQUENCE = 'GEFRRIVQRIRDFLRNLV'
WEIGHT_DIR = './weights/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pep_encoder(pep, max_length=30):
    AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
        'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21}
    pep = pep.upper().strip()
    pep_emb = [AMAs[char] for char in pep] + [0] * (max_length - len(pep))
    return torch.tensor(pep_emb).unsqueeze(0)


class SEQPeptide(nn.Module):
    def __init__(self, q_encoder='lstm', fusion='mlp', classes=6, max_length=30):
        super().__init__()
        self.classes = classes
        # q_encoder could be mlp, gru, rnn, lstm, transformer
        self.q_encoder = SEQ(seq_type=q_encoder, max_length=max_length)

        self.seq_fc = nn.Linear(128, classes)

    def forward(self, seq):
        seq_emb = self.q_encoder(seq)
        pred = self.seq_fc(seq_emb)
        return pred


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
        else:
            raise NotImplementedError(f'\'{seq_type}\' not implemented')

        self.rnn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, seq):
        one_hot_seq = F.one_hot(seq.to(torch.int64), num_classes=21).float()
        r_out = self.rnn(one_hot_seq, None)[0]  # None represents zero initial hidden state
        out = self.rnn_fc(r_out[:, -1, :])
        return out
    

def main():
    model = SEQPeptide().to(DEVICE)
    x = pep_encoder(SEQUENCE).to(DEVICE)
    results = []
    for i in range(5):
        model.load_state_dict(torch.load(f"{WEIGHT_DIR}/model_{i + 1}.pth"))
        model.eval()
        with torch.no_grad():
            results.append(model(x))
    preds = F.sigmoid(torch.cat(results, dim=0).mean(0))
    preds = preds.cpu().numpy()

    print(f'Predictions:')
    print(f'Enterococcus faecalis: {preds[0]:.2%}')
    print(f'Pseudomonas aeruginosa: {preds[1]:.2%}')
    print(f'Staphylococcus aureus: {preds[2]:.2%}')
    print(f'Acinetobacter baumannii: {preds[3]:.2%}')
    print(f'Enterobacteriaceae: {preds[4]:.2%}')
    print(f'Salmonella species: {preds[5]:.2%}')


if __name__ == '__main__':
    main()