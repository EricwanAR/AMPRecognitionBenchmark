import argparse
import json
import logging
import os
import time
import pandas as pd

from dataset import MDataset
from network import VoxPeptide
from sklearn.model_selection import KFold
from train import train
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import numpy as np
from loss import MLCE, SuperLoss, LogCoshLoss
from utils import set_seed


parser = argparse.ArgumentParser(description='resnet26')
# model setting
parser.add_argument('--model', type=str, default='mlp',
                    help='resnet34 resnet50 densenet convnext vit swintf')
parser.add_argument('--in-channels', dest='in_channels', type=int, default=4)
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--fusion', type=str, default='1',
                    help="Seed for splitting dataset (default 1)")

# task & dataset setting
parser.add_argument('--pdb-src', type=str, dest='pdb_src', default='af',
                    help='af or hf')
parser.add_argument('--task-type', type=str, dest='task_type', default='mlc',
                    help='mlc or slc')
parser.add_argument('--data-ver', type=str, dest='data_ver', default='0920',
                    help='data version')
parser.add_argument('--task', type=str, default='all',
                    help='task: anti toxin anti-all mechanism anti-binary anti-regression mic')
parser.add_argument('--classes', type=int, default=6,
                    help='model')
parser.add_argument('--max-length', dest='max_length', type=int, default=30,
                    help='Max length for sequence filtering')
parser.add_argument('--split', type=int, default=5,
                    help="Split k fold in cross validation (default: 5)")
parser.add_argument('--seed', type=int, default=1,
                    help="Seed for splitting dataset (default: 1)")
parser.add_argument('--threshold', type=float, default=128,
                    help="MIC threshold for determine labels (default: 128)")

# training setting
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU index to use, -1 for CPU (default: 0)')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=256,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.0005,
                    help='weight decay (default: 0.0005)')
parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=0,
                    help='number of warm start steps for learning rate (default: 10)')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stopping (default: 10)')
parser.add_argument('--resnet34_16', type=str, default='./run/resnet3416-bce256-0.001-50af',
                    help='path of the pretrain model')
parser.add_argument('--resnet34_32', type=str, default='./run/resnet3432-bce256-0.001-50af',
                    help='path of the pretrain model')
parser.add_argument('--resnet50_16', type=str, default='./run/resnet5016-bce256-0.001-50af',
                    help='path of the pretrain model')
parser.add_argument('--resnet50_32', type=str, default='./run/resnet5032-bce256-0.001-50af',
                    help='path of the pretrain model')
parser.add_argument('--metric-avg', type=str, dest='metric_avg', default='macro',
                    help='metric average type')

# args for losses
parser.add_argument('--loss', type=str, default='bce',
                    help='loss function (mlce, sl, mix)')

parser.add_argument('--bias-curri', dest='bias_curri', action='store_true', default=False,
                    help='directly use loss as the training data (biased) or not (unbiased)')
parser.add_argument('--anti-curri', dest='anti_curri', action='store_true', default=False,
                    help='easy to hard (curri), hard to easy (anti)')
parser.add_argument('--std-coff', dest='std_coff', type=float, default=1,
                    help='the hyper-parameter of std')

args = parser.parse_args()

model_path = {'resnet34@16': args.resnet34_16, 'resnet34@32': args.resnet34_32, 'resnet50@16': args.resnet50_16, 'resnet50@32': args.resnet50_32}


def main():
    set_seed(args.seed)
    device = torch.device("cpu" if args.gpu == -1 or not torch.cuda.is_available() else f"cuda:{args.gpu}")

    results = pd.DataFrame()

    logging.info('Loading Test Dataset')
    qlx_set = MDataset(threshold=args.threshold, mode='qlx', max_length=args.max_length)
    qlx_loader = DataLoader(qlx_set, batch_size=1, shuffle=False)
    
    models = {'resnet34@16': VoxPeptide(classes=qlx_set.num_classes, v_encoder='resnet34', channels=16, in_channels=args.in_channels),
              'resnet34@32': VoxPeptide(classes=qlx_set.num_classes, v_encoder='resnet34', channels=32, in_channels=args.in_channels),
              'resnet50@16': VoxPeptide(classes=qlx_set.num_classes, v_encoder='resnet50', channels=16, in_channels=args.in_channels),
              'resnet50@32': VoxPeptide(classes=qlx_set.num_classes, v_encoder='resnet50', channels=32, in_channels=args.in_channels)}

    for model_name in model_path.keys():
        model = models[model_name]
        pred_all = []
        gt_all = []
        for i in range(1,6):
            model.load_state_dict(torch.load(os.path.join(model_path[model_name], f'model_{i}.pth')))
            model.to(device).eval()
            with torch.no_grad():
                for data in qlx_loader:
                    voxel, seq, gt = data
                    gt_all.append(gt.to(device))
                    out = model((voxel.to(device), seq.to(device)))
                    pred_all.append(out)
        pred_all = torch.nn.functional.sigmoid(torch.cat(pred_all, dim=0)).squeeze().cpu().numpy()
        gt_all = torch.cat(gt_all, dim=0).int().squeeze().cpu().numpy()
        results[model_name] = pred_all.ravel(order='F')

    results['gt'] = gt_all.ravel(order='F')
    
    results.to_csv("preds.csv", index=False)


if __name__ == "__main__":
    main()