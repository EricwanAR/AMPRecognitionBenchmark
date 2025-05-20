import argparse
import json
import logging
import os
import time

from dataset import MDataset
from network import FusionPeptide
from sklearn.model_selection import KFold
from train import train
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import numpy as np
from loss import MLCE, SuperLoss, LogCoshLoss, ResampleLoss
from utils import set_seed


parser = argparse.ArgumentParser(description='resnet26')
# model setting
parser.add_argument('--model', type=str, default='resnet34',
                    help='resnet34 resnet50 densenet')
parser.add_argument('--channels', type=int, default=32)
parser.add_argument('--mode', type=str, default='101',
                    help="0 for off and 1 for on. First digit for seq, second for voxel, third for globf")

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
parser.add_argument('--pretrain', type=str, dest='pretrain', default='',
                    help='path of the pretrain model')  # /home/duadua/Desktop/fetal/3dpretrain/runs/e50.pth
parser.add_argument('--metric-avg', type=str, dest='metric_avg', default='macro',
                    help='metric average type')

# args for losses
parser.add_argument('--loss', type=str, default='bce',
                    help='loss function (bce, mlce, rliv, rlsriv, rlrb, rlcb)')

parser.add_argument('--bias-curri', dest='bias_curri', action='store_true', default=False,
                    help='directly use loss as the training data (biased) or not (unbiased)')
parser.add_argument('--anti-curri', dest='anti_curri', action='store_true', default=False,
                    help='easy to hard (curri), hard to easy (anti)')
parser.add_argument('--std-coff', dest='std_coff', type=float, default=1,
                    help='the hyper-parameter of std')

args = parser.parse_args()


def main():
    set_seed(args.seed)

    if args.task_type == 'slc':
        if args.task == 'all':
            raise ValueError('Choose one task number to run single label classification')
        args.classes = 1
    elif args.task_type == 'mlc':
        pass
    else:
        raise NotImplementedError

    weight_dir = "./run/" + args.mode + '-' + args.loss + str(args.batch_size) + '-' + str(args.lr) + '-' + str(args.epochs) + args.pdb_src
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+'),
        logging.StreamHandler()],
        format="%(asctime)s: %(message)s", datefmt="%F %T", level=logging.INFO)

    logging.info(f'saving_dir: {weight_dir}')
    
    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    device = torch.device("cpu" if args.gpu == -1 or not torch.cuda.is_available() else f"cuda:{args.gpu}")

    logging.info('Loading Training Dataset')
    if args.task_type == 'mlc':
        set_all = MDataset(threshold=args.threshold, mode='train', max_length=args.max_length, pdb_src=args.pdb_src, data_ver=args.data_ver, model_mode=args.mode)
    # else:
    #     set_all = SDataset(threshold=args.threshold, mode='train', task=args.task, max_length=args.max_length, pdb_src=args.pdb_src)
    logging.info('Loading Test Dataset')
    if args.task_type == 'mlc':
        qlx_set = MDataset(threshold=args.threshold, mode='qlx', max_length=args.max_length, pdb_src=args.pdb_src, model_mode=args.mode)
        saap_set = MDataset(threshold=args.threshold, mode='saap', max_length=args.max_length, pdb_src=args.pdb_src, model_mode=args.mode)
    # else:
    #     qlx_set = SDataset(threshold=args.threshold, mode='qlx', task=args.task, max_length=args.max_length, pdb_src=args.pdb_src)
    #     saap_set = SDataset(threshold=args.threshold, mode='saap', task=args.task, max_length=args.max_length, pdb_src=args.pdb_src)

    best_perform_list = [[] for i in range(5)]
    qlx_perform_list = [[] for i in range(5)]
    saap_perform_list = [[] for i in range(5)]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(set_all)):
        train_set= Subset(set_all, train_idx)
        valid_set = Subset(set_all, val_idx)

        if args.loss == 'mlce' and args.task_type != 'slc':
            criterion = MLCE()
        elif args.loss == "bce" or args.task_type == 'slc':
            args.loss = "bce"
            weight = torch.Tensor([0] * args.classes)
            for i in train_set:
                weight += i[-1]
            weight = (len(train_set) - weight) / weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight.to(device))
        elif args.loss in ['rliv', 'rlsriv', 'rlrb', 'rlcb']:
            freq = torch.Tensor([0] * args.classes)
            for i in train_set:
                freq += i[-1]
            neg_freq = torch.Tensor([len(train_set)] * args.classes) - freq
            if args.loss == 'rliv':
                rwf = 'inv'
            elif args.loss == 'rlsriv':
                rwf = 'sqrt_inv'
            elif args.loss == 'rlrb':
                rwf = 'rebalance'
            elif args.loss == 'rlcb':
                rwf = 'CB'
            else:
                rwf = None
            criterion = ResampleLoss(class_freq=freq.to(device), neg_class_freq=neg_freq.to(device), 
                                     reweight_func=rwf)
        else:
            raise NotImplementedError

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        qlx_loader = DataLoader(qlx_set, batch_size=1, shuffle=False)
        saap_loader = DataLoader(saap_set, batch_size=1, shuffle=False)

        model = FusionPeptide(classes=set_all.num_classes, v_encoder=args.model, channels=args.channels, mode=args.mode)
        if len(args.pretrain) != 0:
            logging.info('loading pretrain model')
            # model = load_pretrain_model(model, torch.load(args.pretrain))
            model_state = model.state_dict()
            pretrained_state = torch.load(args.pretrain)
            pretrained_state = {k: v for k, v in pretrained_state.items() if
                                k in model_state and v.size() == model_state[k].size()}
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            # model.load_state_dict(torch.load(args.pretrain), strict=False)
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=True, weight_decay=5e-5)
        weights_path = f"{weight_dir}/model_{fold + 1}.pth"
        # early_stopping = EarlyStopping(patience=args.patience, path=weights_path)
        logging.info(f'Running Cross Validation {fold + 1}')
        logging.info(f'Fold {fold + 1}  Train set:{len(train_set)}, Valid set:{len(valid_set)}, Test set:qlx {len(qlx_set)} saap {len(saap_set)}')
        best_metric = 0
        best_qlx = 0
        best_saap = 0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            if args.task_type in ('mlc', 'slc') :
                train_loss, ap, f1, acc, auc = train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, ap: {ap:.3f}, f1: {f1:.3f}, acc: {acc:.3f}, auc: {auc:.3f}')
                avg_metric = ap + f1 + acc + auc
                if avg_metric > best_metric:
                    logging.info(f'Epoch: {epoch:03d} New Best validation metrics, running test session')
                    torch.save(model.state_dict(), weights_path)
                    best_metric = avg_metric
                    best_perform_list[fold] = np.asarray([ap, f1, acc, auc])
                    
                    _, qlx_ap, qlx_f1, qlx_acc, qlx_auc = train(args, epoch, model, None, qlx_loader, device, None, None)
                    logging.info(f'Epoch: {epoch:03d} QLX results, ap: {qlx_ap:.3f}, f1: {qlx_f1:.3f}, acc: {qlx_acc:.3f}, auc: {qlx_auc:.3f}')
                    qlx_metric = qlx_ap + qlx_f1 + qlx_acc + qlx_auc
                    if qlx_metric > best_qlx:
                        best_qlx = qlx_metric
                        qlx_perform_list[fold] = np.asarray([qlx_ap, qlx_f1, qlx_acc, qlx_auc])
                    
                    _, saap_ap, saap_f1, saap_acc, saap_auc = train(args, epoch, model, None, saap_loader, device, None, None)
                    logging.info(f'Epoch: {epoch:03d} SAAP results, ap: {saap_ap:.3f}, f1: {saap_f1:.3f}, acc: {saap_acc:.3f}, auc: {saap_auc:.3f}')
                    saap_metric = saap_ap + saap_f1 + saap_acc + saap_auc
                    if saap_metric > best_saap:
                        best_saap = saap_metric
                        saap_perform_list[fold] = np.asarray([saap_ap, saap_f1, saap_acc, saap_auc])
                           
            else:
                raise NotImplementedError
            
        logging.info(f'used time {(time.time()-start_time)/3600:.2f}h')

    logging.info(f'Cross Validation Finished!')
    best_perform_list = np.asarray(best_perform_list)
    qlx_perform_list = np.asarray(qlx_perform_list)
    saap_perform_list = np.asarray(saap_perform_list)
    logging.info('Best validation perform list\n%s', best_perform_list)
    logging.info('mean: %s', np.round(np.mean(best_perform_list, 0), 3))
    logging.info('std: %s', np.round(np.std(best_perform_list, 0), 3))
    logging.info('Best qlx perform list\n%s', qlx_perform_list)
    logging.info('mean: %s', np.round(np.mean(qlx_perform_list, 0), 3))
    logging.info('std: %s', np.round(np.std(qlx_perform_list, 0), 3))
    logging.info('Best saap perform list\n%s', saap_perform_list)
    logging.info('mean: %s', np.round(np.mean(saap_perform_list, 0), 3))
    logging.info('std: %s', np.round(np.std(saap_perform_list, 0), 3))
    perform = open(weight_dir+'/result.txt', 'w')
    perform.write('Valid\n')
    perform.write(','.join([str(i) for i in np.mean(best_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(best_perform_list, 0)])+'\n')
    perform.write('qlx\n')
    perform.write(','.join([str(i) for i in np.mean(qlx_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(qlx_perform_list, 0)])+'\n')
    perform.write('saap\n')
    perform.write(','.join([str(i) for i in np.mean(saap_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(saap_perform_list, 0)])+'\n')


if __name__ == "__main__":
    main()
