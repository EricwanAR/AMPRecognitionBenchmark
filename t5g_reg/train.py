import torch
import numpy
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef, KendallRankCorrCoef


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    train_loss = 0
    num_labels = model.out_dim

    metric_mae = MeanAbsoluteError().to(device)
    metric_rse = RelativeSquaredError(num_outputs=num_labels).to(device)
    metric_pcc = PearsonCorrCoef(num_outputs=num_labels).to(device)
    metric_kcc = KendallRankCorrCoef(num_outputs=num_labels).to(device)

    if train_loader is not None:
        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, torch.tensor(numpy.array(data.gt), dtype=torch.float, device=device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            gt_list_valid.append(torch.tensor(data.gt, device=device))
            out = model(data)
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid, dim=0)

    mae = metric_mae(preds, gt_list_valid).item()
    rse = metric_rse(preds, gt_list_valid).item()
    pcc = metric_pcc(preds, gt_list_valid).mean().item()
    kcc = metric_kcc(preds, gt_list_valid).mean().item()
    return train_loss, mae, rse, pcc, kcc

