import torch
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef, KendallRankCorrCoef


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    train_loss = 0
    num_labels = model.classes

    metric_mae = MeanAbsoluteError().to(device)
    metric_rse = RelativeSquaredError(num_outputs=num_labels).to(device)
    metric_pcc = PearsonCorrCoef(num_outputs=num_labels).to(device)
    metric_kcc = KendallRankCorrCoef(num_outputs=num_labels).to(device)
    if train_loader is not None:
        model.train()
        for data in train_loader:
            voxel, seq, globf, gt = data
            out = model((voxel.to(device), seq.to(device), globf.to(device)))
            loss = criterion(out, gt.to(device).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            voxel, seq, globf, gt = data
            gt_list_valid.append(gt.to(device))
            out = model((voxel.to(device), seq.to(device), globf.to(device)))
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid, dim=0)

    mae = metric_mae(preds, gt_list_valid).item()
    rse = metric_rse(preds, gt_list_valid).item()
    pcc = metric_pcc(preds, gt_list_valid).mean().item()
    kcc = metric_kcc(preds, gt_list_valid).mean().item()
    return train_loss, mae, rse, pcc, kcc

