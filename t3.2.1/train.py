import torch
from torchmetrics import F1Score, Accuracy, AveragePrecision, AUROC


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    train_loss = 0
    num_labels = model.classes
    avg = args.metric_avg
    if num_labels == 1:
        task = 'binary'
    else:
        task = 'multilabel'
    metric_macro_acc = Accuracy(average=avg, task=task, num_labels=num_labels, threshold=0.5).to(device)
    metric_macro_f1 = F1Score(average=avg, task=task, num_labels=num_labels, threshold=0.5).to(device)
    metric_macro_ap = AveragePrecision(average=avg, task=task, num_labels=num_labels).to(device)
    metric_auc = AUROC(average=avg, task=task, num_labels=num_labels).to(device)

    if train_loader is not None:
        model.train()
        for data in train_loader:
            voxel, seq, gt = data
            # print(seq_lengths)
            out = model((voxel.to(device), seq.to(device)))
            # print(out[0])
            # print(gt[0])
            loss = criterion(out, gt.to(device).float())
            # loss_0 = criterion(out[0], gt.to(device).float())
            # loss_1 = criterion(out[1], gt.to(device).float())
            # loss_2 = criterion(out[2], gt.to(device).float())
            # loss = loss_0 + loss_1 + loss_2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            voxel, seq, gt = data
            gt_list_valid.append(gt.to(device))
            out = model((voxel.to(device), seq.to(device)))
            preds.append(out)

    # calculate metrics
    preds = torch.nn.functional.sigmoid(torch.cat(preds, dim=0))
    gt_list_valid = torch.cat(gt_list_valid, dim=0).int()
    # if train_loader is None:
    #     print(preds)
    #     print(preds.shape)
    #     print(gt_list_valid)
    #     print(gt_list_valid.shape)

    macro_ap = metric_macro_ap(preds, gt_list_valid).item()
    # class_ap = [round(i.item(), 5) for i in metric_class_ap(preds, gt_list_valid)]
    macro_auc = metric_auc(preds, gt_list_valid).item()
    macro_f1 = metric_macro_f1(preds, gt_list_valid).item()
    macro_acc = metric_macro_acc(preds, gt_list_valid).item()
    return train_loss, macro_ap, macro_f1, macro_acc, macro_auc

