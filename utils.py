import torch
import torch.nn as nn
import torch.nn.init as init

import json

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.2, method='max', improved=False, intra=0.5, lamb=0.01):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.method = method
        self.improved = improved
        self.intra = intra
        self.lamb = lamb

    # im, sen : (n_samples, dim)
    def forward(self, im, sen):
        assert im.size() == sen.size()
        n_samples = im.size(0)
        # sim_mat : (n_samples, n_samples)
        # im * T.sen
        sim_mat = im.mm(sen.t())
        # sim_mat scores of positive sample
        pos = sim_mat.diag().view(-1, 1)
        # positive1, 2 : (n_samples, n_samples)
        # expand row
        positive1 = pos.expand_as(sim_mat)
        # expand collumn
        positive2 = pos.t().expand_as(sim_mat)

        # mask for diagonals
        # positive negative mask by boolian
        mask = (torch.eye(n_samples) > 0.5).to(sim_mat.device)
        # lossmat : (n_samples, n_samples)
        # caption negatives
        lossmat_i = (self.margin + sim_mat - positive1).clamp(min=0).masked_fill(mask, 0)
        # image negatives
        lossmat_c = (self.margin + sim_mat - positive2).clamp(min=0).masked_fill(mask, 0)
        # take only max of hinges loss
        if self.method == "max":
            # lossmat : (n_samples)
            lossmat_i = lossmat_i.max(dim=1)[0]
            lossmat_c = lossmat_c.max(dim=0)[0]
        # sum of hinges loss
        elif self.method == "sum":
            pass

        loss = lossmat_i.sum() + lossmat_c.sum()

        if self.improved:
            loss += self.lamb * ((self.intra - sim_mat.diag()).clamp(min=0).sum())

        return loss / n_samples


# collating function, restrict to 5 captions/image
def collater(data):
    out = {"image": [], "caption": [], "img_id": [], "ann_id": []}
    for obj in data:
        out["image"].append(obj["image"])
        out["caption"].append(obj["caption"])
        out["img_id"].append(obj["img_id"])
        out["ann_id"].append(obj["ann_id"])
    out["image"] = torch.stack(out["image"])
    return out

    

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.Embedding):
        init.uniform_(m.weight.data)

def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)


# build caption txt file from coco annotation json file
def cococap2txt(jsonfile, dst):
    sentences = []
    with open(jsonfile, 'r') as f:
        alldata = json.load(f)
    for ann in alldata["annotations"]:
        sentences.append(ann["caption"].strip())
    with open(dst, 'w+') as f:
        f.write("\n".join(sentences))