import sys, os
import time
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
import torchvision

from utils import sec2str, weight_init

# L2 normalize a batched tensor (bs, ft)
def l2normalize(ten):
    norm = torch.norm(ten, dim=1, keepdim=True)
    return ten / norm


# image feature extractor
class ImageEncoder(nn.Module):
    def __init__(self, out_size=256, cnn_type="resnet18", pretrained=True):
        super(ImageEncoder, self).__init__()
        # get pretrained model
        self.cnn = getattr(torchvision.models, cnn_type)(pretrained)
        # replace final fc layer to output size
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.model.classifier._modules['6'].in_features,
                                out_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.fc.in_features, out_size)
            self.cnn.fc = nn.Sequential()
        if not pretrained:
            self.cnn.apply(weight_init)
        self.fc.apply(weight_init)

    def forward(self, x):
        resout = self.cnn(x)
        out = self.fc(resout)
        normed_out = l2normalize(out)
        return normed_out


# caption feature extractor
class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=256, out_size=256, rnn_type="LSTM", padidx=0):
        super(CaptionEncoder, self).__init__()
        self.out_size = out_size
        self.padidx = padidx
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = getattr(nn, rnn_type)(emb_size, out_size, batch_first=True)

        self.emb.apply(weight_init)
        self.rnn.apply(weight_init)

    # x(sentence_idx): batch_size * len_sentence
    # lengths(len_sentence without padding): batch_size
    def forward(self, x, lengths):
        # vocab_size to emb_size 
        emb = self.emb(x)
        # input: batch_size * len_sentence * emb_size
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        # lstm
        self.rnn.flatten_parameters()
        output, _ = self.rnn(packed)
        # output: batch_size * len_sentence * out_size
        output = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)[0]
        # lengths(len_sentence without padding - 1): batch_size * 1 * out_size
        lengths = lengths.view(-1, 1, 1).expand(-1, -1, self.out_size) - 1
        # out(uses the last output before padding part): bs * out_size
        out = torch.gather(output, 1, lengths).squeeze(1)
        normed_out = l2normalize(out)
        return normed_out



if __name__ == '__main__':
    ten = torch.randn((16, 3, 224, 224))

    cnn = ImageEncoder()
    out = cnn(ten)
    print(out.size())

    cap = CaptionEncoder(vocab_size=100)
    seq = torch.randint(100, (16, 30), dtype=torch.long)
    len = torch.randint(1, 31, (16,), dtype=torch.long)
    out = cap(seq, len)
    print(out.size())