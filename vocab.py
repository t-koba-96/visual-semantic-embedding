"""
python file which makes the vocab dictionary from sentence_batch 
"""

import sys, os
import time
import json
import pickle

import torch
import spacy
import torchtext

from utils import sec2str , cococap2txt

sp = spacy.load('en')


class Vocabulary():
    def __init__(self, min_freq=5, max_len=30):
        self.min_freq = min_freq
        self.max_len = max_len
        self.text_proc = torchtext.data.Field(sequential=True, init_token="<bos>", eos_token="<eos>", lower=True, fix_length=self.max_len, tokenize="spacy", batch_first=True)

    """
    build vocabulary from textfile.
    """
    # making the vocab dict
    def load_vocab(self, textfile):
        before = time.time()
        print("building vocabulary...", flush=True)
        # Append each line of txt to list 
        with open(textfile, 'r') as f:
            sentences = f.readlines()
        # divide to words and punctuation 
        sent_proc = list(map(self.text_proc.preprocess, sentences))
        # make vocab dictionary
        self.text_proc.build_vocab(sent_proc, min_freq=self.min_freq)
        self.len = len(self.text_proc.vocab)
        # padding index
        self.padidx = self.text_proc.vocab.stoi["<pad>"]
        print("done building vocabulary, minimum frequency is {} times".format(self.min_freq), flush=True)
        print("# of words in vocab: {} | {}".format(self.len, sec2str(time.time()-before)), flush=True)

    # sentence list (str) → index list(torch.Longtensor)
    def return_idx(self, sentence_batch):
        out = []
        preprocessed = list(map(self.text_proc.preprocess, sentence_batch))
        out = self.text_proc.process(preprocessed)
        return out

    # index list(torch.Longtensor) → sentence list (str)
    def return_sentences(self, ten):
        if isinstance(ten, torch.Tensor):
            ten = ten.tolist()
        out = []
        for idxs in ten:
            tokenlist = [self.text_proc.vocab.itos[idx] for idx in idxs]
            out.append(" ".join(tokenlist))
        return out

    def __len__(self):
        return self.len



if __name__ == '__main__':

    file = "/home/takuya/workspace/datasets/coco2017/annotations/captions_train2017.json"
    dest = "captions_train2017.txt"
    # first time only
    if not os.path.exists(dest):
        cococap2txt(file, dest)
    vocab = Vocabulary()
    vocab.load_vocab(dest)
    sentence = ["The cat and the hat sat on a mat."]
    ten = vocab.return_idx(sentence)
    print(ten)
    sent = vocab.return_sentences(ten)
    print(sent)