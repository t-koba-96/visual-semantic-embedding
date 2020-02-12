import sys, os, glob
import argparse
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from addict import Dict
import yaml

from dataset import CocoDset, EmbedDset
from utils import sec2str, collater
from model import ImageEncoder, CaptionEncoder
from vocab import Vocabulary


def get_arguments():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--vocab_path', type=str, default='captions_train2017.txt')
    parser.add_argument('--method', type=str, default="PCA", help='Name of dimensionality reduction method, should be {T-SNE | PCA}')

    # training config
    parser.add_argument('--no_cuda', action='store_true', help="disable gpu training")
    parser.add_argument('--device', type=list, default=[1], help='choose device')


    # retrieval config
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint, will load model from there')


    args = parser.parse_args()

    return args


def dimension_reduction(numpyfile, dstfile, method="PCA"):
    all = np.load(numpyfile)
    begin = time.time()
    print("conducting {} on data...".format(method), flush=True)
    if method == "T-SNE":
        all = TSNE(n_components=2).fit_transform(all)
    elif method == "PCA":
        all = PCA(n_components=2).fit_transform(all)
    else:
        raise NotImplementedError()
    print("done | {} ".format(sec2str(time.time()-begin)), flush=True)
    np.save(dstfile, all)
    print("saved {} embeddings to {}".format(method, dstfile), flush=True)

def plot_embeddings(numpyfile, n_v, out_file, method="PCA"):
    all = np.load(numpyfile)
    assert all.shape[1] == 2
    fig = plt.figure(clear=True)
    fig.suptitle("visualization of embeddings using {}".format(method))
    plt.scatter(all[:n_v, 0], all[:n_v, 1], s=2, c="red", label="image")
    plt.scatter(all[n_v::5, 0], all[n_v::5, 1], s=2, c="blue", label="caption")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(out_file)
    print("saved {} plot to {}".format(method, out_file), flush=True)


def main():
    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'), encoding='utf8')))
    print(args)
    args.device = list (map(str,args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device)

    # image transformer
    transform = transforms.Compose([
        transforms.Resize((SETTING.imsize, SETTING.imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    if args.dataset == 'coco':
        val_dset = CocoDset(root=SETTING.root_path, img_dir='val2017', ann_dir='annotations/captions_val2017.json', transform=transform)
    val_loader = DataLoader(val_dset, batch_size=SETTING.batch_size, shuffle=False, num_workers=SETTING.n_cpu, collate_fn=collater)

    vocab = Vocabulary(max_len=SETTING.max_len)
    vocab.load_vocab(args.vocab_path)

    imenc = ImageEncoder(SETTING.out_size, SETTING.cnn_type)
    capenc = CaptionEncoder(len(vocab), SETTING.emb_size, SETTING.out_size, SETTING.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    imenc = imenc.to(device)
    capenc = capenc.to(device)

    assert args.checkpoint is not None
    print("loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    imenc.load_state_dict(ckpt["encoder_state"])
    capenc.load_state_dict(ckpt["decoder_state"])

    begin = time.time()
    dset = EmbedDset(val_loader, imenc, capenc, vocab, args)
    print("database created | {} ".format(sec2str(time.time()-begin)), flush=True)

    savedir = os.path.join("out", args.config_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, 0o777)

    image = dset.embedded["image"]
    caption = dset.embedded["caption"]
    n_i = image.shape[0]
    n_c = caption.shape[0]
    all = np.concatenate([image, caption], axis=0)

    emb_file = os.path.join(savedir, "embedding_{}.npy".format(n_i))
    save_file = os.path.join(savedir, "{}.npy".format(SETTING.method))
    vis_file = os.path.join(savedir, "{}.png".format(SETTING.method))
    np.save(emb_file, all)
    print("saved embeddings to {}".format(emb_file), flush=True)
    dimension_reduction(emb_file, save_file, method=SETTING.method)
    plot_embeddings(save_file, n_i, vis_file, method=SETTING.method)

if __name__ == '__main__':
    main()