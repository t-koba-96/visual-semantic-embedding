import argparse
import time

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import faiss
import skimage.io as io
from addict import Dict
import yaml

from dataset import CocoDset, EmbedDset
from utils import sec2str, collater
from model import ImageEncoder, CaptionEncoder
from vocab import Vocabulary

def get_arguments():
    parser = argparse.ArgumentParser()

    # Setting file
    parser.add_argument('arg', type=str, help='arguments file name')

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--vocab_path', type=str, default='captions_train2017.txt')


    # training config
    parser.add_argument('--no_cuda', action='store_true', help="disable gpu training")
    parser.add_argument('--device', type=list, default=[1], help='choose device')

    # retrieval config
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint, will load model from there')
    parser.add_argument('--image_path', type=str, default='demo/sample1.jpg')
    parser.add_argument('--output_dir', type=str, default='demo')
    parser.add_argument('--caption', type=str, default='the cat is walking on the street')

    args = parser.parse_args()

    return args


def retrieve_i2c(dset, v_dset, impath, imenc, transform, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im = Image.open(impath)
    print("-"*50)
    plt.title("source image")
    plt.imshow(np.asarray(im))
    plt.axis('off')
    plt.show(block=False)
    plt.show()

    im = transform(im).unsqueeze(0)
    begin = time.time()
    with torch.no_grad():
        im = im.to(device)
        im = imenc(im)
    im = im.cpu().numpy()
    cap = dset.embedded["caption"]
    nd = cap.shape[0]
    d = cap.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    print("# captions: {}, dimension: {}".format(nd, d), flush=True)

    # im2cap
    cpu_index.add(cap)
    D, I = cpu_index.search(im, k)
    nnann = []
    for i in range(k):
        nnidx = I[0, i]
        ann_ids = [a for ids in dset.embedded["ann_id"] for a in ids]
        nnann_id = ann_ids[nnidx]
        nnann.append(nnann_id)
    anns = v_dset.coco.loadAnns(nnann)
    print("retrieval time {}".format(sec2str(time.time()-begin)), flush=True)
    print("-"*50)
    print("{} nearest neighbors of image:".format(k))
    v_dset.coco.showAnns(anns)
    print("-"*50)


def retrieve_c2i(dset, v_dset, savedir, caption, capenc, vocab, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    begin = time.time()
    print("-"*50)
    print("source caption: '{}'".format(caption), flush=True)
    cap = vocab.return_idx([caption])
    length = torch.tensor([torch.sum(torch.ne(cap, vocab.padidx)).item()]).to(device, dtype=torch.long)
    with torch.no_grad():
        cap = cap.to(device)
        cap = capenc(cap, length)
    cap = cap.cpu().numpy()
    im = dset.embedded["image"]
    nd = im.shape[0]
    d = im.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    print("# images: {}, dimension: {}".format(nd, d), flush=True)

    # cap2im
    cpu_index.add(im)
    D, I = cpu_index.search(cap, k)
    print("retrieval time {}".format(sec2str(time.time()-begin)), flush=True)
    nnimid = []
    for i in range(k):
        nnidx = I[0, i]
        nnim_id = dset.embedded["img_id"][nnidx]
        nnimid.append(nnim_id)
    img = v_dset.coco.loadImgs(nnimid)
    print("-"*50)
    print("{} nearest neighbors of '{}'".format(k, caption))
    if k == 1:
        plt.figure(figsize=(8, 10))
        nnim = io.imread(img[0]['coco_url'])
        plt.imshow(nnim)
        plt.axis('off')
    elif k > 1:
        fig, axs = plt.subplots(1, k, figsize=(8*k, 10))
        fig.suptitle("retrieved {} nearest neighbors of '{}'".format(k, caption))
        for i in range(k):
            nnim = io.imread(img[i]['coco_url'])
            axs[i].imshow(nnim)
            axs[i].axis('off')
    else:
        raise
    #plt.show(block=False)
    #plt.show()
    if not os.path.exists(savedir):
         os.makedirs(savedir)
    plt.savefig(os.path.join(savedir, "output.png"))
    print("-"*50)


def main():

    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'), encoding='utf8')))
    print(args)
    args.device = list (map(str,args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device)

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
    ckpt = torch.load(args.checkpoint)
    imenc.load_state_dict(ckpt["encoder_state"])
    capenc.load_state_dict(ckpt["decoder_state"])

    begin = time.time()
    dset = EmbedDset(val_loader, imenc, capenc, vocab, args)
    print("database created | {} ".format(sec2str(time.time()-begin)), flush=True)

    retrieve_i2c(dset, val_dset, args.image_path, imenc, transform)
    retrieve_c2i(dset, val_dset, args.output_dir, args.caption, capenc, vocab)



if __name__ == '__main__':
    main()