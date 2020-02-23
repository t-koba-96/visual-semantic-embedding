import sys, os
import argparse
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import torchvision
import torchvision.transforms as transforms
import faiss
from addict import Dict
import yaml

from dataset import CocoDset, EmbedDset
from utils import sec2str, PairwiseRankingLoss, collater
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
    parser.add_argument('--improved', action='store_true', help="improved triplet loss")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")
    parser.add_argument('--device', type=list, default=[1], help='choose device')
    parser.add_argument('--dataparallel', action='store_true', help='use data parallel')

    # checkpoint
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint if any, will restart training from there')


    args = parser.parse_args()

    return args



def train(epoch, loader, imenc, capenc, optimizer, lossfunc, vocab, args, SETTING):
    begin = time.time()
    # max iteration_num
    maxit = int(len(loader.dataset) / SETTING.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cumloss = 0
    for it, data in enumerate(loader):
        """image, target, index, img_id"""
        # batch_size * img_size
        image = data["image"]
        # batch_size * sentence * 5 
        caption = data["caption"]
        # chose 1 random caption from 5  
        caption = [i[np.random.randint(0, len(i))] for i in caption]
        img_id = data["img_id"]
        # caption sentence → id
        target = vocab.return_idx(caption)
        # lengths of each sentence
        lengths = target.ne(vocab.padidx).sum(dim=1)

        optimizer.zero_grad()

        image = image.to(device)
        target = target.to(device)
        lengths = lengths.to(device)

        im_emb = imenc(image)
        cap_emb = capenc(target, lengths)
        lossval = lossfunc(im_emb, cap_emb)
        lossval.backward()

        # clip gradient norm
        if SETTING.grad_clip > 0:
            clip_grad_norm_(imenc.parameters(), SETTING.grad_clip)
            clip_grad_norm_(capenc.parameters(), SETTING.grad_clip)
        optimizer.step()
        cumloss += lossval.item()


        if it % SETTING.log_every == SETTING.log_every-1:
            print("epoch {} | {} | {:06d}/{:06d} iterations | loss: {:.08f}".format(epoch, sec2str(time.time()-begin), it+1, maxit, cumloss/SETTING.log_every), flush=True)
            cumloss = 0




def validate(epoch, loader, imenc, capenc, vocab, args, SETTING):
    begin = time.time()
    print("begin validation for epoch {}".format(epoch), flush=True)
    dset = EmbedDset(loader, imenc, capenc, vocab, args)
    print("val dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    im = dset.embedded["image"]
    cap = dset.embedded["caption"]

    nd = im.shape[0]
    nq = cap.shape[0]
    d = im.shape[1]
    cpu_index = faiss.IndexFlatIP(d)

    print("# images: {}, # captions: {}, dimension: {}".format(nd, nq, d), flush=True)

    # im2cap
    cpu_index.add(cap)
    # calculate every conbination and sort 
    # D = result , I = imgid
    D, I = cpu_index.search(im, nq)
    data = {}
    allrank = []
    # TODO: Make more efficient, do not hardcode 5
    cap_per_image = 5
    # brinf correct answer rank for each sentence(their are 5 each)
    for i in range(cap_per_image):
        gt = (np.arange(nd) * cap_per_image).reshape(-1, 1) + i
        rank = np.where(I == gt)[1]
        allrank.append(rank)
    allrank = np.stack(allrank)
    # minimal rank for ans(best of 5 each)
    allrank = np.amin(allrank, 0)
    # how many images were correct bellow @num
    for rank in [1, 5, 10, 20]:
        data["i2c_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / len(allrank)
    data["i2c_median@r"] = np.median(allrank) + 1
    data["i2c_mean@r"] = np.mean(allrank)

    # cap2im
    cpu_index.reset()
    cpu_index.add(im)
    D, I = cpu_index.search(cap, nd)
    # TODO: Make more efficient, do not hardcode 5
    gt = np.arange(nq).reshape(-1, 1) // cap_per_image
    allrank = np.where(I == gt)[1]
    for rank in [1, 5, 10, 20]:
        data["c2i_recall@{}".format(rank)] = 100 * np.sum(allrank < rank) / len(allrank)
    data["c2i_median@r"] = np.median(allrank) + 1
    data["c2i_mean@r"] = np.mean(allrank)

    print("-"*50)
    print("results of cross-modal retrieval")
    for key, val in data.items():
        print("{}: {}".format(key, val), flush=True)
    print("-"*50)
    return data




def visualize(metrics, args, SETTING):
    savedir = os.path.join("out", args.arg)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Recall for lr={}, margin={}, bs={}".format(float(SETTING.lr_cnn), SETTING.margin, SETTING.batch_size))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Recall")
    for k, v in metrics.items():
        if "recall" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savedir, "recall.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Median ranking for lr={}, margin={}, bs={}".format(float(SETTING.lr_cnn), SETTING.margin, SETTING.batch_size))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Median")
    for k, v in metrics.items():
        if "median" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savedir, "median.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Mean ranking for lr={}, margin={}, bs={}".format(float(SETTING.lr_cnn), SETTING.margin, SETTING.batch_size))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean")
    for k, v in metrics.items():
        if "mean" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savedir, "mean.png"))



def main():

    # ignore warnings
    #warnings.simplefilter('ignore')

    args = get_arguments()
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'), encoding='utf8')))
    print(args)
    args.device = list (map(str,args.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device)

    #image transformer
    train_transform = transforms.Compose([
        transforms.Resize(SETTING.imsize_pre),
        transforms.RandomCrop(SETTING.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    val_transform = transforms.Compose([
        transforms.Resize(SETTING.imsize_pre),
        transforms.CenterCrop(SETTING.imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    # data load
    if args.dataset == 'coco':
        train_dset = CocoDset(root=SETTING.root_path,img_dir='train2017', ann_dir='annotations/captions_train2017.json', transform=train_transform)
        val_dset = CocoDset(root=SETTING.root_path, img_dir='val2017', ann_dir='annotations/captions_val2017.json', transform=val_transform)
    train_loader = DataLoader(train_dset, batch_size=SETTING.batch_size, shuffle=True, num_workers=SETTING.n_cpu, collate_fn=collater)
    val_loader = DataLoader(val_dset, batch_size=SETTING.batch_size, shuffle=False, num_workers=SETTING.n_cpu, collate_fn=collater)

    # setup vocab dict
    vocab = Vocabulary(max_len=SETTING.max_len)
    vocab.load_vocab(args.vocab_path)

    # setup encoder
    imenc = ImageEncoder(SETTING.out_size, SETTING.cnn_type)
    capenc = CaptionEncoder(len(vocab), SETTING.emb_size, SETTING.out_size, SETTING.rnn_type, vocab.padidx)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    imenc = imenc.to(device)
    capenc = capenc.to(device)

    # learning rate
    cfgs = [{'params' : imenc.fc.parameters(), 'lr' : float(SETTING.lr_cnn)},
            {'params' : capenc.parameters(), 'lr' : float(SETTING.lr_rnn)}]

    # optimizer
    if SETTING.optimizer == 'SGD':
        optimizer = optim.SGD(cfgs, momentum=SETTING.momentum, weight_decay=SETTING.weight_decay)
    elif SETTING.optimizer == 'Adam':
        optimizer = optim.Adam(cfgs, betas=(SETTING.beta1, SETTING.beta2), weight_decay=SETTING.weight_decay)
    elif SETTING.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(cfgs, alpha=SETTING.alpha, weight_decay=SETTING.weight_decay)
    if SETTING.scheduler == 'Plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=SETTING.dampen_factor, patience=SETTING.patience, verbose=True)
    elif SETTING.scheduler == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SETTING.patience, gamma=SETTING.dampen_factor)

    # loss
    lossfunc = PairwiseRankingLoss(margin=SETTING.margin, method=SETTING.method, improved=args.improved, intra=SETTING.intra, lamb=SETTING.imp_weight)


    # if start from checkpoint
    if args.checkpoint is not None:
        print("loading model and optimizer checkpoint from {} ...".format(args.checkpoint), flush=True)
        ckpt = torch.load(args.checkpoint)
        imenc.load_state_dict(ckpt["encoder_state"])
        capenc.load_state_dict(ckpt["decoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if SETTING.scheduler != 'None':
            scheduler.load_state_dict(ckpt["scheduler_state"])
        offset = ckpt["epoch"]
        data = ckpt["stats"]
        bestscore = 0
        for rank in [1, 5, 10, 20]:
            bestscore += data["i2c_recall@{}".format(rank)] + data["c2i_recall@{}".format(rank)]
        bestscore = int(bestscore)
    # start new training
    else:
        offset = 0
        bestscore = -1
    
    if args.dataparallel:
        print("Using Multiple GPU . . . ")
        imenc = nn.DataParallel(imenc)
        capenc = nn.DataParallel(capenc)

    metrics = {}
    es_cnt = 0
   
    # training
    assert offset < SETTING.max_epochs
    for ep in range(offset, SETTING.max_epochs):

        epoch = ep+1

        # unfreeze cnn parameters
        if epoch == SETTING.freeze_epoch:
            if args.dataparallel:
                optimizer.add_param_group({'params': imenc.module.cnn.parameters(), 'lr': float(SETTING.lr_cnn)})
            else:
                optimizer.add_param_group({'params': imenc.cnn.parameters(), 'lr': float(SETTING.lr_cnn)})


        #train(1epoch)
        train(epoch, train_loader, imenc, capenc, optimizer, lossfunc, vocab, args, SETTING)

        #validate
        data = validate(epoch, val_loader, imenc, capenc, vocab, args, SETTING)
        totalscore = 0
        for rank in [1, 5, 10, 20]:
            totalscore += data["i2c_recall@{}".format(rank)] + data["c2i_recall@{}".format(rank)]
        totalscore = int(totalscore)

        #scheduler update
        if SETTING.scheduler == 'Plateau':
            scheduler.step(totalscore)
        if SETTING.scheduler == 'Step':
            scheduler.step()

        # update checkpoint
        if args.dataparallel:
            ckpt = {
                    "stats": data,
                    "epoch": epoch,
                    "encoder_state": imenc.module.state_dict(),
                    "decoder_state": capenc.module.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                    }
        else:
            ckpt = {
                    "stats": data,
                    "epoch": epoch,
                    "encoder_state": imenc.state_dict(),
                    "decoder_state": capenc.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                    }

                
        if SETTING.scheduler != 'None':
            ckpt['scheduler_state'] = scheduler.state_dict()

        # make savedir
        savedir = os.path.join("models", args.arg)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        #
        for k, v in data.items():
            if k not in metrics.keys():
                metrics[k] = [v]
            else:
                metrics[k].append(v)

        # save checkpoint
        savepath = os.path.join(savedir, "epoch_{:04d}_score_{:03d}.ckpt".format(epoch, totalscore))
        if int(totalscore) > int(bestscore):
            print("score: {:03d}, saving model and optimizer checkpoint to {} ...".format(totalscore, savepath), flush=True)
            bestscore = totalscore
            torch.save(ckpt, savepath)
            es_cnt = 0
        else:
            print("score: {:03d}, no improvement from best score of {:03d}, not saving".format(totalscore, bestscore), flush=True)
            es_cnt += 1
            # early stopping
            if es_cnt == SETTING.es_cnt:
                print("early stopping at epoch {} because of no improvement for {} epochs".format(epoch, SETTING.es_cnt))
                break

        
        print("done for epoch {:04d}".format(epoch), flush=True)

    visualize(metrics, args, SETTING)
    print("complete training")

if __name__ == '__main__':
    main()