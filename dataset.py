import os
import json as jsonmod

import numpy as np
import torch
import torchtext
import spacy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import nltk
from PIL import Image
from pycocotools.coco import COCO
from addict import Dict
import yaml

from vocab import Vocabulary

sp = spacy.load("en")


class CocoDset(Dataset):

    def __init__(self, root, img_dir='train2017', ann_dir='annotations/captions_train2017.json', transform=None):
        """
        Args:
            img_dir: imgfile path.
            ann_dir: annotation file path.
            transform: image transformer.
        """
        self.coco = COCO(os.path.join(root, ann_dir))
        self.img_dir = os.path.join(root, img_dir)

        self.imgids = self.coco.getImgIds()
        self.annids = [self.coco.getAnnIds(id) for id in self.imgids]
        self.transform = transform

    def __getitem__(self, index):

        img_id = self.imgids[index]
        ann_id = self.annids[index]
        # get img 
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        # get caption
        caption = [obj['caption'] for obj in self.coco.loadAnns(ann_id)]

        # 1 img　←→  5 captions
        caption = caption[:5]
        ann_id = ann_id[:5]

        # image transform
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        return {"image": image, "caption": caption, "index": index, "img_id": img_id, "ann_id": ann_id}

    def __len__(self):
        return len(self.imgids)


class EmbedDset(Dataset):
    """Makes the embeded data from dataloader"""

    def __init__(self, loader, image_model, caption_model, vocab, args):
        """
        Args:
            loader: DataLoader for validation images and captions
            model: trained model to evaluate
        """
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.embedded = {"image": [], "caption": [], "img_id": [], "ann_id": []}
        for data in loader:
            im = data["image"]
            caption = data["caption"]
            caption = [c for cap in caption for c in cap]
            cap = vocab.return_idx(caption)
            lengths = cap.ne(vocab.padidx).sum(dim=1).to(device)
            im = im.to(device)
            cap = cap.to(device)
            with torch.no_grad():
                emb_im = image_model(im)
                emb_cap = caption_model(cap, lengths)
            self.embedded["image"].append(emb_im.cpu().numpy())
            self.embedded["caption"].append(emb_cap.cpu().numpy())
            self.embedded["img_id"].extend(data["img_id"])
            self.embedded["ann_id"].extend(data["ann_id"])
        self.embedded["image"] = np.concatenate(self.embedded["image"], axis=0)
        self.embedded["caption"] = np.concatenate(self.embedded["caption"], axis=0)

    def __len__(self):
        return len(self.embedded["img_id"])




if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    SETTING = Dict(yaml.safe_load(open(os.path.join('arguments',args.arg+'.yaml'), encoding='utf8')))
    cocodset = CocoDset(root=SETTING.root_path, transform=transform)
    print(cocodset[10000])