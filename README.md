
# Visual Semantic Embedding   

## Installation  

```bash
$ python -m spacy download en
```

### Pip users
`pip install -r requirements.txt`

### Dataset
Run `download_coco.sh`


## Training
```bash
$ python train.py --root_path $ROOTPATH
```


## Evaluation, Visualization
```bash
$ python eval.py --root_path $ROOTPATH --checkpoint hogehoge.ckpt --image_path $IMAGE --caption $CAPTION
```

[References](https://github.com/skasai5296/VSE)