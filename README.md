
# Visual Semantic Embedding   

## Installation  

```bash
$ python -m spacy download en
```

### Pip users
```bash
$ pip install -r requirements.txt
```

### Dataset
```bash
$ download_coco.sh
```


## Training
```bash
$ python train.py --root_path $ROOTPATH
```

## Trained model  
Download the trained model from [link](https://drive.google.com/drive/u/1/folders/19FgmEiWzEPTs-L6hsGzZssRzR952Mddt) to just try eval.py. Make sure to change the checkpoint arguments to the downloaded model path. 


## Evaluation, Visualization
```bash
$ python eval.py --root_path $ROOTPATH --checkpoint *.ckpt --image_path $IMAGE --caption $CAPTION
```

[References](https://github.com/skasai5296/VSE)