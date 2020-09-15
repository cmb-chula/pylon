# Official implementation of Pyramid Localization Network (PYLON)

From the paper, Redesigning weakly supervised localization architectures for medical images.

**High accuracy localization:**
![high accuracy localization](figs/example4_paper.jpg)

**PYLON's architecture:**
![PYLON architecture](figs/figure_pylonv2.png)

## What's included

- Pytorch implementaiton of PYLON
- Additional results
- Reproducing the main results

## Additional results

1. [Picked localization images](figs/picked) (~20)
2. [All localization images](figs/all) (~1000)

## Reproducing results

### Requirements

Listed in requiremnets.txt:

- Pytorch 1.5
- Torchvision
- albumentations
- Segmentation models pytorch (a3cc9ac) 
- Nvidia's Apex (5d9b5cb)

You need to download the Chest X-Ray 14 dataset by yourself from https://nihcc.app.box.com/v/ChestXray-NIHCC.

Extract all the images into a single big directory `./data/images`, containing 100k images.

### Installing Segmentation models pytorch

```
git clone https://github.com/qubvel/segmentation_models.pytorch
cd segmentation_models.pytorch
git checkout a3cc9ac
pip install ./
```

### Installing Nvidia's APEX for mixed precision training

Install without C++ extensions: 

```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5d9b5cb
pip install ./
```

### Run

The following code will run PYLON with mix-precision on Chest X-Ray dataset:

```
python train.py
```

Stats will be available at `csv/pylon/0.csv` where `0` is the seed number. 

The checkpoint will be available at `save/pylon/0/best`.

Editing `train.py` to other seeds is straightforward. 

#### Evaluation

This step loads the best model (by validation loss), then runs it against test data.

##### AUROC

```
python eval_auc.py
```

Seed 0 result (the paper is 5 seeds):

| Atelectasis | Cardiomegaly | Effusion    | Infiltration | Mass        | Nodule      | Pneumonia   | Pneumothorax | Consolidation | Edema       | Emphysema   | Fibrosis   | Pleural_Thickening | Hernia      | micro       | macro       |
|-------------|--------------|-------------|--------------|-------------|-------------|-------------|--------------|---------------|-------------|-------------|------------|--------------------|-------------|-------------|-------------|
| 0.775346694 | 0.889010614  | 0.832113964 | 0.698973491  | 0.830520259 | 0.770125505 | 0.732714384 | 0.874371667  | 0.753967082   | 0.846177885 | 0.930146985 | 0.82124598 | 0.783688504        | 0.898773395 | 0.793113828 | 0.816941172 |

##### Point localization

```
python eval_loc.py
```

Seed 0 result (the paper is 5 seeds):

| Atelectasis | Cardiomegaly | Effusion    | Infiltration | Mass        | Nodule      | Pneumonia   | Pneumothorax | micro       |
|-------------|--------------|-------------|--------------|-------------|-------------|-------------|--------------|-------------|
| 0.511111111 | 1            | 0.424836601 | 0.715447154  | 0.717647059 | 0.455696203 | 0.733333333 | 0.193877551  | 0.604674797 |

Note: with 5 seeds the scores will be closer to that in the paper.

##### Visualizing the localization

```
python generate_heatmap.py
```

Generated heapmaps will be in `figs/picked` and `figs/all`. 