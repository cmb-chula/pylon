# Official implementation of Pyramid Localization Network (PYLON)

From the paper, "High resolution weakly supervised localization architectures for medical images". [(Preprint)](https://arxiv.org/abs/2010.11475)

**High accuracy localization:**
![high accuracy localization](figs/example4_paper.jpg)

**PYLON's architecture:**
![PYLON architecture](figs/figure_pylon_crop.png)

## What's included

- Pytorch implementation of PYLON
    * [Pretrained weights](https://drive.google.com/file/d/1v26dU21hjePidW5crSXsrpf3OCJWLoWp/view?usp=sharing)
- Additional results and heatmaps 
- Code to reproduce the main results

## Additional results

1. [Picked localization images](figs/picked) (~20)

    * [PYLON](figs/picked)
    * [Backbone](figs/picked-backbone)
    * [DeeplabV3+](figs/picked-deeplabv3+)
    * [DeeplabV3+ (No GAP)](figs/picked-deeplabv3+,nogap)
    * [FPN](figs/picked-fpn)
    * [FPN (BN)](figs/picked-fpn,bn)
    * [Li 2018](figs/picked-li2018)
    * [PAN](figs/picked-pan)
    * [UNET](figs/picked-unet)

2. [All localization images](figs/all) (~1000)

## Reproducing results

### Requirements

It was tested with Pytorch 1.7.1.

Install other related libraries:

```
pip install -r requirements.txt
```

### Preparing datasets


#### NIH's Chest X-Ray 14
You need to download the Chest X-Ray 14 dataset by yourself from https://nihcc.app.box.com/v/ChestXray-NIHCC.

Extract all the images into a single big directory `data/nih14/images`, containing 100k images.

#### VinDr-CXR

Download the DICOM version from Kaggle https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview

You need to convert them into PNG. The script is provided in `scripts/convert_dcm_to_png.py`. The conversion will not alter the aspect ratio it will aim for maximum 1024 either width or height.

Put all the PNG files into directory `data/vin/images`.

### Run

The main run files are `train_nih.py` and `train_vin.py`. The files are straightforward to edit. Make changes or read before you run:

```
python train_nih.py
```

And

```
python train_vin.py
```

Stats will be available at `eval_auc` and `eval_loc`. 

The figures will be available at `figs/picked` and `figs/all`.