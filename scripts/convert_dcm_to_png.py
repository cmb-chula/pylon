import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from pydicom import dcmread
from tqdm.autonotebook import tqdm

cv2.setNumThreads(1)


def resize_ratio(h, w, target):
    ratio = target / max(h, w)
    return ratio


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(),
                                         number_bins,
                                         density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = image.max() * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def process_dcm(dcm, equalize=True):
    # invert
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        # invert
        img = dcm.pixel_array
        if hasattr(dcm, 'WindowWidth'):
            img = float(dcm.WindowWidth) - img
        else:
            img = dcm.pixel_array.max() - img
    elif dcm.PhotometricInterpretation == 'MONOCHROME2':
        img = dcm.pixel_array
    else:
        raise NotImplementedError()

    # min, max normalize
    img = img - img.min()
    img = img / img.max()

    # normalization
    if equalize:
        img, cdf = image_histogram_equalization(img)
    img = (img * 255).astype(np.uint8)

    # resize to 1024
    ratio = resize_ratio(h=dcm.Rows, w=dcm.Columns, target=1024)
    h, w = (np.array(img.shape) * ratio).astype(int)
    img = cv2.resize(img, (w, h))
    return img, ratio


class Converter:
    def __init__(self, image_path, target_path, equalize) -> None:
        self.image_path = image_path
        self.target_path = target_path
        self.equalize = equalize

    def __call__(self, name):
        dcm = dcmread(f'{self.image_path}/{name}.dicom')
        img, ratio = process_dcm(dcm, self.equalize)
        cv2.imwrite(f'{self.target_path}/{name}.png', img)
        return ratio


class ResizeRatio:
    def __init__(self, image_path) -> None:
        self.image_path = image_path

    def __call__(self, name):
        dcm = dcmread(f'{self.image_path}/{name}.dicom',
                      stop_before_pixels=True)
        return resize_ratio(h=dcm.Rows, w=dcm.Columns, target=1024)


def list_image_names(image_path):
    image_names = [
        os.path.splitext(each)[0] for each in sorted(os.listdir(image_path))
        if '.dicom' in each
    ]
    return image_names


def convert_dcm_to_png(image_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    image_names = list_image_names(image_path)
    convert = Converter(image_path, target_path, equalize=True)

    with Pool(processes=5) as pool:
        out = {'image_id': [], 'resize_ratio': []}
        for name, ratio in tqdm(zip(image_names,
                                    pool.imap(convert, image_names)),
                                total=len(image_names)):
            out['image_id'].append(name)
            out['resize_ratio'].append(ratio)

        out = pd.DataFrame(out)
        out.to_csv(f'{target_path}_ratio.csv', index=False)


def get_all_resize_ratios(image_path, target_path):
    image_names = list_image_names(image_path)
    processor = ResizeRatio(image_path)
    with Pool(processes=5) as pool:
        out = {'image_id': [], 'resize_ratio': []}
        for name, ratio in tqdm(
                zip(
                    image_names,
                    pool.imap(processor, image_names),
                ),
                total=len(image_names),
        ):
            out['image_id'].append(name)
            out['resize_ratio'].append(ratio)

        out = pd.DataFrame(out)
        out.to_csv(f'{target_path}_ratio.csv', index=False)


if __name__ == '__main__':
    image_path = 'path to directiory containing dicoms'
    target_path = 'target path will contain pngs'

    convert_dcm_to_png(image_path, target_path)
    get_all_resize_ratios(image_path, target_path)
