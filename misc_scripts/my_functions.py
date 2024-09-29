
from PIL import Image
import numpy as np
import os
import os.path
import glob
import warnings
import shutil
import random
from scipy import ndimage
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def read_nifti(filepath_image, filepath_label=False):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    try:
        lbl = nib.load(filepath_label)
        label_data = lbl.get_fdata()
    except:
        label_data = 0

    return image_data, label_data, img

def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)





