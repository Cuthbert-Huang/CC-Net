import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import nibabel as nib
import pandas as pd
import xlrd
import pdb
import SimpleITK as sitk
from skimage import transform, measure
import os
import pydicom
import matplotlib.pyplot as plt
#%matplotlib inline
output_size =[96,96,96]
def ImageResample(sitk_image, new_spacing = [1,1,1], is_label = False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]
    if not is_label:
        print(size)
        print(new_size)
        print(spacing)
        print(new_spacing_refine)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    newimage = resample.Execute(sitk_image)
    return newimage

def set_window_wl_ww(tensor):
    sl_window = [75,400]
    [wl,ww] = sl_window
    w_min, w_max = wl - ww//2, wl + ww//2
    tensor[tensor < w_min] = w_min
    tensor[tensor > w_max] = w_max
    tensor = (tensor - w_min) / (w_max - w_min)
    ### min max Normalization
    return tensor

def crop_roi(image, label):
    assert(image.shape == label.shape)
    print (image.shape)
    ### crop based on lung segmentation
    w, h, d = label.shape

    tempL = np.nonzero(label)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])

    px = max(output_size[0] - (maxx - minx), 0) // 2
    py = max(output_size[1] - (maxy - miny), 0) // 2
    minx = max(minx - px - 25, 0) #np.random.randint(5, 10)
    maxx = min(maxx + px + 25, w) #np.random.randint(5, 10)
    miny = max(miny - py - 25, 0)
    maxy = min(maxy + py + 25, h)
    
    image = image[minx:maxx, miny:maxy,:].astype(np.float32)
    label = label[minx:maxx, miny:maxy,:].astype(np.float32)
    return image, label

listt = glob('./Pancreas-CT/data/*')

for item in tqdm(listt):
    name_image = str(item)
    name_label = name_image.replace('/data/PANCREAS_', '/TCIA_pancreas_labels-02-05-2017/label')
#     pdb.set_trace
    itk_img = sitk.ReadImage(name_image)
#     origin =itk_img.GetOrigin()
#     direction = itk_img.GetDirection()
#     space = itk_img.GetSpacing()
    itk_img = ImageResample(itk_img)
    image = sitk.GetArrayFromImage(itk_img)
    image = np.transpose(image, (2,1,0))
    
    itk_label = sitk.ReadImage(name_label)
    itk_label = ImageResample(itk_label, is_label = True)
    label = sitk.GetArrayFromImage(itk_label)
    label = np.transpose(label, (2,1,0))

    assert(np.max(label) == 1 and np.min(label) == 0)
    assert(np.unique(label).shape[0] == 2)
    assert(np.shape(label)==np.shape(image))
    image = set_window_wl_ww(image)
#     print(image.shape)
#     plt.figure(figsize=(10, 10))
#     plt.title('CT Slice_enhanced_100')
#     plt.imshow(image[:,:,100],cmap='gray')
#     plt.show()
    image, label = crop_roi(image, label)
    image = (image - np.mean(image)) / np.std(image)
    print(image.shape)
    f = h5py.File(('./Pancreas_h5/image'+name_image[28:32] + '_norm.h5'), 'w')
    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()