import torch as pt
import torch.nn as nn
import torch.utils.data as data

import numpy as np
import scipy.stats as st
import math
from imageio import imread
import pandas as pd

from glob import glob

def identity(x):
    return x


# two values are cache here to save computational time
_LOGL = None
_P = None



def PU_table():
    """
    return lookup table that transform log10(Luminance) to perceptually uniform code values
    """
    global _LOGL
    global _P
    if _LOGL is None or _P is None:
        PU_table = pd.read_csv('./loader/pu_space.csv',
                header=None, names=['L', 'P'])
        logL = np.log10(np.array(PU_table['L']))
        P = np.array(PU_table['P'])
        _LOGL = logL
        _P = P
    return _LOGL, _P

def scale(x, min_value, max_value):
    """
    scale x to values between 0 and 1
    """
    domain = max_value - min_value
    x = (x - min_value) / domain
    return x

def pu(x):
    logL, P = PU_table()
    epsilon = 1e-8
    logx = np.clip(np.log10(np.clip(x, epsilon, None)), logL.min(), logL.max())
    pux = np.interp(logx, logL, P)
    if np.any(np.isnan(pux)):
        import ipdb; ipdb.set_trace()
    return scale(pux, P.min(), P.max())


def image2tensor(filepath, transform=None):
    """
    read image
    display model + perceptural transformation
    permute the tensor to CxHxW
    """
    image = imread(filepath)
    transform = transform or identity
    image = pu(transform(image))
    image = np.transpose(image.astype(np.float32), (2, 0, 1))#image.astype(np.float32)#
    return image


def load_ref_dis_images(path_ref, path_dist,dynamic_range, lum_top=100, lum_bottom=0.5):
    ref_image = load_image(path_ref,dynamic_range, lum_top, lum_bottom)
    dist_image = load_image(path_dist,dynamic_range, lum_top, lum_bottom)
    
    return ref_image, dist_image
    


def load_image(path_im,dynamic_range, lum_top=100, lum_bottom=0.5):
    
    transforms={'hdr': lambda x: x,
                'ldr': lambda x: (lum_top - lum_bottom) * ((x/255)**2.2) + lum_bottom}

    image = image2tensor(path_im, transforms[dynamic_range])  
    
    return image



def image2patches(reference, image,  patch_size=64, shift_patch = 64):
    '''
    Split image into patches. The patches are extracted in a grid like manner.
    '''

    _,H, W = reference.shape
    
    weight_map  = np.zeros(shape=(H,W))
    
    rows_factor = (H-(patch_size-shift_patch))/shift_patch
    cols_factor = (W-(patch_size-shift_patch))/shift_patch

    rows =  int(np.floor(rows_factor))
    cols =  int(np.floor(cols_factor))

    weight_factor = np.ones(shape=(patch_size,patch_size))
    reference_patches = []
    image_patches = []
    h = 0
    for jj in range(0,cols):
        for ii in range(0,rows):
            h = ii*shift_patch
            w = jj*shift_patch
            reference_patches.append(pt.tensor(
                reference[:,h:h+patch_size, w:w+patch_size]))
            image_patches.append(pt.tensor(
                image[:,h:h+patch_size, w:w+patch_size]))
            weight_map[h:h+patch_size, w:w+patch_size] += weight_factor
        
    if H%shift_patch!=0:
        for jj in range(0,cols):
            w = jj*shift_patch
            reference_patches.append(pt.tensor(
                reference[:,(H-patch_size):H, w:w+patch_size]))
            image_patches.append(pt.tensor(
                image[:,(H-patch_size):H, w:w+patch_size]))
            
            weight_map[(H-patch_size):H, w:w+patch_size] += weight_factor
            
    if W%shift_patch!=0:
        for ii in range(0,rows):
            h = ii*shift_patch
            reference_patches.append(pt.tensor(
                reference[:,h:h+patch_size, (W-patch_size):W]))
            image_patches.append(pt.tensor(
                image[:,h:h+patch_size, (W-patch_size):W]))
            
            weight_map[h:h+patch_size, (W-patch_size):W] += weight_factor
            
    if W%shift_patch!=0 and H%shift_patch!=0:
        reference_patches.append(pt.tensor(
                reference[:,(H-patch_size):H, (W-patch_size):W]))
        image_patches.append(pt.tensor(
                image[:,(H-patch_size):H, W-patch_size:W]))
        weight_map[(H-patch_size):H, W-patch_size:W] += weight_factor
    weight_map = np.divide(1,weight_map)         

    return pt.stack(reference_patches), pt.stack(image_patches), weight_map


def patches2image(patches, weights, scores, score_weights, patch_size=64, shift_patch = 64):
    '''
    Function merge patches into an image. 
    '''

    (H, W) = np.shape(weights)
    output_image = np.zeros(shape=(3, H, W))

    output_map = np.zeros(shape=(3, H, W))
    
    rows_factor = (H-(patch_size-shift_patch))/shift_patch
    cols_factor = (W-(patch_size-shift_patch))/shift_patch

    rows =  int(np.floor(rows_factor))
    cols =  int(np.floor(cols_factor))
    ones_patch =np.ones(shape=(3,patch_size,patch_size))
    h = 0
    for jj in range(0,cols):
        for ii in range(0,rows):
            h = ii*shift_patch
            w = jj*shift_patch
            output_image[:,h:h+patch_size,w:w+patch_size] += np.multiply(
                patches[rows*jj+ii,:,:,:],weights[h:h+patch_size,w:w+patch_size])

            output_map[:,h:h+patch_size,w:w+patch_size] += np.multiply(ones_patch*scores[rows*jj+ii]*
                score_weights[rows*jj+ii],weights[h:h+patch_size,w:w+patch_size])
        
    if H%shift_patch!=0:
        for jj in range(0,cols):
            w = jj*shift_patch
            output_image[:,(H-patch_size):H, w:w+patch_size] += np.multiply(
                patches[rows*cols+jj,:,:,:],weights[(H-patch_size):H, w:w+patch_size])

            output_map[:,(H-patch_size):H, w:w+patch_size] += np.multiply(ones_patch*scores[rows*cols+jj]*
                score_weights[rows*cols+jj],weights[(H-patch_size):H, w:w+patch_size])


    if W%shift_patch!=0:
        for ii in range(0,rows):
            h = ii*shift_patch
            output_image[:,h:h+patch_size, (W-patch_size):W] += np.multiply(
                patches[(rows+1)*cols+ii,:,:,:],weights[h:h+patch_size, (W-patch_size):W])

            output_map[:,h:h+patch_size, (W-patch_size):W] += np.multiply(ones_patch*scores[(rows+1)*cols+ii]*
                score_weights[(rows+1)*cols+ii],weights[h:h+patch_size, (W-patch_size):W])
            
    if W%shift_patch!=0 and H%shift_patch!=0:
        
        output_image[:,(H-patch_size):H, (W-patch_size):W] += np.multiply(
                patches[(rows+1)*cols+rows,:,:,:],weights[(H-patch_size):H, (W-patch_size):W])

        output_map[:,(H-patch_size):H, (W-patch_size):W] += np.multiply(ones_patch*scores[(rows+1)*cols+rows]*
                score_weights[(rows+1)*cols+rows],weights[(H-patch_size):H, (W-patch_size):W])
                    
    return output_image, output_map



class ImageLoader():
    def __init__(self):
        self._weight_map = None
        pass

    def read_image_from_file_and_split(self,ref_path,dist_path,dynamic_range, lum_top=100, lum_bottom=0.5,shift_patch= 64):
        ref_image, dist_image =load_ref_dis_images(path_ref=ref_path, path_dist=dist_path, dynamic_range=dynamic_range, lum_top=lum_top, lum_bottom=lum_bottom)
        ref_p, dist_p, weight_map = image2patches(reference=ref_image,image=dist_image,  patch_size=64, shift_patch = shift_patch)
        self._weight_map = weight_map
        return ref_p, dist_p, weight_map

    def create_error_map(self,patches, weights, scores, score_weights, patch_size=64, shift_patch = 64 ):
        img, err_map = patches2image(patches, weights, scores, score_weights, patch_size, shift_patch)

        return img, err_map
