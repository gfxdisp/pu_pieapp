import torch as pt
import numpy as np
import scipy.stats as st
import torch.utils.data as data
import math
from imageio import imread
from functools import reduce

import pandas as pd
import torch.nn as nn
import torch.utils.data as data
import os.path as path
import os
import h5py
import re
from glob import glob

def identity(x):
    return x

def image2patchesG(reference,image, num_patches=121, patch_size=64, number_of_gpus = 1):
    """
    reference: a CxHxW tensor-like
    image: a CxHxW tensor-like
    num_patches: number of patches to generate,
                 should be a tuple (rows, cols) or a number (rows=cols)
    patch_size: the size of the patches generated

    this function split both reference and A images into
    patches of patch_size x patch_size. The process is random
    """
    

    total_number_of_patches = num_patches

    _, H, W = reference.shape
    rows_factor = H/64
    
    cols_factor = W/64

    rows = int(np.floor(rows_factor*np.sqrt(total_number_of_patches/(rows_factor*cols_factor))))
    cols = int(np.floor(cols_factor*np.sqrt(total_number_of_patches/(rows_factor*cols_factor))))


    #if rows*cols>144*number_of_gpus:
    #    factor = np.sqrt(144*number_of_gpus/rows/cols)
    #    rows = int(np.floor(factor*rows))
    #    cols = int(np.floor(factor*cols))


    r, c = np.meshgrid(
        np.linspace(0, 1-1/(rows+1), (rows+1)),
        np.linspace(0, 1-1/(cols+1), (cols+1))
    )
    
    r += np.random.rand(*r.shape) * (1/(rows+1))
    c += np.random.rand(*c.shape) * (1/(cols+1))
    r, c = r.ravel(), c.ravel()
    
    reference_patches = []
    image_patches = []
    for i in range(total_number_of_patches):
        h = int(r[i] * (H-patch_size-1))
        w = int(c[i] * (W-patch_size-1))
        reference_patches.append(pt.tensor(
            reference[:, h:h+patch_size, w:w+patch_size]))
        image_patches.append(pt.tensor(
            image[:, h:h+patch_size, w:w+patch_size]))

    return pt.stack(reference_patches), pt.stack(image_patches)


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
        PU_table = pd.read_csv(
                path.join( './loader/pu_space.csv'),
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

def image2patches_det(reference, image,  patch_size=64, shift_patch = 64):


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


def patches2image_det(patches, weights, scores, score_weights, patch_size=64, shift_patch = 64):

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


def image2patches(reference, A, num_patches=150, patch_size=64):
    """
    reference: a CxHxW tensor-like
    image: a CxHxW tensor-like
    num_patches: number of patches to generate, if set to None,
                 generate all patches
    patch_size: the size of the patches generated

    this function split both reference and A images into
    patches of patch_size x patch_size. The process is random
    """
    _, H, W = reference.shape
    total_batches = (H-patch_size) * (W-patch_size)
    num_patches = num_patches or total_batches
    reference_patches = []
    A_patches = []
    idxs = pt.randperm(total_batches)
    for i in idxs[:num_patches]:
        c = i.item() // (W-patch_size)
        r = i.item() % (W-patch_size)
        reference_patches.append(pt.tensor(
            reference[:, c:c+patch_size, r:r+patch_size]))
        A_patches.append(pt.tensor(
            A[:, c:c+patch_size, r:r+patch_size]))
    return pt.stack(reference_patches), pt.stack(A_patches)

class ImageLoader():
    def __init__(self):
        self._weight_map = None
        pass

    def read_image_from_file_and_split(self,ref_path,dist_path,dynamic_range, lum_top=100, lum_bottom=0.5,shift_patch= 64):
        ref_image, dist_image =load_ref_dis_images(path_ref=ref_path, path_dist=dist_path, dynamic_range=dynamic_range, lum_top=lum_top, lum_bottom=lum_bottom)
        ref_p, dist_p, weight_map = image2patches_det(reference=ref_image,image=dist_image,  patch_size=64, shift_patch = shift_patch)
        self._weight_map = weight_map
        return ref_p, dist_p, weight_map

    def create_error_map(self,patches, weights, scores, score_weights, patch_size=64, shift_patch = 64 ):
        img, err_map = patches2image_det(patches, weights, scores, score_weights, patch_size, shift_patch)

        return img, err_map


class PatchesDataset(data.Dataset):

    def __init__(self, h5file):
        """
        the dataset should be extracted from the h5file
        The h5file has the following structure
        scores:
            keys: names
            values: score
        images:
            [name]: CxHxW image
        """
        
        self.number_of_gpus = 1
        self.names = list(h5file['scores/keys'])
        self.scores = np.array(h5file['scores/values'])
        self.images = h5file['images']
        self.num_patches = 121


    def __len__(self):
        return len(self.names)


    def set_number_of_gpus(self,number_of_gpus):
        self.number_of_gpus = number_of_gpus
        self.num_patches =  self.num_patches*self.number_of_gpus


    def __getitem__(self, idx):
        name = self.names[idx]
        score = self.scores[idx]
        img = self.images[name]
        ref = self.images[img.attrs['reference']]
        ref_patches, img_patches = image2patchesG(reference = ref, image = img, num_patches = self.num_patches)
        return idx, score, img_patches, ref_patches

class PairPatchesDataset(data.Dataset):

    def __init__(self, h5file):
        self.number_of_gpus = 1
        self.dataset = PatchesDataset(h5file)

    def set_number_of_gpus(self,number_of_gpus):
        self.number_of_gpus = number_of_gpus
        self.dataset.set_number_of_gpus(number_of_gpus)

    def __len__(self):
        n = len(self.dataset)
        return n * (n-1)//2

    def __getitem__(self, idx):
        n = len(self.dataset)
        idx1 = idx // (n-1)
        idx2 = idx % (n-1)
        if idx2 >= idx1:
            idx2 += 1
        
        # If conditions are too far, we don't want to compare them, even to 
        # retreat them from the dataset.
        if abs(self.dataset.scores[idx1]- self.dataset.scores[idx2])<0.8 or abs(self.dataset.scores[idx1]- self.dataset.scores[idx2])>1.1:
            return -1,-1, -1, -1, -1

        _, score1, img1, ref1 = self.dataset[idx1]
        _, score2, img2, ref2 = self.dataset[idx2]

        # convert to the probability of being better from the difference in 
        # scores
        score = st.norm(0,1.4826).cdf(score1-score2)

        return score, img1, ref1, img2, ref2

