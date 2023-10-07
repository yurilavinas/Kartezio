#!/usr/bin/env python
# coding: utf-8

# Importing Required Modules
import openslide
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import math
from tqdm.notebook import tqdm

from numena.image.color import rgb2bgr

def tile_saving(patient, img_tiles, mask_tiles, folder, thres):
    for i, img_crop in tqdm(enumerate(img_tiles)):
    
        uniques = np.unique(img_crop)
        if np.sum(uniques) == 0:
            continue
        elif np.sum(uniques) == 255:
            continue
        else:
            img_c = 255-img_crop # image complement
            if img_c.mean() < thres: # pixel value mean greater threshold -> select tile 
                #make mask of where the transparent bits are
                trans_mask = img_crop[:,:,3] == 0
                #replace areas of transparency with white and not transparent
                img_crop[trans_mask] = [255, 255, 255, 255]
                
                #make mask of where the black bits are
                black_pixels = np.where(
                    (img_crop[:, :, 0] == 0) & 
                    (img_crop[:, :, 1] == 0) & 
                    (img_crop[:, :, 2] == 0)
                )

                # set those pixels to white
                img_crop[black_pixels] = [255, 255, 255, 255]

                #new image without alpha channel...
                new_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2BGRA)
                # new_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
                cv2.imwrite(os.path.join(folder,f'{patient}_{i}.png'), new_img)
                cv2.imwrite(os.path.join(folder, f'{patient}_{i}_mask.png'), mask_tiles[i])

    
# adapted from: https://www.kaggle.com/code/analokamus/a-fast-tile-generation
# from: https://www.kaggle.com/code/nghihuynh/wsi-preprocessing-tiling-tissue-segmentation
def make_tiles(img, tile_size=256):
# def make_tiles(img, mask, tile_size=256):
    '''
    img: np.ndarray with dtype np.uint8 and shape (width, height, channel)
    mask: np.ndarray with dtype np.uint9 and shape (width, height)
    '''
    if len(img.shape) == 3:
        w_i, h_i, ch = img.shape
    else:
        w_i, h_i = img.shape

    pad0, pad1 = (tile_size - w_i%tile_size) % tile_size, (tile_size - h_i%tile_size) % tile_size
   
    if len(img.shape) == 3:
        padding_i = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
        # img = np.pad(img, padding_i, mode='constant', constant_values=0)
        img = np.pad(img, padding_i)
        img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size, ch)
        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, ch)
    else:
        padding_m = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2]]
        # img = np.pad(img, padding_m, mode='constant', constant_values=0)
        img = np.pad(img, padding_m)
        img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size)
        img = img.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)
    return img


level = 8
img_folder = "../tiles"
thres_otsu = 190#190
size = 1024
top = 100
perc = 0.8


# patients = ["patient_004_node_4", "patient_010_node_4", "patient_017_node_4", "patient_020_node_4", "patient_022_node_4", "patient_099_node_4"]
# patients = ["patient_017_node_4"]
cellpose = ["000_img.png"]
cellpose_mask = ["000_masks.png"]
for i, cp in enumerate(cellpose):

    region_image = cv2.imread("/home/yuri/Documents/postdoc/CGP/datasets/cellpose/train/{cp}")
    cv2.imshow('image',region_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop image 
    region_image = region_image[top:int(region_image.shape[0]*perc), :] 
    region_mask = region_mask[top:int(region_mask.shape[0]*perc), :] 

    img_tiles = make_tiles(region_image, tile_size=size)
    mask_tiles = make_tiles(region_mask, tile_size=size)

    # tile_saving(patient, img_tiles, mask_tiles, img_folder, thres_otsu)

