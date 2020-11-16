# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: April 10, 2020

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Boilerplate routines.  

"""

# %% All imports
import nibabel as nib, numpy as np
import keras.backend as K
from skimage.draw import circle
from skimage.transform import hough_circle, hough_circle_peaks


# %%
# =============================================================================
# List to Nibabel 
# =============================================================================
def list_to_nib(mask_list,target_affine):
    imask_nib = mask_list[0]
    for i in range(1,len(mask_list)):
        imask_nib = np.dstack((imask_nib,mask_list[i]))
    return nib.Nifti1Image(imask_nib,affine=target_affine)

# %%
# =============================================================================
# Custom Loss for the denoising - Always add all custom loss functions here. 
# =============================================================================

def custom_loss(y_true,y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        loss2 =  ( 1 - SS_res/(SS_tot + K.epsilon()) )
        return -loss2 
    
# %%
# =============================================================================
# Get Rid of the Notch!! 
# =============================================================================    
def imask_ut(imask):
    imask_utils = []
    for i in range(imask.shape[2]):
        img = imask.get_data()[:,:,i]
        hough_radii = np.arange(2, 16, 1)
        hough_res = hough_circle(img, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)
        img2 = np.zeros(img.shape)
        rr,cc = circle(cy[0],cx[0],radii[0])
        img2[rr,cc]=1
    
        imask_utils.append(img2)  
    return(imask_utils)

# %%
# =============================================================================
# Removes Outlier Voxels - Edge Effect during Ground Truth Creation 
# =============================================================================
def outlier(stack_sim_1,sd):
    temp_s = []
    for i in range(len(stack_sim_1)):
        temp_s.append(np.std(stack_sim_1[i]))
    temp_s = np.array(temp_s)
    def reject_outliers(data, m=sd):
        index = []
        for i in range(len(data)):
            if abs(data[i] - np.mean(data)) > (m * np.std(data)):
                index.append(i)
        return index
    index = reject_outliers(temp_s)
    return index

# %%