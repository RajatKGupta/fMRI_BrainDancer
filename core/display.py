# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: September 3, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Functions to help with display.  

"""

# %% All imports
import nibabel as nib
import matplotlib.pyplot as plt

# %%
# =============================================================================
# Display EPI slices
# =============================================================================
def display_all_slices(data,volume_number):
    data_read = data
    count = 1
    
    fig = plt.figure(figsize=(15,data_read.header['dim'][3]+10))
    
    for i in range(data_read.header['dim'][3]):
        fig.subplots_adjust(hspace=0, wspace=0.0005)      
        ax = fig.add_subplot(data_read.header['dim'][3]/4 +1 ,4, count)
        im= plt.imshow(data_read.get_data()[:,:,i,volume_number])
        ax.set_title('Slice '+str(i),fontsize=18)
        ax.set_axis_off()
        count+=1
# %%
# =============================================================================
# Display T2* maps
# =============================================================================
def display_T2starmaps(image,vmax=55,vmin=40,colormap='seismic'):
    
        plt.imshow(np.nan_to_num(image),cmap=colormap,vmax=vmax, vmin=vmin)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        
# %%