# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: October 21, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Script for executing the analysis routine for denoising.

To do:
Include the option to generate outer mask and corresponding time_series.

"""

# %% All imports
import nibabel as nib, pandas as pd, numpy as np, matplotlib.pyplot as plt
from display import display_all_slices
from cartridge import findcartridge,inner_mask, outer_mask, cen_rotation
from quadrants import quadrant_mask_T2
from t2 import T2Star
from epi import phantom_motion, create_mean_slices, simulate_inner, scanner_output


# %%
# =============================================================================
# Main Routine
# =============================================================================


def create_denoising_dataset(epi_path,log_path,acqtimes_path,rot_dir=-1):
    data_read = nib.load(epi_path)
    
    
    display_all_slices(data_read,0)
    plt.show()
    
    start = int(input('Enter the first good slice: '))
    end = int(input('Enter the last good slice: '))
    
    log = pd.read_csv(log_path)
    motion_time = np.max(log['Tmot'].values)
    acq_times = pd.read_csv(acqtimes_path)
    motionfree = acq_times[acq_times['Time']>motion_time]['Slice'].values
    total_slices = []    
    for i in list(motionfree):
        if start<= i <= end:
            total_slices.append(i)
    print('Selected Slices for Analysis are: ', total_slices)
    
    imask = []
    cen = []
    imask_metrics = []
    center_rotation_all = []
    omask = []
    detect_remove = []
    updated_total_slices = []
    for i in range(len(total_slices)): 
               
       
        img_complete,cy_complete,cx_complete, radii_complete = inner_mask(epi_path,total_slices[i],volume_num=0,lvl=0.004,rad1=7,rad2=50,step=1)
        
        
        center_rotation  = cen_rotation(epi_path,total_slices[i],img_complete,cy_complete,cx_complete,radii_complete, canny_sgm=1)
       
        
        detect = int(input('Enter 1 if this slice is good'))
        
        if detect ==1:
            
            center_rotation_all.append(center_rotation)
            imask.append(img_complete)
            updated_total_slices.append(total_slices[i])
        # TO DO - Include the option to generate outer mask and corresponding time_series, with something like below:
        #out_mask = outer_mask(data_read,findcartridge(data_read,total_slices[i],0),total_slices[i],0)
        #omask.append(out_mask)
    
        
    positions = phantom_motion(log_path)
    synth =create_mean_slices(data_read,updated_total_slices,imask,200)
    simulated_data = simulate_inner(synth,positions,updated_total_slices,imask,center_rotation_all,rot_dir)
    scanner_inner = scanner_output(data_read,positions,updated_total_slices,imask,200) # add omask in future for outer cylinder
    
    return simulated_data, scanner_inner, imask, center_rotation_all, updated_total_slices


# %%
