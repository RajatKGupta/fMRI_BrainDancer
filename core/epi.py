# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: January 9, 2020

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Scripts related to EPI preprocessing. 

"""


# %% All imports
import pandas as pd, nibabel as nib, numpy as np
import matplotlib.pyplot as plt
from skimage.transform import (rotate as rt,rescale,downscale_local_mean)
from statsmodels.tsa.tsatools import detrend

# %%
# =============================================================================
# Converting encoder readings to degrees
# =============================================================================
def phantom_motion(log_path,first_motion_slice=200):
    log = pd.read_csv(log_path)
    final_pos = log['EndPos'].values 
    positions = final_pos
    positions = positions.flatten() * 0.04392
    for index in range(len(positions)):
        if positions[index] >90:
            positions[index] = positions[index] -360
    plt.figure(figsize=(25,5))
    positions = positions[first_motion_slice:]-positions[0]
    plt.plot(positions[:])
    plt.xlabel('TRs',fontsize=20)
    plt.ylabel('Phantom Motion in Degrees',fontsize=20)
    
    return positions

# %%
# =============================================================================
# Create mean slices for generating ground truth
# =============================================================================
def create_mean_slices(data_read,best_slices,imask,first_motion_slice,ignore=10):
    
    mean_data_best_slices = None
    for i in range(len(best_slices)):
        temp_mean = np.mean(data_read.get_data()[:,:,best_slices[i],ignore:first_motion_slice],axis=2)
        temp_mean = temp_mean * imask[i]
    
        if mean_data_best_slices is None:
            mean_data_best_slices = temp_mean
        else:
            mean_data_best_slices = np.dstack((mean_data_best_slices,temp_mean))
    mean_data_best_slices = nib.Nifti1Image(mean_data_best_slices,data_read.affine)
    
    return mean_data_best_slices


# %%
# =============================================================================
# Create ground truth time series. 
# =============================================================================
def simulate_inner(mean_data_best_slices,positions,best_slices,imask,cen_r,rot_dir=-1):
    dataset = []
   
    for move in range(len(positions)):
        slices = None
        for i in range(len(best_slices)):
            # move the inner cylinder
            inner = rescale(mean_data_best_slices.get_data()[:,:,i],5,order=3,preserve_range=True, mode='constant')
            inner = rt(inner, angle = rot_dir*positions[move],center=(np.array((cen_r[i][1],cen_r[i][0]))*5), order=3, preserve_range=True)
            inner = downscale_local_mean(inner, (5,5))
            inner = inner * imask[i]
            if slices is None:
                slices = inner
            else:
                slices = np.dstack((slices,inner))
        
        dataset.append([slices])
        
    initial = dataset[0][0].reshape(mean_data_best_slices.header['dim'][1],mean_data_best_slices.header['dim'][2],mean_data_best_slices.header['dim'][3],1)
    
    for i in range(1, len(positions)):
        initial = np.concatenate((initial,dataset[i][0].reshape(mean_data_best_slices.header['dim'][1],mean_data_best_slices.header['dim'][2],mean_data_best_slices.header['dim'][3],1)), axis=3)
    
    dataset_final = nib.Nifti1Image(initial,mean_data_best_slices.affine)

    return dataset_final

# %%
# =============================================================================
# Create scanner output - measured fMRI time series extraction. 
# Disabled the outer cylinder Extraction. Preserved for future use. 
# =============================================================================


def scanner_output(data_read,positions,best_slices,imask,first_motion_slice): # add omask in future for outer cylinder

    scanner_data = []
    #scanner_data_outer = []
    for move in range(len(positions)):
        slices = None
        #slices_outer = None
        for i in range(len(best_slices)):
            # Extract the inner cylinder
            inner = data_read.get_data()[:,:,best_slices[i],move+first_motion_slice]
            inner = inner * imask[i]
        
            #outer = data_read.get_data()[:,:,best_slices[i],move+first_motion_slice]
            #outer = outer * omask[i]
        
            if slices is None:
                slices = inner
            else:
                slices = np.dstack((slices,inner))
            
            #if slices_outer is None:
               # slices_outer = outer
            #else:
               # slices_outer = np.dstack((slices_outer,outer))
        
        scanner_data.append([slices])
        #scanner_data_outer.append([slices_outer])
        
    initial = scanner_data[0][0].reshape(data_read.header['dim'][1],data_read.header['dim'][1],len(best_slices),1)
    for i in range(1, len(positions)):
        initial = np.concatenate((initial,scanner_data[i][0].reshape(data_read.header['dim'][1],data_read.header['dim'][1],len(best_slices),1)), axis=3)
    
    #initial_outer = scanner_data_outer[0][0].reshape(data_read.header['dim'][1],data_read.header['dim'][1],len(best_slices),1)
    #for i in range(1, len(positions)):
        #initial_outer = np.concatenate((initial_outer,scanner_data_outer[i][0].reshape(data_read.header['dim'][1],data_read.header['dim'][1],len(best_slices),1)), axis=3)  
     
    scanner_dataset_final = nib.Nifti1Image(initial,data_read.affine)
    #scanner_dataset_final_outer = nib.Nifti1Image(initial_outer,data_read.affine)
    
    return scanner_dataset_final#, scanner_dataset_final_outer

# %%
# =============================================================================
# Data preparation for CNN denoising.  
# =============================================================================


def data_prep_ml(data_read_simulation,data_read_scanner,imask,nwindows,lwindows):
    stack_sim = None
    stack_scn = None
    noise = None

    stack_sim_flip= None
    stack_scn_flip = None
    noise_flip = None
    
    
    for i in range(len(imask)):
        simulation = data_read_simulation.get_data()[np.nonzero(imask[i])[0],np.nonzero(imask[i])[1],i,:]
        scanner = data_read_scanner.get_data()[np.nonzero(imask[i])[0],np.nonzero(imask[i])[1],i,:]
        
        for j in range(np.count_nonzero(imask[i])):
        
            sim = simulation[j,:]
            sim = sim - np.mean(sim)
        
            scn_orig = scanner[j,:]
            scn = scanner[j,:]
            scn = detrend(scn,2) 
            
        
            noise_temp= scn- sim
        
        
            sim_flip = np.flip(sim,axis=-1)
            scn_flip = np.flip(scn,axis=-1)
            noise_temp_flip = np.flip(noise_temp,axis=-1)
        
        
            if stack_sim is None:
                stack_sim = sim.reshape((nwindows,lwindows))
                stack_scn= scn.reshape((nwindows,lwindows))
                noise = noise_temp.reshape((nwindows,lwindows))
            
                stack_sim_flip = sim_flip.reshape((nwindows,lwindows))
                stack_scn_flip= scn_flip.reshape((nwindows,lwindows))
                noise_flip = noise_temp_flip.reshape((nwindows,lwindows))
            else:
                stack_sim = np.vstack((stack_sim,sim.reshape((nwindows,lwindows))))
                stack_scn = np.vstack((stack_scn,scn.reshape((nwindows,lwindows))))
                noise = np.vstack((noise,noise_temp.reshape((nwindows,lwindows))))
            
                stack_sim_flip = np.vstack((stack_sim_flip,sim_flip.reshape((nwindows,lwindows))))
                stack_scn_flip = np.vstack((stack_scn_flip,scn_flip.reshape((nwindows,lwindows))))
                noise_flip = np.vstack((noise_flip,noise_temp_flip.reshape((nwindows,lwindows))))

    return stack_scn, stack_sim, noise, stack_scn_flip, stack_sim_flip, noise_flip

# %%
# ===============================================================================================
# Data preparation for data quality metrics calculation; Use Obsolete; Preserved for future use.
# ===============================================================================================


def data_prep_metrics(data_read_simulation,data_read_scanner,imask):
    
    stack_sim = None
    stack_scn = None
    noise = None   
    
    for i in range(len(imask)):
        simulation = data_read_simulation.get_data()[np.nonzero(imask[i])[0],np.nonzero(imask[i])[1],i,:]
        scanner = data_read_scanner.get_data()[np.nonzero(imask[i])[0],np.nonzero(imask[i])[1],i,:]
        
        for j in range(np.count_nonzero(imask[i])):
        
            sim = simulation[j,:]      
            scn = scanner[j,:] 
            noise_temp= scn- sim                    
        
            if stack_sim is None:
                stack_sim = sim
                stack_scn= scn
                noise = noise_temp               
            else:
                stack_sim = np.vstack((stack_sim,sim))
                stack_scn = np.vstack((stack_scn,scn))
                noise = np.vstack((noise,noise_temp))                
          
    return stack_scn, stack_sim, noise

# %%
