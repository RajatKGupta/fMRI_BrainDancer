# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: June 29, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
This function maps T2* decay across all voxels of an input 4D multi-echo gre T2*-weighted dataset.

Inputs:
1. Input 4D dataset in nib dataformat.
2. Echo times as a list. 
Outputs:
1. Three dimensional T2* map. 

"""
# %% All imports
import numpy as np

# %%
# =============================================================================
# T2 Relaxometry
# =============================================================================
def T2Star(data4D,tes=[]):
    data = data4D
    if data.shape[3] != len(tes):
        print("Incorrect number of echo times entered. Expected number of TEs =",data.shape[3])
        return 
    else:
        tes = np.array(tes)
        numrows = data.shape[0]
        numcols = data.shape[1]
        numslices = data.shape[2]
        
        maps = []
        T2data = abs(data.get_data())
        ydata = T2data.reshape(numrows*numcols*numslices,data.shape[3])
        
        for i in range(numrows*numcols*numslices):
            
            fit = np.polyfit(tes,np.log(ydata[i]),1)
            maps.append(-1/fit[0])

    maps = np.array(maps).reshape((numrows,numcols,numslices))
        
    return maps            

# %%