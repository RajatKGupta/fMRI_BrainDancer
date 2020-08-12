## BrainDancer Analysis Package v1.0

![BrainDancer](https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/cover.png)

This repository contains code for quantifying and removing scanner-induced noise from fMRI data using the BrainDancer Dynamic Phantom. Details about the dynamic phantom and analysis algorithms are provided in the accompanying manuscript: https://arxiv.org/abs/2004.06760.


Author: Rajat Kumar, rajat.kumar@stonybrook.edu; rajatkgupta@protonmail.ch


#### Installation:
1) Download/Clone the repository.
2) Install Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/. 
3) Install the Environment: conda env create  -f environment.yml -n BrainDancer

#### To run, follow the steps below (ipynb files needs to be opened in Jupyter Notebook ):
1) Before any data-analysis, the phantom data should be corrected for spatial intensity non-uniformity (INU). We recommend using the N4ITK algorithm implemented in ANTs Toolbox. Download and installation is available from: http://stnava.github.io/ANTs/. <br/>
*Recommended settings for N4ITK:* --bspline-fitting [300] -d 4 --convergence [150x150x150, 1e-06]. 

2) Run extract_ts_phantom.ipynb: This generates the ground-truth data and extracts the voxel time-series from fMRI output. Needs phantom-log file obtained from your BrainDancer device, slice-acquistion order of your acquistion protocol in csv format, and INU-corrected fMRI measurement in nifti format.<br/> <br/>
*Visual inspection and decision making when running extract_ts_phantom.ipynb*
![BrainDancer](https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/slices.png)

*After slice selection, if you observe no contours for a slice - the default threshold being used is too high for your dataset. The program will prompt you to enter a new threshold.*
![BrainDancer](https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/thresholding.png)

3) Run ML.ipynb: This implements training of CNN for learning noise characteristics. It requires three nifti files that were generated in step 1 output (scanner data, ground-truth data and masks). 

4) Run Phantom_Analysis: This implements data cleaning using trained CNN. It requires three nifti files that were generated in step 1 and the .h5 trained weights file from step 2. 
