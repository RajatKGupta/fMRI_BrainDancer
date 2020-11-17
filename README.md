# BrainDancer Analysis Package

![BrainDancer](https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/cover.png)

This repository contains code for quantifying and removing scanner-induced noise from fMRI data using the BrainDancer Dynamic Phantom. Details about the dynamic phantom and analysis algorithms are provided in the accompanying manuscript: https://arxiv.org/abs/2004.06760.


Author: Rajat Kumar, rajat.kumar@stonybrook.edu; rajatkgupta@protonmail.ch


## Installation:
1) Download/Clone the repository.
2) Install Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/. 
3) Install the Environment: conda env create  -f environment.yml -n BrainDancer


Before any data-analysis, the phantom data should be corrected for spatial intensity non-uniformity (INU). We recommend using the N4ITK algorithm implemented in ANTs Toolbox. Download and installation is available from: http://stnava.github.io/ANTs/.
*B-Spline fitting and convergence parameter should be chosen carefully and is highly dependent on main MR field strength and acquisition protocol used.* 
<p align="center">
  <img src="https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/Biasfield.png">
</p>
 

## To denoise fMRI data from scanner-induced variance, follow the steps below (Jupyter Notebooks are preferred for the analysis):
1) Run Example_TimeSeriesExtraction.ipynb: This file provides a skeleton code for generating the ground-truth data and extracts the voxel time-series from fMRI output. Needs phantom-log file obtained from your BrainDancer device, slice-acquistion order of your acquistion protocol in csv format, and INU-corrected fMRI measurement in nifti format.<br/> <br/>
*Visual inspection and decision making when running Example_TimeSeriesExtraction.ipynb*
![BrainDancer](https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/slices.png)

*After slice selection, if you observe no contours for a slice - the default threshold being used is incorrect for your dataset. The program will prompt you to enter a new threshold.*
![BrainDancer](https://github.com/RajatKGupta/fMRI_BrainDancer/blob/master/assets/thresholding.png)

2) Run Example_TrainingCNN.ipynb: This implements training of CNN for learning noise characteristics. It requires three nifti files that were generated in step 1 output (measured fMRI data, ground-truth data and masks). 

3) Run Example_Denoising.ipynb: This implements data denoising using the trained CNN. It requires data to be denoised (phantom or human) as numpy array and the .h5 trained weights file from step 2. 


## For caluclating data-quality metrics, follow the steps below:
1) Run Example_TimeSeriesExtraction.ipynb: This file provides a skeleton code for generating the ground-truth data and extracts the voxel time-series from fMRI output. Needs phantom-log file obtained from your BrainDancer device, slice-acquistion order of your acquistion protocol in csv format, and INU-corrected fMRI measurement in nifti format.<br/>

2) Run Example_DataQualityAssesment.ipynb: This file provides code for calculating data quality metrics namely – *Standardized SNR (ST-SNR), Dynamic Fidelity and Scanner Instability*, as defined in [Kumar et. al.](https://arxiv.org/abs/2004.06760). It requires three nifti files that were generated in step 1 output (measured fMRI data, ground-truth data and masks). 


### To be made available in next release:
1) Analysis routines for static phantoms - Temporal Singal-to-Fluctuation-Noise Ratio (SFNR), Weisskoff plot "Radius of Decorrelation".
2) Analysis routines for instability measurement using the FBIRN phantom, with two flip angle measurement [method]( https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.22691). 
3) Colored noise based motion sequence generator for the phantom.</br></br>





THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
