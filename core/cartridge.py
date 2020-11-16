# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: December 30, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Script for calculating the cartridge masks and the center of rotation. 

"""

# %% All imports
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import nibabel as nib, nilearn as nil
import matplotlib.patches as mpatches
from skimage.draw import circle_perimeter,line, polygon,circle, line_aa
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.measure import find_contours
from skimage.transform import (rotate as rt,rescale,downscale_local_mean)
from skimage.filters import sobel



# %%
# =============================================================================
# Finding the inner cartridge
# =============================================================================
def findcartridge(data,slice_num,volume_num,sig=2,lt=0,ht=100,rad1=8,rad2=52, step=1, n =3):
    image = data.get_data()[:,:,slice_num,volume_num]
    edges = canny(image, sigma=sig, low_threshold=lt, high_threshold=ht)
    hough_radii = np.arange(rad1, rad2, step)
    hough_res = hough_circle(edges, hough_radii)
    accums, cr, cc, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=n)
    return [image,edges,cr,cc,radii]

# %%
# =============================================================================================================
# Mask calculation for the inner cartridge (Obsolete!! Changed to the new algorith based on finding the notch!)
# =============================================================================================================


#def inner_mask(data,findcartridge_parameters,slice_num,volume_num):
    
    #data_best_slices = data
    #temp_ind = findcartridge_parameters
    #count = 0
    #choice = 0
    
    #while(choice != 1):
        #if count == 0:
            #r_cord_ind = np.argmin(temp_ind[4])
            #r_cord = temp_ind[4][r_cord_ind]
            #x_cord = temp_ind[2][r_cord_ind]
            #y_cord = temp_ind[3][r_cord_ind]
            
        #else:
            #user_input = [float(p) for p in input('Enter x,y,r with a space').split()]
            #x_cord = user_input[0]
            #y_cord = user_input[1]
            #r_cord = user_input[2]
        
        #mask_image = np.zeros(data_best_slices.get_data()[:,:,slice_num,volume_num].shape) 
        #patch = mpatches.Wedge((y_cord,x_cord),r_cord,0,360)  
        #vertices = patch.get_path().vertices
        #x=[]
        #y=[]
        
        #for k in range(len(vertices)):
            #x.append(int(vertices[k][0]))
            #y.append(int(vertices[k][1]))
        #x.append(x[0])
        #y.append(y[0])
        #rr,cc = polygon(x,y)
        #mask_image[rr, cc] = 1

        
        #plt.figure()
        #plt.imshow(mask_image*np.mean(data_best_slices.get_data()[:,:,slice_num,volume_num].flatten())*5 + data_best_slices.get_data()[:,:,slice_num,volume_num])
        #plt.show()
        #print('Currently used x,y,r',[x_cord,y_cord,r_cord])
        #choice_list = [int(x) for x in input('Enter 1 to go to next slice, 0 to change x,y,r').split()]
        #choice = choice_list[0]
    
        #if choice == 1:
            #mask = mask_image
            
            #center = [x_cord,y_cord,r_cord] # this is the center of the mask
        #count +=1
    
    #return mask,center

    
# %%
# =================================================================================================
# Mask calculation for the inner cartridge 
# =================================================================================================

def inner_mask(data_path,slice_num,volume_num=0,lvl=0.004,rad1=7,rad2=50,step=1):
    
    im = nib.load(data_path).get_data()[:,:,slice_num,volume_num]
    im_sobel = sobel(im)
    repeat = 1
    
    while(repeat):
        contours =find_contours(im_sobel,level=lvl,fully_connected='high')
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
        print('If this is not showing four circle boundaries, clearly - you need to change the thresholding level for finding contours again. If not changed, all subsequent estimation might fail.')
        repeat = int(input('Do you want to repeat and change the thresholding level? 1 for yes, 0 for no'))
        if repeat:
            print('Current level is:',lvl)
            print('\n')
            lvl = input('Enter the new lvl (integer)')
            
    smallest_circle = [] #detects the inner circle with notch
    for i in range(len(contours)):
        smallest_circle.append(contours[i].shape[0])
    temp_var = np.array(smallest_circle)
    temp_var = np.delete(temp_var,np.argmax(temp_var))
    temp_var = np.delete(temp_var,np.argmax(temp_var))
    temp_var = np.delete(temp_var,np.argmax(temp_var))
    index = np.argwhere(np.array(smallest_circle)==np.max(temp_var))[0][0]
    
    img = np.zeros(im.shape)
    img[(contours[index][:,0]).astype('int'),(contours[index][:,1]).astype('int')]=1
     
    hough_radii = np.arange(rad1, rad2, step)
    hough_res = hough_circle(img, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=2)    
    radii_complete = radii[np.argmax(radii)]  # complete refers to the full circle with notch
    cx_complete = cx[np.argmax(radii)]
    cy_complete = cy[np.argmax(radii)]

    radii_incomplete = radii[np.argmin(radii)]  # complete refers to the circle without notch
    cx_incomplete = cx[np.argmin(radii)]
    cy_incomplete = cy[np.argmin(radii)]
    
    rr,cc = circle(cy_complete,cx_complete,radii_complete-1) # Erroded by 1 voxel for removing the notch
    img_complete = np.zeros(im.shape)
    img_complete[rr,cc]=1
    
    rr,cc = circle(cy_incomplete,cx_incomplete,radii_incomplete)
    img_incomplete = np.zeros(im.shape)
    img_incomplete[rr,cc]=1
    
    
    return img_complete,cy_complete,cx_complete,radii_complete 


# %%
# =================================================================================================
# Finding the center of rotation
# =================================================================================================

def cen_rotation(data_path,slice_num,img_complete,cy_complete,cx_complete,radii_complete,canny_sgm=1):
    
    temp_img= img_complete * (nib.load(data_path).get_data()[:,:,slice_num,0])
    
    cir_mask = np.zeros(temp_img.shape)
    rr,cc = circle(cy_complete,cx_complete,radii_complete-2) # erosion to get rid of boundaries
    cir_mask[rr,cc] = 1
    
    contrast_enh= exposure.equalize_hist(temp_img)
    sobel_edges = sobel(contrast_enh)
    sobel_masked = sobel_edges  *cir_mask
    im = np.power(sobel_masked,5) # increases the contrast such that the quadrant intersection is visible; depends on T2* relaxation, so can vary with the age of the cartridge. 
    

    
    dotp_all = []
    for i in range(len(np.nonzero(cir_mask)[0])):
        possible_angles = np.linspace(0,360,720)
        all_coords = np.nonzero(cir_mask)
   
        test_line = np.zeros(temp_img.shape)
        rr,cc = line(all_coords[0][i]-(radii_complete-1),all_coords[1][i],all_coords[0][i]+(radii_complete-1),all_coords[1][i])
        test_line[rr,cc]=1
        rr,cc = line(all_coords[0][i],all_coords[1][i]-(radii_complete-1),all_coords[0][i],all_coords[1][i]+(radii_complete-1))
        test_line[rr,cc]=1
    
        for j in possible_angles:
            test_line_rt = rt(test_line, angle = j,center=(np.array((all_coords[1][i],all_coords[0][i]))), order=3, preserve_range=True) # the format for center is col, row; not row, col - documentation of skimage is incorrect for 0.13.x
            dotp = np.sum((test_line_rt * im).flatten())
            dotp_all.append(dotp)
   
    row_cor = np.nonzero(cir_mask)[0][int(np.argmax(dotp_all)/len(possible_angles))]
    col_cor = np.nonzero(cir_mask)[1][int(np.argmax(dotp_all)/len(possible_angles))]
    angle_move = possible_angles[np.argmax(dotp_all[(int(np.argmax(dotp_all)/720)*720):(int(np.argmax(dotp_all)/720)*720)+720])]  
    
    ## Just for visualization
    
    vis_line = np.zeros(temp_img.shape)
    rr,cc = line(row_cor-(radii_complete-1),col_cor,row_cor+(radii_complete-1),col_cor)
    vis_line[rr,cc]=1
    rr,cc = line(row_cor,col_cor-(radii_complete-1),row_cor,col_cor+(radii_complete-1))
    vis_line[rr,cc]=1
    vis_line_rt = rt(vis_line, angle =angle_move,center=(np.array((col_cor,row_cor))), order=3, preserve_range=True)

    plt.imshow(temp_img)
    plt.title('Original Image')
    plt.figure()
    plt.imshow(contrast_enh)
    plt.title('Contrast Enhanced Image')
    plt.figure()
    plt.imshow(im)
    plt.title('Sobel Image')
    plt.figure()
    plt.imshow(vis_line_rt )
    plt.title('Estimated Center')
    plt.show()

    print('COR,COC',row_cor,col_cor)
        
    
    return [row_cor,col_cor]

# %%
# =================================================================================================
# Mask calculation for the outer cartridge 
# =================================================================================================

def outer_mask(data,findcartridge_parameters,slice_num,volume_num):
    
    data_best_slices = data
    temp_ind = findcartridge_parameters
    count = 0
    choice = 0
    
    while(choice != 1):
        if count == 0:
            r_cord_ind = np.argmax(temp_ind[4])
            r_cord_neg = np.argmin(temp_ind[4])
            
            for w in [0,1,2]:
                if (w!= r_cord_ind) and (w!=r_cord_neg):
                    mid = w
            
            r_cord = temp_ind[4][r_cord_ind]
            x_cord = temp_ind[3][r_cord_ind]
            y_cord = temp_ind[2][r_cord_ind]
            #w = temp_ind[4][mid]
            w = r_cord-14
        else:
            user_input = [float(p) for p in input('Enter x,y,r,w with a space').split()]
            x_cord = user_input[0] 
            y_cord = user_input[1]
            r_cord = user_input[2]
            w = user_input[3] 
        omask_image = np.zeros(data_best_slices.get_data()[:,:,slice_num,volume_num].shape)
        patch = mpatches.Wedge((x_cord,y_cord),r_cord,0,360,w) # Checked the values for an excellent overlap mask for inner cyl.
        vertices = patch.get_path().vertices
        x=[]
        y=[]
        for k in range(len(vertices)):
            x.append(int(vertices[k][0]))
            y.append(int(vertices[k][1]))
        x.append(x[0])
        y.append(y[0])
        rr,cc = polygon(x,y)
        omask_image[rr, cc] = 1

        
        plt.figure()
        plt.imshow(omask_image*np.mean(data_best_slices.get_data()[:,:,slice_num,volume_num].flatten())*5 + data_best_slices.get_data()[:,:,slice_num,volume_num])
        plt.show()
        print('Currently used x,y,r,w',[x_cord,y_cord,r_cord,w])
        choice_list = [int(x) for x in input('Enter 1 to go to next slice, 0 to change x,y,r,w').split()]
        choice = choice_list[0]
    
        if choice == 1:
            omask = omask_image            
        count +=1
    
    return omask



# %%