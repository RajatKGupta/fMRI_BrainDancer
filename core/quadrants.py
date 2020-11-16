# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: September 13, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Script for calculating quadrant-wise masks for the inner cylinder. 
# Needs input image to be inner cylinder images.

"""

# %% All imports
from skimage import measure, feature,draw
from skimage.transform import rescale, downscale_local_mean, resize, rotate as rt
from skimage.draw import circle_perimeter,line, polygon
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# %%
# =============================================================================
# Calculating quadrant-wise mask for EPI images. 
# =============================================================================

def quadrant_mask_EPI(inp_image,x,y,r,lev=300):
    
    temp = x
    x = y
    y=temp
    
    contours = measure.find_contours(inp_image,level=lev)
    
    #display contour overlaid on original input - Checkpoint
    fig, ax = plt.subplots()
    ax.imshow(inp_image,interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    plt.show()
    
    dist =[]
    for i in range(len(contours[0])):
        ecd = np.sqrt(pow((x-contours[0][i][0]),2) + pow((y-contours[0][i][1]),2))
        dist.append(ecd)
    dist = np.array(dist)
    
    coords =draw.line(x,y,int(contours[0][np.argmin(dist)][0]),int(contours[0][np.argmin(dist)][1]))
    mark = np.zeros(inp_image.shape)
    mark[coords] = 1
    
    rt_template = np.zeros(inp_image.shape)
    new_coords = draw.line(x,y,x,y+r)
    rt_template[new_coords]=1
    
    dot = []
    for i in range(0,360):
        dot.append(np.sum((mark*rt(rt_template,i,preserve_range=True,center=(y,x)))))
    dot = np.array(dot)
    #Checkpoint
    plt.imshow(mark+rt(rt_template,np.argmax(dot),preserve_range=True,center=(y,x)))
    
    
    #MASK1
    wedge_mask1 = np.zeros(inp_image.shape)
    wedge_mask1 = rescale(wedge_mask1,10,mode='constant')
    wedge1 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+45-22.5,np.argmax(dot)+45+22.5,(r-1)*10)
    ver1 = wedge1.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver1)):
        q.append(int(ver1[i][0]))
        t.append(int(ver1[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask1[rr, cc] = 1
    
    #MASK2
    wedge_mask2 = np.zeros(inp_image.shape)
    wedge_mask2 = rescale(wedge_mask2,10,mode='constant')
    wedge2 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+135-22.5,np.argmax(dot)+135+22.5,(r-1)*10)
    ver2 = wedge2.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver2)):
        q.append(int(ver2[i][0]))
        t.append(int(ver2[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask2[rr, cc] = 1
    
    #MASK3
    wedge_mask3 = np.zeros(inp_image.shape)
    wedge_mask3 = rescale(wedge_mask3,10,mode='constant')
    wedge3 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+225-22.5,np.argmax(dot)+225+22.5,(r-1)*10)
    ver3 = wedge3.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver3)):
        q.append(int(ver3[i][0]))
        t.append(int(ver3[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask3[rr, cc] = 1
    
    #MASK4
    wedge_mask4 = np.zeros(inp_image.shape)
    wedge_mask4 = rescale(wedge_mask4,10,mode='constant')
    wedge4 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+315-22.5,np.argmax(dot)+315+22.5,(r-1)*10)
    ver4 = wedge4.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver4)):
        q.append(int(ver4[i][0]))
        t.append(int(ver4[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask4[rr, cc] = 1
    
    mm1 = downscale_local_mean(wedge_mask1,(10,10))
    mm2 = downscale_local_mean(wedge_mask2,(10,10))
    mm3 = downscale_local_mean(wedge_mask3,(10,10))
    mm4 = downscale_local_mean(wedge_mask4,(10,10))
    
    return mm1,mm2,mm3,mm4

# %%
# =============================================================================
# Calculating quadrant-wise mask for T2* images. 
# =============================================================================

def quadrant_mask_T2(inp_image,x,y,r,lev=300):
    
    temp = x
    x = y
    y=temp
    
    contours = measure.find_contours(inp_image,level=lev)
    
    #display contour overlaid on original input - Checkpoint
    fig, ax = plt.subplots()
    ax.imshow(inp_image,interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    plt.show()
    
    dist =[]
    for i in range(len(contours[0])):
        ecd = np.sqrt(pow((x-contours[0][i][0]),2) + pow((y-contours[0][i][1]),2))
        dist.append(ecd)
    dist = np.array(dist)
    
    coords =draw.line(x,y,int(contours[0][np.argmin(dist)][0]),int(contours[0][np.argmin(dist)][1]))
    mark = np.zeros(inp_image.shape)
    mark[coords] = 1
    
    rt_template = np.zeros(inp_image.shape)
    new_coords = draw.line(x,y,x,y+r)
    rt_template[new_coords]=1
    
    dot = []
    for i in range(0,360):
        dot.append(np.sum((mark*rt(rt_template,i,preserve_range=True,center=(y,x)))))
    dot = np.array(dot)
    #Checkpoint
    plt.imshow(mark+rt(rt_template,np.argmax(dot),preserve_range=True,center=(y,x)))
    
    
    #MASK1
    wedge_mask1 = np.zeros(inp_image.shape)
    wedge_mask1 = rescale(wedge_mask1,10,mode='constant')
    wedge1 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)-45,np.argmax(dot)+45,(r-1)*10)
    ver1 = wedge1.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver1)):
        q.append(int(ver1[i][0]))
        t.append(int(ver1[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask1[rr, cc] = 1
    
    #MASK2
    wedge_mask2 = np.zeros(inp_image.shape)
    wedge_mask2 = rescale(wedge_mask2,10,mode='constant')
    wedge2 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+90-45,np.argmax(dot)+90+45,(r-1)*10)
    ver2 = wedge2.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver2)):
        q.append(int(ver2[i][0]))
        t.append(int(ver2[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask2[rr, cc] = 1
    
    #MASK3
    wedge_mask3 = np.zeros(inp_image.shape)
    wedge_mask3 = rescale(wedge_mask3,10,mode='constant')
    wedge3 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+180-45,np.argmax(dot)+180+45,(r-1)*10)
    ver3 = wedge3.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver3)):
        q.append(int(ver3[i][0]))
        t.append(int(ver3[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask3[rr, cc] = 1
    
    #MASK4
    wedge_mask4 = np.zeros(inp_image.shape)
    wedge_mask4 = rescale(wedge_mask4,10,mode='constant')
    wedge4 = mpatches.Wedge((x*10,y*10),(r-1)*10,np.argmax(dot)+270-45,np.argmax(dot)+270+45,(r-1)*10)
    ver4 = wedge4.get_path().vertices
    q=[]
    t=[]
    for i in range(len(ver4)):
        q.append(int(ver4[i][0]))
        t.append(int(ver4[i][1]))
    q.append(q[0])
    t.append(t[0])
    rr,cc = polygon(q,t)
    wedge_mask4[rr, cc] = 1
    
    mm1 = downscale_local_mean(wedge_mask1,(10,10))
    mm2 = downscale_local_mean(wedge_mask2,(10,10))
    mm3 = downscale_local_mean(wedge_mask3,(10,10))
    mm4 = downscale_local_mean(wedge_mask4,(10,10))
    
    return mm1,mm2,mm3,mm4

# %%
