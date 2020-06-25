#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:42:59 2019

@author: sayin
"""

import numpy as np
import os
from matplotlib import pyplot as plt

#%%
nf = 800
nx = 257
ny = 257

path = '../../../00_data/01_gan_data/spectral/data_256_800/02_vorticity/'
#print(os.listdir(path))

w = np.empty([0,1])    
for i in range(1,nf+1):
    wc1   = np.load(path+'w_'+str(i)+'.npy')
    temp2 = wc1.reshape((-1,1))
    w     = np.vstack((w,temp2))    
    
w = w.reshape(nf,nx,ny)

path = '../../../00_data/01_gan_data/spectral/data_256_800/01_streamfunction/'
s = np.empty([0,1])    
for i in range(1,nf+1):
    sc2    =  np.load(path+'s_'+str(i)+'.npy')
    temp3 = sc2.reshape((-1,1))
    s   = np.vstack((s,temp3)) 
    
s = s.reshape(nf,nx,ny)

#%%
plt.figure()
img=plt.imshow(np.flipud(s[200]),cmap='jet',vmin=-0.32,vmax=0.32)
plt.colorbar(img, ticks=np.linspace(-0.32,0.32,9))
plt.show()
#%%
plt.figure()
img=plt.contourf(s[200],101,cmap='jet',vmin=-0.32,vmax=0.32)
plt.colorbar(img, ticks=np.linspace(-0.32,0.32,9))  
plt.show()

#%%

nxc = 33
nyc = 33

path = '../../../00_data/01_gan_data/spectral/data_256_32_800/00_wc/'
#print(os.listdir(path))
i =1
wc1   = np.load(path+'wc_'+str(i)+'.npy') 
#%%
wc = np.empty([0,1])    
for i in range(1,nf+1):
    wc1   = np.load(path+'wc_'+str(i)+'.npy')
    temp2 = wc1.reshape((-1,1))
    wc     = np.vstack((wc,temp2))    
    
wc = wc.reshape(nf,nxc,nyc)

path = '../../../00_data/01_gan_data/spectral/data_256_32_800/00_sc/'
sc = np.empty([0,1])    
for i in range(1,nf+1):
    sc1   = np.load(path+'sc_'+str(i)+'.npy')
    temp3 = sc1.reshape((-1,1))
    sc   = np.vstack((sc,temp3)) 
    
sc = sc.reshape(nf,nxc,nyc)

#%%
plt.figure()
img=plt.imshow(np.flipud(wc[-1]),cmap='jet')
plt.colorbar(img)
plt.show()

plt.figure()
img=plt.contourf(wc[-1],101,cmap='jet')
plt.colorbar(img)
plt.show()


#%%

data_hr = np.empty((s.shape[0],nx,ny,2))

for i in range(s.shape[0]):
    xt = s[i,:,:]
    yt = w[i,:,:]
    rt = np.dstack((xt, yt))
    data_hr[i,:,:,:] = rt
    
data_lr = np.empty((sc.shape[0],nxc,nyc,2))
for i in range(sc.shape[0]):
    xt = sc[i,:,:]
    yt = wc[i,:,:]
    rt = np.dstack((xt, yt))
    data_lr[i,:,:,:] = rt      
    
 
  
#%%
temp = nxc-1    
path =  './data/2d_turb_'+ str(temp) +'_256_'+ str(nf)+'/'
np.savez(path+'data_'+str(temp) +'_256_'+str(nf)+'.npz', data_hr=data_hr,data_lr=data_lr)


#%%
#path = './data/turb_data/input/'
#print(os.listdir(path))

#nf = 400
#nx = 1025
#ny = 1025
#
##u data 
#u = np.empty([0,1])    
#for i in range(1,nf+1):
#    uc1    = np.load(path+'u_'+str(i)+'.npy')
#    temp2 = uc1.reshape((-1,1))
#    u   = np.vstack((u,temp2))    
#    
#u = u.reshape(nf,nx,ny)
#
##v data 
#v = np.empty([0,1])    
#for i in range(1,nf+1):
#    vc2    = np.load(path+'v_'+str(i)+'.npy')
#    temp3 = vc2.reshape((-1,1))
#    v   = np.vstack((v,temp3)) 
#    
#v = v.reshape(nf,nx,ny)

#%%
#path = './data/turb_data/output/'
#
#nxc = 65
#nyc = 65
#
##u data 
#u = np.empty([0,1])    
#for i in range(1,nf+1):
#    uc1    = np.load(path+'uc_'+str(i)+'.npy')
#    temp2 = uc1.reshape((-1,1))
#    u   = np.vstack((u,temp2))    
#    
#uc = u.reshape(nf,nxc,nyc)
#
##v data 
#v = np.empty([0,1])    
#for i in range(1,nf+1):
#    vc2    = np.load(path+'vc_'+str(i)+'.npy')
#    temp3 = vc2.reshape((-1,1))
#    v   = np.vstack((v,temp3)) 
#    
#vc = v.reshape(nf,nxc,nyc)   
#path =  './data/turb_data/input/'
#np.savez(path+'coarse.npz',wc=wc,sc=sc,vc=vc,uc=uc)
#
#path =  './data/turb_data/output/'
#np.savez(path+'fine.npz',w=w,s=s,v=v,u=u)
