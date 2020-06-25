#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:13:00 2020

@author: sayin
"""
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from util import *

nx = 256
ny = 256

x = np.linspace(0,2.0*np.pi,nx+1)
y =  np.linspace(0,2.0*np.pi,ny+1)

nxc = 32
nyc = 32

xc = np.linspace(0,2.0*np.pi,nxc+1)
yc =  np.linspace(0,2.0*np.pi,nyc+1)



path = './data/2d_turb_'+str(nxc)+'_'+str(nx)+'_'+str(800)+'/data_'+str(nxc)+'_256_'+str(800)
pathout = 'output/'

data_  = np.load(path+'.npz')

data_hr = data_['data_hr']
data_lr = data_['data_lr']

box3 = data_hr[-1,:nx,:ny,1]
box1 = data_lr[-1,:nxc,:nyc,1]


Ti1 = interp2d(xc[:-1], yc[:-1], box1,kind='linear')
#Ti2 = interp2d(xc[:-1], yc[:-1], box1,kind='cubic')
Til = Ti1(x[:-1], y[:-1])
#Tib = Ti2(x[:-1], y[:-1])
#plt.contourf(X[:-1,:-1], Y[:-1,:-1], Ti)

enc, nc = energy_spectrum(nxc,nyc,box1)
#ensr, nsr = energy_spectrum_s(nx,ny, box2)
enf, nf = energy_spectrum(nx,ny,box3)
#    enn, nn = energy_spectrum_s(nx,ny,Ti)
enl, nl = energy_spectrum(nx,ny,Til)
#enb, nb = energy_spectrum_s(nx,ny,Tib)

kc = np.linspace(1,nc,nc)
kf = np.linspace(1,nf,nf)
#kb = np.linspace(1,nb,nb)

kl = kf[5:128]
line = 200*kl**(-3.0) #*kl**3


fig, ax = plt.subplots(figsize=(5,5))
ax.loglog(kf,enf[1:], 'g', lw = 2, label = 'DNS')
ax.loglog(kc,enc[1:], 'b--', lw = 2, label = 'FDNS')
#ax.loglog(kf,enn[1:], 'r', lw = 2, label = 'Nearest')
ax.loglog(kb,enl[1:], 'y', lw = 2, label = 'Linear')
#ax.loglog(kb,enb[1:], 'k', lw = 2, label = 'Cubic')

ax.loglog(kl,line, 'k-.', lw = 2)

plt.xlabel('$K$')
plt.ylabel('$E(K)$')
plt.legend(loc=0)
plt.xlim(1,256)
#plt.ylim(1e-8,1e-0)
plt.text(kl[10], line[10]+0.2, '$k^-3$', color='k')
#plt.savefig(folder+'energy_spec_32_256.pdf')
plt.show()

#%%
#fig, ax = plt.subplots(nrows=2, ncols=2)
## Plot the model function and the randomly selected sample points
#ax[0,0].contourf(Xc[:-1,:-1], Yc[:-1,:-1], lr_ref)
#ax[0,0].set_title('Sample points on f(X,Y)')
#
#for i, method in enumerate(('nearest', 'linear', 'cubic')):
#    Ti = griddata((Xcr, Ycr), lr_ref_r, (X[:-1,:-1], Y[:-1,:-1]), method=method)
##    Ti = griddata((Xc[:-1,:-1], Yc[:-1,:-1]), lr_ref, (X[:-1,:-1], Y[:-1,:-1]), method=method)
#    r, c = (i+1) // 2, (i+1) % 2
#    ax[r,c].contourf(X[:-1,:-1], Y[:-1,:-1], Ti)
#    ax[r,c].set_title("method = '{}'".format(method))

#%%
#
#x = np.linspace(-1,1,100)
#y =  np.linspace(-1,1,100)
#X, Y = np.meshgrid(x,y)
#
#def f(x, y):
#    s = np.hypot(x, y)
#    phi = np.arctan2(y, x)
#    tau = s + s*(1-s)/5 * np.sin(6*phi) 
#    return 5*(1-tau) + tau
#
#T = f(X, Y)
## Choose npts random point from the discrete domain of our model function
#npts = 400
#px, py = np.random.choice(x, npts), np.random.choice(y, npts)
#
#fig, ax = plt.subplots(nrows=2, ncols=2)
## Plot the model function and the randomly selected sample points
#ax[0,0].contourf(X[:-1,:-1], Y[:-1,:-1], lr_ref)
#ax[0,0].scatter(px, py, c='k', alpha=0.2, marker='.')
#ax[0,0].set_title('Sample points on f(X,Y)')
#
## Interpolate using three different methods and plot
#for i, method in enumerate(('nearest', 'linear', 'cubic')):
#    Ti = griddata((px, py), f(px,py), (X[:-1,:-1], Y[:-1,:-1]), method=method)
#    r, c = (i+1) // 2, (i+1) % 2
#    ax[r,c].contourf(X, Y, Ti)
#    ax[r,c].set_title("method = '{}'".format(method))
#
#plt.tight_layout()
#plt.show()