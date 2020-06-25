#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:51:13 2019

@author: Suraj Pawar

DNS solver for decaying homegenous isotropic turbulence problem for cartesian periodic 
domain with [0,2pi] X [0,2pi] dimension and is discretized uniformly in x and y direction. 
The solver uses pseudo-spectral method for solving two-dimensional incompressible 
Navier-Stokes equation in vorticity-streamfunction formulation. The solver employs 
hybrid third-order Runge-Kutta implicit Crank-Nicolson scheme for time integration. 

"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import UnivariateSpline
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ESRGAN_2d import *
import keras.backend as K
from keras.models import load_model


font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def exact_tgv(nx,ny,time,re):
    
    '''
    compute exact solution for TGV problem
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    time : time at which the exact solution is to be computed
    re : Reynolds number
    
    Output
    ------
    ue : exact solution for TGV problem
    '''
    
    ue = np.empty((nx+1,ny+1))
    x = np.linspace(0.0,2.0*np.pi,nx+1)
    y = np.linspace(0.0,2.0*np.pi,ny+1)
    x, y = np.meshgrid(x, y, indexing='ij')
    
    nq = 4.0
    ue = 2.0*nq*np.cos(nq*x)*np.cos(nq*y)*np.exp(-2.0*nq*nq*time/re)
    

    return ue

#%%
def tgv_ic(nx,ny):
    
    '''
    compute initial condition for TGV problem
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    
    Output
    ------
    w : initial condiition for vorticity for TGV problem
    '''
    
    w = np.empty((nx+1,ny+1))
    nq = 4.0
    x = np.linspace(0.0,2.0*np.pi,nx+1)
    y = np.linspace(0.0,2.0*np.pi,ny+1)
    x, y = np.meshgrid(x, y, indexing='ij')
    
    w = 2.0*nq*np.cos(nq*x)*np.cos(nq*y)

    return w

#%%
def vm_ic(nx,ny):
    
    '''
    compute initial condition for vortex-merger problem
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    
    Output
    ------
    w : initial condiition for vorticity for vortex-merger problem
    '''
    
    w = np.empty((nx+1,ny+1))

    sigma = np.pi
    xc1 = np.pi-np.pi/4.0
    yc1 = np.pi
    xc2 = np.pi+np.pi/4.0
    yc2 = np.pi
    
    x = np.linspace(0.0,2.0*np.pi,nx+1)
    y = np.linspace(0.0,2.0*np.pi,ny+1)
    
    x, y = np.meshgrid(x, y, indexing='ij')
    
    w = np.exp(-sigma*((x-xc1)**2 + (y-yc1)**2)) \
            + np.exp(-sigma*((x-xc2)**2 + (y-yc2)**2))

    return w

#%%
def pbc(nx,ny,u):
    
    '''
    assign periodic boundary condition in physical space
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    u : solution field
    
    Output
    ------
    u : solution field with periodic boundary condition applied
    '''    
    
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]

#%%
# set initial condition for decay of turbulence problem
def decay_ic(nx,ny,dx,dy,k0):
    
    '''
    assign initial condition for vorticity for DHIT problem
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    dx,dy : grid spacing in x and y direction
    
    Output
    ------
    w : initial condition for vorticity for DHIT problem
    '''
    
    w = np.empty((nx+1,ny+1))
    
    epsilon = 1.0e-6
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    ksi = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    eta = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    
    phase = np.zeros((nx,ny), dtype='complex128')
    wf =  np.empty((nx,ny), dtype='complex128')
    
    phase[1:int(nx/2),1:int(ny/2)] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,1:int(ny/2)] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]))

    phase[1:int(nx/2),ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                 eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                eta[1:int(nx/2),1:int(ny/2)]))

    k0 = k0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es = c*(kk**4)*np.exp(-(kk/k0)**2)
    wf[:,:] = np.sqrt((kk*es/np.pi)) * phase[:,:]*(nx*ny)
            
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    ut = np.real(fft_object_inv(wf)) 
    
    #periodicity
    w[0:nx,0:ny] = ut
    w[:,ny] = w[:,0]
    w[nx,:] = w[0,:]
    w[nx,ny] = w[0,0] 
    
    return w

#%%
def wave2phy(nx,ny,uf):
    
    '''
    Converts the field form frequency domain to the physical space.
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    uf : solution field in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    u : solution in physical space (along with periodic boundaries)
    '''
    
    u = np.empty((nx+1,ny+1))
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')

    u[0:nx,0:ny] = np.real(fft_object_inv(uf))
    # periodic BC
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    
    return u

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,w):
    
    '''
    Computation of energy spectrum and maximum wavenumber from vorticity field
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    w : vorticity field in physical spce (including periodic boundaries)
    
    Output
    ------
    en : energy spectrum computed from vorticity field
    n : maximum wavenumber
    '''
    
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    dx = 2*np.pi/nx
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[0:nx,0:ny]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
                    
        en[k] = en[k]/ic
        
    return en, n


#%%
# fast poisson solver using second-order central difference scheme
def fps(nx,ny,dx,dy,k2,f):
    
    '''
    FFT based fast poisson solver 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction
    dx,dy : grid spacing in x and y direction
    k2 : absolute wavenumber over 2D domain
    f : right hand side of poisson equation in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    u : solution to the Poisson eqution in physical space (including periodic boundaries)
    '''
    
    u = np.zeros((nx+1,ny+1))
         
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
       
    # the donominator is based on the scheme used for discrtetizing the Poisson equation
    data1 = f/(-k2)
    
    # compute the inverse fourier transform
    u[0:nx,0:ny] = np.real(fft_object_inv(data1))
    pbc(nx,ny,u)
    
    return u


#%%
def coarsen(nx,ny,nxc,nyc,uf):  
    
    '''
    coarsen the data along with the size of the data 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    uf : solution field on fine grid in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    u : caorsened solution in frequency domain (excluding periodic boundaries)
    '''
    
    ufc = np.zeros((nxc,nyc),dtype='complex')
    
    ufc[0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]
    ufc[int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)]    
    ufc[0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):]
    ufc[int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    
    return ufc

       
#%%
def nonlineardealiased(nx,ny,kx,ky,k2,wf):    
    
    '''
    compute the Jacobian with 3/2 dealiasing 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kx,ky : wavenumber in x and y direction
    k2 : absolute wave number over 2D domain
    wf : vorticity field in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    jf : jacobian in frequency domain (excluding periodic boundaries)
         (d(psi)/dy*d(omega)/dx - d(psi)/dx*d(omega)/dy)
    '''
    
    j1f = 1.0j*kx*wf/k2
    j2f = 1.0j*ky*wf
    j3f = 1.0j*ky*wf/k2
    j4f = 1.0j*kx*wf
    
    nxe = int(nx*3/2)
    nye = int(ny*3/2)
    
    j1f_padded = np.zeros((nxe,nye),dtype='complex128')
    j2f_padded = np.zeros((nxe,nye),dtype='complex128')
    j3f_padded = np.zeros((nxe,nye),dtype='complex128')
    j4f_padded = np.zeros((nxe,nye),dtype='complex128')
    
    j1f_padded[0:int(nx/2),0:int(ny/2)] = j1f[0:int(nx/2),0:int(ny/2)]
    j1f_padded[int(nxe-nx/2):,0:int(ny/2)] = j1f[int(nx/2):,0:int(ny/2)]    
    j1f_padded[0:int(nx/2),int(nye-ny/2):] = j1f[0:int(nx/2),int(ny/2):]    
    j1f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j1f[int(nx/2):,int(ny/2):] 
    
    j2f_padded[0:int(nx/2),0:int(ny/2)] = j2f[0:int(nx/2),0:int(ny/2)]
    j2f_padded[int(nxe-nx/2):,0:int(ny/2)] = j2f[int(nx/2):,0:int(ny/2)]    
    j2f_padded[0:int(nx/2),int(nye-ny/2):] = j2f[0:int(nx/2),int(ny/2):]    
    j2f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j2f[int(nx/2):,int(ny/2):] 
    
    j3f_padded[0:int(nx/2),0:int(ny/2)] = j3f[0:int(nx/2),0:int(ny/2)]
    j3f_padded[int(nxe-nx/2):,0:int(ny/2)] = j3f[int(nx/2):,0:int(ny/2)]    
    j3f_padded[0:int(nx/2),int(nye-ny/2):] = j3f[0:int(nx/2),int(ny/2):]    
    j3f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j3f[int(nx/2):,int(ny/2):] 
    
    j4f_padded[0:int(nx/2),0:int(ny/2)] = j4f[0:int(nx/2),0:int(ny/2)]
    j4f_padded[int(nxe-nx/2):,0:int(ny/2)] = j4f[int(nx/2):,0:int(ny/2)]    
    j4f_padded[0:int(nx/2),int(nye-ny/2):] = j4f[0:int(nx/2),int(ny/2):]    
    j4f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j4f[int(nx/2):,int(ny/2):] 
    
    j1f_padded = j1f_padded*(nxe*nye)/(nx*ny)
    j2f_padded = j2f_padded*(nxe*nye)/(nx*ny)
    j3f_padded = j3f_padded*(nxe*nye)/(nx*ny)
    j4f_padded = j4f_padded*(nxe*nye)/(nx*ny)
    
    
    a = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a1 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b1 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a2 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b2 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a3 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b3 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a4 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b4 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    fft_object_inv1 = pyfftw.FFTW(a1, b1,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv2 = pyfftw.FFTW(a2, b2,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv3 = pyfftw.FFTW(a3, b3,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv4 = pyfftw.FFTW(a4, b4,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    j1 = np.real(fft_object_inv1(j1f_padded))
    j2 = np.real(fft_object_inv2(j2f_padded))
    j3 = np.real(fft_object_inv3(j3f_padded))
    j4 = np.real(fft_object_inv4(j4f_padded))
    
    jacp = j1*j2 - j3*j4
    
    jacpf = fft_object(jacp)
    
    jf = np.zeros((nx,ny),dtype='complex128')
    
    jf[0:int(nx/2),0:int(ny/2)] = jacpf[0:int(nx/2),0:int(ny/2)]
    jf[int(nx/2):,0:int(ny/2)] = jacpf[int(nxe-nx/2):,0:int(ny/2)]    
    jf[0:int(nx/2),int(ny/2):] = jacpf[0:int(nx/2),int(nye-ny/2):]    
    jf[int(nx/2):,int(ny/2):] =  jacpf[int(nxe-nx/2):,int(nye-ny/2):]
    
    jf = jf*(nx*ny)/(nxe*nye)
    
    return jf

#%%
# def coarsenf(nx,ny,nxc,nyc,uf):
    
#     '''
#     coarsen the solution field along with the size of the data 
    
#     Inputs
#     ------
#     nx,ny : number of grid points in x and y direction on fine grid
#     nxc,nyc : number of grid points in x and y direction on coarse grid
#     u : solution field on fine grid
    
#     Output
#     ------
#     uc : solution field on coarse grid [nxc X nyc]
#     '''
    
   
#     ufc = np.zeros((nxc,nyc),dtype='complex')
    
#     ufc [0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]     
#     ufc [int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)] 
#     ufc [0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):] 
#     ufc [int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
       
#     return ufc

#%%
def nonlineardealiased_ad(nx,ny,kx,ky,k2,nxf,nyf,kxf,kyf,k2f,wnf,gan_d):
    
    wp = wave2phy(nx, ny, wnf)
    sp = fps(nx,ny,dx,dy,k2,-wnf)
    
    data = np.load('scaling_3.npz')
    data1 = np.load('scaling.npz')
    
    s_hr_max1, s_hr_min1 = data1['sh'][0], data1['sh'][1]
    w_hr_max1, w_hr_min1 = data1['wh'][0], data1['wh'][1]
    s_lr_max1, s_lr_min1 = data1['sl'][0], data1['sl'][1]
    w_lr_max1, w_lr_min1 = data1['wl'][0], data1['wl'][1]
    
    s_hr_max, s_hr_min = data['s_hr_max'], data['s_hr_min']
    w_hr_max, w_hr_min = data['w_hr_max'], data['w_hr_min']
    s_lr_max, s_lr_min = data['s_lr_max'], data['s_lr_min']
    w_lr_max, w_lr_min = data['w_lr_max'], data['w_lr_min']
        
    imgs_lr = np.zeros((1,nx,ny,2))
    
    sp_n = (sp - s_lr_min)/(s_lr_max - s_lr_min)
    wp_n = (wp - w_lr_min)/(w_lr_max - w_lr_min)
        
    imgs_lr[0,:,:,0] = sp_n[:nx,:ny]
    imgs_lr[0,:,:,1] = wp_n[:nx,:ny]
    imgs_hr = gan_d.generator.predict(imgs_lr,steps=1)
    
    #s_sr = imgs_hr[0,:,:,0]
    w_sr = imgs_hr[0,:,:,1]
    #s_sr = s_sr*(s_hr_max - s_hr_min) + s_hr_min
    w_sr = w_sr*(w_hr_max - w_hr_min) + w_hr_min
    
    
    # enc, nc = energy_spectrum(nx,ny,wp)
    # ensr, nf = energy_spectrum(nxf,nyf,w_sr)
    # kc = np.linspace(1,nc,nc)
    # kf = np.linspace(1,nf,nf)
    
    # fig,ax = plt.subplots(1,3,figsize=(12,4))
    # cs = ax[0].imshow(np.flipud(wp),cmap='jet'); 
    # fig.colorbar(cs,ax=ax[0],shrink =0.6)
    # cs = ax[1].imshow(np.flipud(w_sr),cmap='jet'); 
    # fig.colorbar(cs,ax=ax[1],shrink =0.6)
    # ax[2].loglog(kc,enc[1:], 'b', lw = 2, label = 'UNS')
    # ax[2].loglog(kf,ensr[1:], 'r', lw = 2, label = 'SRGAN')
    # ax[2].legend(loc=0)
    # plt.show()
    
    a = pyfftw.empty_aligned((nxf,nyf),dtype= 'complex128')
    b = pyfftw.empty_aligned((nxf,nyf),dtype= 'complex128')
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    wnf_fine = fft_object(w_sr)
    #wnf_fine = k2f*snf_fine # w = -lap(psi) => w_fft = -(-k^2*psi_fft)
    
    jf_fine = nonlineardealiased(nxf,nyf,kxf,kyf,k2f,wnf_fine)
    
    jf = coarsen(nxf,nyf,nx,ny,jf_fine)
    
    return jf
        
#%%
def nonlinear(nx,ny,kx,ky,k2,wf):  
    
    '''
    compute the Jacobian without dealiasing 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kx,ky : wavenumber in x and y direction
    k2 : absolute wave number over 2D domain
    wf : vorticity field in frequency domain (excluding periodic boundaries)
    
    Output
    ------
    jf : jacobian in frequency domain (excluding periodic boundaries)
         (d(psi)/dx*d(omega)/dy - d(psi)/dy*d(omega)/dx)
    '''
    
    j1f = 1.0j*kx*wf/k2
    j2f = 1.0j*ky*wf
    j3f = 1.0j*ky*wf/k2
    j4f = 1.0j*kx*wf
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a1 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b1 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a2 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b2 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a3 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b3 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a4 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b4 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    fft_object_inv1 = pyfftw.FFTW(a1, b1,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv2 = pyfftw.FFTW(a2, b2,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv3 = pyfftw.FFTW(a3, b3,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv4 = pyfftw.FFTW(a4, b4,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    j1 = np.real(fft_object_inv1(j1f))
    j2 = np.real(fft_object_inv2(j2f))
    j3 = np.real(fft_object_inv3(j3f))
    j4 = np.real(fft_object_inv4(j4f))
    
    jac = j1*j2 - j3*j4
    
    jf = fft_object(jac)
    
    return jf


#%% coarsening
def write_data(nx,ny,dx,dy,kx,ky,k2,nxc,nyc,dxc,dyc,wf,w0,n,freq,dt):
    
    '''
    write the data to .csv files for post-processing
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    dx,dy : grid spacing in x and y direction
    kx,ky : wavenumber in x and y direction
    k2 : absolute wave number over 2D domain
    nxc,nyc : number of grid points in x and y direction on caorse grid
    dxc,dyc : grid spacing in x and y direction for coarse grid
    wf : vorticity field in frequency domain (excluding periodic boundaries)
    n : time step
    freq : frequency at which to write the data
    
    Output/ write
    ------
    jc : coarsening of Jacobian computed at fine grid
    jcoarse : Jacobian computed for coarsed solution field
    sgs : subgrid scale term
    w : vorticity in physical space for fine grid (including periodic boundaries)
    s : streamfunction in physical space for fine grid (including periodic boundaries) 
    '''
    
    s = fps(nx,ny,dx,dy,k2,-wf)
    w = wave2phy(nx,ny,wf)
   
    kxc = np.fft.fftfreq(nxc,1/nxc)
    kyc = np.fft.fftfreq(nyc,1/nyc)
    kxc = kxc.reshape(nxc,1)
    kyc = kyc.reshape(1,nyc)
    
    k2c = kxc*kxc + kyc*kyc
    k2c[0,0] = 1.0e-12
     
    jf = nonlineardealiased(nx,ny,kx,ky,k2,wf)
    j = wave2phy(nx,ny,jf) # jacobian for fine solution field

    jc = np.zeros((nxc+1,nyc+1)) # coarsened(jacobian field)
    jfc = coarsen(nx,ny,nxc,nyc,jf) # coarsened(jacobian field) in frequency domain
    jc = wave2phy(nxc,nyc,jfc) # coarsened(jacobian field) physical space
       
    wfc = coarsen(nx,ny,nxc,nyc,wf)       
    jcoarsef = nonlineardealiased(nxc,nyc,kxc,kyc,k2c,wfc) # jacobian(coarsened solution field) in frequency domain
    jcoarse = wave2phy(nxc,nyc,jcoarsef) # jacobian(coarsened solution field) physical space
    
    sgs = jc - jcoarse
    
    folder = 'data_'+str(nx)
    if not os.path.exists("../data_spectral/"+folder):
        os.makedirs("../data_spectral/"+folder)
        os.makedirs("../data_spectral/"+folder+"/01_coarsened_jacobian_field")
        os.makedirs("../data_spectral/"+folder+"/02_jacobian_coarsened_field")
        os.makedirs("../data_spectral/"+folder+"/03_subgrid_scale_term")
        os.makedirs("../data_spectral/"+folder+"/04_vorticity")
        os.makedirs("../data_spectral/"+folder+"/05_streamfunction")
    
    filename = "../data_spectral/"+folder+"/01_coarsened_jacobian_field/J_fourier_"+str(int(n/freq))+".csv"
    np.savetxt(filename, jc, delimiter=",")    
    filename = "../data_spectral/"+folder+"/02_jacobian_coarsened_field/J_coarsen_"+str(int(n/freq))+".csv"
    np.savetxt(filename, jcoarse, delimiter=",")
    filename = "../data_spectral/"+folder+"/03_subgrid_scale_term/sgs_"+str(int(n/freq))+".csv"
    np.savetxt(filename, sgs, delimiter=",")
    filename = "../data_spectral/"+folder+"/04_vorticity/w_"+str(int(n/freq))+".csv"
    np.savetxt(filename, w, delimiter=",")
    filename = "../data_spectral/"+folder+"/05_streamfunction/s_"+str(int(n/freq))+".csv"
    np.savetxt(filename, s, delimiter=",")
    
    if n%(50*freq) == 0:
        fig, axs = plt.subplots(1,2,sharey=True,figsize=(9,5))

        cs = axs[0].contourf(w0.T, 120, cmap = 'jet', interpolation='bilinear')
        axs[0].text(0.4, -0.1, '$t = 0.0$', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
        
        cs = axs[1].contourf(w.T, 120, cmap = 'jet', interpolation='bilinear')
        axs[1].text(0.4, -0.1, '$t = '+str(dt*n)+'$', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
        
        fig.tight_layout() 
        fig.subplots_adjust(bottom=0.15)
        
        cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
        fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
            
        filename = "../data_spectral/"+folder+"/field_spectral_"+str(int(n/freq))+".png"
        fig.savefig(filename, bbox_inches = 'tight')

#%%
def export_data(nx,ny,dx,dy,re,n,wf,k2):
    
    s = fps(nx,ny,dx,dy,k2,-wf)
    w = wave2phy(nx,ny,wf)
    
    folder = 'data_'+str(int(re)) + '_' + str(nx) + '_' + str(ny)
        
    if not os.path.exists('./results/'+folder):
        os.makedirs('./results/'+folder)
        
    filename = './results/'+folder+'/results_' + str(int(n))+'.npz'
    np.savez(filename,w=w,s=s)    
    
#%% 
# read input file
l1 = []
with open('input.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nd = np.int64(l1[0][0])
nt = np.int64(l1[1][0])
re = np.float64(l1[2][0])
dt = np.float64(l1[3][0])
ns = np.int64(l1[4][0])
isolver = np.int64(l1[5][0])
isc = np.int64(l1[6][0])
ich = np.int64(l1[7][0])
ipr = np.int64(l1[8][0])
ndc = np.int64(l1[9][0])
ichkp = np.int64(l1[10][0])
istart = np.int64(l1[11][0])
ndf = np.int64(l1[12][0])
k0 = np.float64(l1[13][0])

freq = int(nt/ns)

if (ich != 19):
    print("Check input.txt file")

# assign parameters
nx = nd
ny = nd

nxc = ndc
nyc = ndc

nxf = ndf
nyf = ndf

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

dxf = lx/np.float64(nxf)
dyf = ly/np.float64(nyf)

ifile = 0
time = ichkp*freq*istart*dt
folder = 'data_'+str(nx)

#%%
# set the initial condition based on the problem selected
if (ipr == 1):
    w0 = tgv_ic(nx,ny) # taylor-green vortex problem
elif (ipr == 2):
    w0 = vm_ic(nx,ny) # vortex-merger problem
elif (ipr == 3):
    w0 = decay_ic(nx,ny,dx,dy,k0) # decaying homegeneous isotropic turbulence problem

#%%  
if ichkp == 0:
    w = np.copy(w0)
elif ichkp == 1:
    print(istart)
    file_input = "../data_spectral/"+folder+"/04_vorticity/w_"+str(istart)+".csv"
    w = np.genfromtxt(file_input, delimiter=',')
    
#%%
# compute frequencies, vorticity field in frequency domain
kx = np.fft.fftfreq(nx,1/nx)
ky = np.fft.fftfreq(ny,1/ny)

kx = kx.reshape(nx,1)
ky = ky.reshape(1,ny)
    
data = np.empty((nx,ny), dtype='complex128')

data = np.vectorize(complex)(w[0:nx,0:ny],0.0)

a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')

wnf = fft_object(data) # fourier space forward


#%%
# initialize variables for time integration
a1, a2, a3 = 8.0/15.0, 2.0/15.0, 1.0/3.0
g1, g2, g3 = 8.0/15.0, 5.0/12.0, 3.0/4.0
r2, r3 = -17.0/60.0, -5.0/12.0

k2 = kx*kx + ky*ky
k2[0,0] = 1.0e-12

z = 0.5*dt*k2*(1/re)
d1 = a1*z
d2 = a2*z
d3 = a3*z

kxf = np.fft.fftfreq(nxf,1/nxf)
kyf = np.fft.fftfreq(nyf,1/nyf)

kxf = kxf.reshape(nxf,1)
kyf = kyf.reshape(1,nyf)

k2f = kxf*kxf + kyf*kyf
k2f[0,0] = 1.0e-12

export_data(nx,ny,dx,dy,re,0,wnf,k2)

w1f = np.empty((nx,ny), dtype='complex128')
w2f = np.empty((nx,ny), dtype='complex128')

#%%
# load the trained model and test
gan_d = PIESRGAN(height_lr=32, width_lr=32, upscaling_factor=8,training_mode=False,gen_lr=1e-4, dis_lr=1e-5,loss_weights={'percept':1e-1,'gen':1e-3, 'pixel':5.0, 'phy':0.001, 'ens':0.1})

# Load weights - this assumes you've already trained the network!
# Replace filename with the location of your trained weights
gan_d.load_weights('post_generator_8X_epoch_100000.h5') 

#%%
clock_time_init = tm.time()
# time integration using hybrid third-order Runge-Kutta implicit Crank-Nicolson scheme
# refer to Orlandi: Fluid flow phenomenon
for n in range(int(ichkp*istart*freq)+1,nt+1):
    time = time + dt
    # 1st step
    #jnf = nonlineardealiased(nx,ny,kx,ky,k2,wnf)    
    jnf = nonlineardealiased_ad(nx,ny,kx,ky,k2,nxf,nyf,kxf,kyf,k2f,wnf,gan_d)    
    w1f[:,:] = ((1.0 - d1)/(1.0 + d1))*wnf[:,:] + (g1*dt*jnf[:,:])/(1.0 + d1)
    w1f[0,0] = 0.0
    
    # 2nd step
    #j1f = nonlineardealiased(nx,ny,kx,ky,k2,w1f)
    j1f = nonlineardealiased_ad(nx,ny,kx,ky,k2,nxf,nyf,kxf,kyf,k2f,w1f,gan_d)    
    w2f[:,:] = ((1.0 - d2)/(1.0 + d2))*w1f[:,:] + (r2*dt*jnf[:,:]+ g2*dt*j1f[:,:])/(1.0 + d2)
    w2f[0,0] = 0.0
    
    # 3rd step
    #j2f = nonlineardealiased(nx,ny,kx,ky,k2,w2f)
    j2f = nonlineardealiased_ad(nx,ny,kx,ky,k2,nxf,nyf,kxf,kyf,k2f,w2f,gan_d)
    wnf[:,:] = ((1.0 - d3)/(1.0 + d3))*w2f[:,:] + (r3*dt*j1f[:,:] + g3*dt*j2f[:,:])/(1.0 + d3)
    wnf[0,0] = 0.0
    
    if (n%freq == 0):
        #write_data(nx,ny,dx,dy,kx,ky,k2,nxc,nyc,dxc,dyc,wnf,w0,n,freq,dt)
        export_data(nx,ny,dx,dy,re,int(n/freq),wnf,k2)
        print(n, " ", time, " ",wnf.shape[0], " ", wnf.shape[1])
    
w = wave2phy(nx,ny,wnf) # final vorticity field in physical space            

total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)  

#%%
# compute the exact, initial and final energy spectrum for DHIT problem
if (ipr == 3):
    en, n = energy_spectrum(nx,ny,w)
    en0, n = energy_spectrum(nx,ny,w0)
    k = np.linspace(1,n,n)
    
    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    ese = c*(k**4)*np.exp(-(k/k0)**2)
    
    folder = 'data_'+str(nx)
    
    #np.savetxt("../data_spectral/"+folder+"energy_spectral_"+str(nd)+"_"+str(int(re))+".csv", en, delimiter=",")

#%%
# contour plot for initial and final vorticity
fig, axs = plt.subplots(1,2,sharey=True,figsize=(9,5))

cs = axs[0].contourf(w0.T, 120, cmap = 'jet', interpolation='bilinear')
axs[0].text(0.4, -0.1, '$t = 0.0$', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')

cs = axs[1].contourf(w.T, 120, cmap = 'jet', interpolation='bilinear')
axs[1].text(0.4, -0.1, '$t = '+str(dt*nt)+'$', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

fig.savefig("field_spectral_ad.png", bbox_inches = 'tight')


#%%
# energy spectrum plot for DHIT problem
if (ipr == 3):
    #en_a = np.loadtxt("energy_arakawa_"+str(nd)+"_"+str(int(re))+".csv") 
    fig, ax = plt.subplots()
    fig.set_size_inches(7,5)
    
    line = 100*k**(-3.0)
    
    ax.loglog(k,ese[:],'g', lw = 2, label='Exact')
    ax.loglog(k,en0[1:],'r', ls = '--', lw = 2, label='$t = 0.0$')
    ax.loglog(k,en[1:], 'b', lw = 2, label = '$t = '+str(dt*nt)+'$')
    #ax.loglog(k,en_a[1:], 'y', lw = 2, label = '$t = '+str(dt*nt)+'$')
    ax.loglog(k,line, 'k--', lw = 2, label = '$k^{-3}$')
    
    plt.xlabel('$K$')
    plt.ylabel('$E(K)$')
    plt.legend(loc=0)
    plt.ylim(1e-9,1e0)
    fig.savefig('es_spectral.png', bbox_inches = 'tight', pad_inches = 0)












