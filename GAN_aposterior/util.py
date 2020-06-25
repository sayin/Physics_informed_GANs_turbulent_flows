import tensorflow as tf
from keras import backend as K
from keras.utils import conv_utils
from keras.layers.convolutional import UpSampling3D
from keras.engine import InputSpec
from tensorlayer.layers import *
import h5py as h5
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import datetime
import pyfftw
from scipy.interpolate import griddata, interp2d



#%%
class UpSampling3D(Layer):
   
    def __init__(self, size=(2, 2, 2), **kwargs):
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.input_spec = InputSpec(ndim=5)
        super(UpSampling3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        dim1 = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        dim2 = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        dim3 = self.size[2] * input_shape[3] if input_shape[3] is not None else None
        return (input_shape[0],
                dim1,
                dim2,
                dim3,
                input_shape[4])

    def call(self, inputs):
        return K.resize_volumes(inputs,
                                self.size[0], self.size[1], self.size[2],
                                self.data_format)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(UpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
def subPixelConv3d(net, height_hr, width_hr, depth_hr, stepsToEnd, n_out_channel):
    """ pixle-shuffling for 3d data"""
    i = net
    r = 2
    a, b, z, c = int(height_hr/ (2 * stepsToEnd)), int(width_hr / (2 * stepsToEnd)), int(
        depth_hr / (2 * stepsToEnd)), tf.shape(i)[4]
    batchsize = tf.shape(i)[0]   # Handling Dimension(None) type for undefined batch dim
    xs = tf.split(i, r, 4)       # b*h*w*d*r*r*r
    xr = tf.concat(xs, 3)        # b*h*w*(r*d)*r*r
    xss = tf.split(xr, r, 4)     # b*h*w*(r*d)*r*r
    xrr = tf.concat(xss, 2)      # b*h*(r*w)*(r*d)*r
    x = tf.reshape(xrr, (batchsize, r * a, r * b, r * z, n_out_channel))  # b*(r*h)*(r*w)*(r*d)*n_out 

    return x

#%%
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
def en_sp(box1, box2, box3,out):
    
    folder = './results/'
    nxc, nyc = box1.shape[0], box1.shape[1]
    nx, ny = box2.shape[0], box2.shape[1]    
    
    x = np.linspace(0,2.0*np.pi,nx+1)
    y =  np.linspace(0,2.0*np.pi,ny+1)      

    
    xc = np.linspace(0,2.0*np.pi,nxc+1)
    yc =  np.linspace(0,2.0*np.pi,nyc+1)   

    
    Ti1 = interp2d(xc[:-1], yc[:-1], box1,kind='linear')
    Ti2 = interp2d(xc[:-1], yc[:-1], box1,kind='cubic')
    Til = Ti1(x[:-1], y[:-1])
    Tib = Ti2(x[:-1], y[:-1])
    #plt.contourf(X[:-1,:-1], Y[:-1,:-1], Ti)
    
    enc, nc = energy_spectrum(nxc,nyc,box1)
    ensr, nsr = energy_spectrum(nx,ny, box2)
    enf, nf = energy_spectrum(nx,ny,box3)
    enl, nl = energy_spectrum(nx,ny,Til)
    enb, nb = energy_spectrum(nx,ny,Tib)
    
    kc = np.linspace(1,nc,nc)
    kf = np.linspace(1,nf,nf)
    kb = np.linspace(1,nb,nb)
    
    kl = kf[5:128]
    line = 200*kl**(-3.0) #*kl**3   
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.loglog(kf,enf[1:], 'g', lw = 2, label = 'DNS')
    ax.loglog(kc,enc[1:], 'b--', lw = 2, label = 'FDNS')
    ax.loglog(kf,ensr[1:], 'r', lw = 2, label = 'ESRGAN')  
    ax.loglog(kb,enl[1:], 'y', lw = 2, label = 'Linear')
    ax.loglog(kb,enb[1:], 'm', lw = 2, label = 'Cubic')
    
    ax.loglog(kl,line, 'k-.', lw = 2)
    
    plt.xlabel('$K$')
    plt.ylabel('$E(K)$')
    plt.legend(loc=0)
    plt.xlim(1,256)
    #plt.ylim(1e-8,1e-0)
    plt.text(kl[10], line[10]+0.2, '$k^-3$', color='k')
    plt.savefig(folder+'en_32_256_'+out+'.pdf',bbox_inches='tight')
#    plt.show()
    
    
def s_generator(box1,box2,box3,out):

    folder = './results/'

    dim=(1, 3)
    ax = plt.figure(figsize=(15, 5))
    vmin=np.min(box3) + 0.15
    vmax=np.max(box3) - 0.15
    
    ax=plt.subplot(dim[0], dim[1], 1)
    img=plt.imshow(np.flipud(box1),cmap='jet',vmin=vmin,vmax=vmax)
    ax.set_title('FDNS ('+str(32)+r'$\times$'+str(32)+')')
    plt.axis('off')
    plt.colorbar(img,shrink=0.8, ticks=np.linspace(vmin,vmax,11))
    
    ax=plt.subplot(dim[0], dim[1], 2)
    img=plt.imshow(np.flipud(box2),cmap='jet',vmin=vmin,vmax=vmax)
    ax.set_title(r'Reconstructed ('+str(256)+r'$\times$'+str(256)+')')
    plt.axis('off')
    plt.colorbar(img,shrink=0.8, ticks=np.linspace(vmin,vmax,11))
    
    ax=plt.subplot(dim[0], dim[1], 3)
    img=plt.imshow(np.flipud(box3),cmap='jet',vmin=vmin,vmax=vmax)
    ax.set_title(r'DNS ('+str(256)+r'$\times$'+str(256)+')')
    plt.axis('off')    
    plt.colorbar(img,shrink=0.8, ticks=np.linspace(vmin,vmax,11))   
    plt.savefig(folder+'s_32_256_'+out+'.pdf')
    plt.tight_layout()
#    plt.show()
    
def w_generator(box1,box2,box3,out):
    folder = './results/'
    
    dim=(1, 3)
    ax = plt.figure(figsize=(15, 5))
    vmin=np.min(box3) + 5
    vmax=np.max(box3) - 5
    
    ax=plt.subplot(dim[0], dim[1], 1)
    img=plt.imshow(np.flipud(box1),cmap='jet',vmin=vmin,vmax=vmax)
#    img=plt.contourf(box1,cmap='jet',vmin=-45,vmax=80)
    ax.set_title('FDNS ('+str(32)+r'$\times$'+str(32)+')')
    plt.axis('off')
    plt.colorbar(img, shrink=0.8, ticks=np.linspace(vmin,vmax,11))
    
    ax=plt.subplot(dim[0], dim[1], 2)
    img=plt.imshow(np.flipud(box2),cmap='jet',vmin=vmin,vmax=vmax)
#    img=plt.contourf(box2,cmap='jet',vmin=-45,vmax=80)
    ax.set_title(r'Reconstructed ('+str(256)+r'$\times$'+str(256)+')')
    plt.axis('off')
    plt.colorbar(img,shrink=0.8, ticks=np.linspace(vmin,vmax,11))
    
    ax=plt.subplot(dim[0], dim[1], 3)
    img=plt.imshow(np.flipud(box3),cmap='jet',vmin=vmin,vmax=vmax)
#    img=plt.contourf(box3,cmap='jet',vmin=-45,vmax=60)
    ax.set_title(r'DNS ('+str(256)+r'$\times$'+str(256)+')')
    plt.axis('off')    
    plt.colorbar(img,shrink=0.8, ticks=np.linspace(vmin,vmax,11))   
    plt.savefig(folder+'w_32_256_'+out+'.pdf')
    plt.tight_layout()    
#    plt.show()

def loss_plot(gen_loss, dis_loss):
    folder = './results/'

    plt.figure()
    plt.semilogy(gen_loss[:,0], label='total loss')
    plt.xlabel('Epochs')
    plt.ylabel('Total loss')
    plt.legend()
    plt.savefig(folder+'total_loss.pdf')
    
    plt.figure()
    plt.semilogy(gen_loss[:,1], label='grad loss')
    plt.xlabel('Epochs')
    plt.ylabel('Grad loss')
    plt.legend()
    plt.savefig(folder+'grad_loss.pdf')
    
    plt.figure()
    plt.semilogy(gen_loss[:,2], label='Adversarial loss')
    plt.xlabel('Epochs')
    plt.ylabel('Adversarial loss')
    plt.legend()    
    plt.savefig(folder+'adv_loss.pdf')
    
    plt.figure()
    plt.semilogy(gen_loss[:,3], label='MSE loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.legend()    
    plt.savefig(folder+'mse_loss.pdf')
      
    plt.figure()
    plt.semilogy(gen_loss[:,4], label='Physics loss')
    plt.xlabel('Epochs')
    plt.ylabel('Physics loss')
    plt.legend()  
    plt.savefig(folder+'phy_loss.pdf')




   