#! /usr/bin/python
# -*- coding: utf-8 -*-
# ! /usr/bin/python
import os          # enables interactions with the operating system
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import sys
import pickle      # object-->byte system
import datetime    # manipulating dates and times
import numpy as np

# Import keras + tensorflow without the "Using XX Backend" message
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import gc
import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import Conv3dLayer, LambdaLayer
from keras.datasets import mnist
from keras.utils import conv_utils
from keras.engine import InputSpec
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Add, Concatenate, Multiply, add
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, Conv3D,ZeroPadding3D
from keras.layers import UpSampling2D, Lambda, Dropout
from keras.optimizers import Adam, RMSprop
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K           #campatability
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
sys.stderr = stderr
from util import *

import h5py as h5
import matplotlib.pyplot as plt
'''Use Horovod in case of multi nodes parallelizations'''
#import horovod.keras as hvd
plt.switch_backend('agg')


#%%
def lrelu1(x):
    return tf.maximum(x, 0.25 * x)
def lrelu2(x):
    return tf.maximum(x, 0.3 * x)

def grad0(matrix): 
    return np.gradient(matrix,axis=0)

def grad1(matrix): 
    dx = 2.0*np.pi/256.0
    return np.gradient(matrix,dx,axis=1)

def grad2(matrix):
    dy = 2.0*np.pi/256.0
    return np.gradient(matrix,dy,axis=2)

def grad3(matrix): 
    return np.gradient(matrix,axis=3)

def vort(m1,m2):
    return np.multiply(1,np.add(m1,m2))

def continuity(m1, m2):
    return np.add(m1, m2)

class PIESRGAN():
    """
    Implementation of PIESRGAN as described in the paper
    """

    def __init__(self,
                 height_lr=16, width_lr=16, channels=2,
                 upscaling_factor=16,
                 gen_lr=1e-4, dis_lr=1e-7,
                 # VGG scaled with 1/12.75 as in paper
                 loss_weights={'percept':1e-1,'gen':5e-5, 'pixel':5e0, 'phy':0.0125, 'ens':0.0125},
                 training_mode=True,
                 refer_model=None,
                 ):
        """
        :param int height_lr: Height of low-resolution DNS data
        :param int width_lr: Width of low-resolution DNS data
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """
        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr
#        self.depth_lr=depth_lr
        
        # High-resolution image dimensions are identical to those of the LR, removed the upsampling block!
        if upscaling_factor not in [2, 4, 8, 16]:
            raise ValueError('Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)
        
        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)
        
        # Low-resolution and high-resolution shapes
        """ DNS-Data only has one channel, when only using PS field, when using u,v,w,ps, change to 4 channels """
#        self.shape_lr = (self.height_lr, self.width_lr, 1)
#        self.shape_hr = (self.height_hr, self.width_hr,1)
#        self.batch_shape_lr = (None,self.height_lr, self.width_lr, 1)
#        self.batch_shape_hr = (None,self.height_hr, self.width_hr, 1)
        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        
        # Scaling of losses
        self.loss_weights = loss_weights
        
        # Gan setup settings
        self.gan_loss = 'mse'
        self.dis_loss = 'binary_crossentropy'
        
        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)
        #self.refer_model = refer_model
        
        # If training, build rest of GAN network
        if training_mode:
            self.discriminator = self.build_discriminator()
            self.RaGAN = self.build_RaGAN()
            self.piesrgan = self.build_piesrgan()
#            self.compile_discriminator(self.RaGAN)
#            self.compile_piesrgan(self.piesrgan)
            

            
    def SubpixelConv2D(self, name, scale=2):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space
        :param scale: upsampling scale compared to input_shape. Default=2
        :return:
        """

        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape
    
        def subpixel(x):
            return tf.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)
                
    def build_generator(self, ):
        """
         Build the generator network according to description in the paper.
         First define seperate blocks then assembly them together
        :return: the compiled model
        """
#        w_init = tf.random_normal_initializer(stddev=0.02)
#        height_hr=self.height_hr
#        width_hr=self.width_hr
#        depth_hr=self.depth_hr

        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  #added x3, which ESRGAN didn't include

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            """here: assumed beta=0.2"""
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            """here: assumed beta=0.2 as well"""
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
            x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x
        
        """----------------Assembly the generator-----------------"""
        # Input low resolution image
        lr_input = Input(shape=(None, None, self.channels))

        # Pre-residual
        x_start = Conv2D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])
        
        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)  
        if self.upscaling_factor > 8:
            x = upsample(x, 4) 
        if self.upscaling_factor > 16:
            x = upsample(x, 5)     
#        x = Conv3D(512,kernel_size=3, strides=1, padding='same',activation=lrelu1)(x)
#        x = Conv3D(512,kernel_size=3, strides=1, padding='same',activation=lrelu1)(x)
        #Final 2 convolutional layers
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        #Uncomment this if using multi GPU model
        #model=multi_gpu_model(model,gpus=2,cpu_merge=True)
        # model.summary()
        return model
        
        
        
    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters * 2)
        x = conv2d_block(x, filters * 2, strides=2)
        x = conv2d_block(x, filters * 4)
        x = conv2d_block(x, filters * 4, strides=2)
        x = conv2d_block(x, filters * 8)
        x = conv2d_block(x, filters * 8, strides=2)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x)
        return model

    def build_piesrgan(self):
        """Create the combined PIESRGAN network"""
        def comput_loss(x):
            img_hr, generated_hr = x             

############### Perceptual Loss #################### 
            
            # Compute the Perceptual loss ###based on GRADIENT-field MSE
            grad_hr_1 = tf.py_func(grad1,[img_hr],tf.float32)
            grad_hr_2 = tf.py_func(grad2,[img_hr],tf.float32)
#            grad_hr_3 = tf.py_func(grad3,[img_hr],tf.float32)            
            
            grad_sr_1 = tf.py_func(grad1,[generated_hr],tf.float32)
            grad_sr_2 = tf.py_func(grad2,[generated_hr],tf.float32)  

            grad_loss = K.mean( 
                tf.losses.mean_squared_error(grad_hr_1,grad_sr_1)+
                tf.losses.mean_squared_error(grad_hr_2,grad_sr_2))
            
############### Physics Loss #################### 
            
            dx_sr =  tf.py_func(grad1,[generated_hr[:,:,:,0]],tf.float32)
            dy_sr =  tf.py_func(grad1,[generated_hr[:,:,:,0]],tf.float32)
            
            d2x_sr  = tf.py_func(grad1,[dx_sr],tf.float32)
            d2y_sr  = tf.py_func(grad2,[dy_sr],tf.float32)

            w_sr = tf.py_func(vort,[d2x_sr,d2y_sr],tf.float32)
            
            phy_er= tf.math.add(w_sr,generated_hr[:,:,:,1])
            
            phy_loss =  tf.math.reduce_mean(phy_er)

############### Enstropy Loss ####################     
            en_sr = tf.math.square(generated_hr[:,:,:,1])
            en_hr = tf.math.square(img_hr[:,:,:,1])
            
            ens_loss= tf.losses.mean_squared_error(en_hr,en_sr)           
            
############### RaGAN Loss ####################              
            # Compute the RaGAN loss
            fake_logit, real_logit = self.RaGAN([img_hr,generated_hr])
#            gen_loss = K.mean(
#                K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
#                K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))
            epsilon = 0.000001
            gen_loss =-(K.mean(K.log(K.sigmoid(fake_logit)+epsilon))+K.mean(K.log(1-K.sigmoid(real_logit)+epsilon)))
            
############### pixel Loss ####################            
            # Compute the pixel_loss with L1 loss
            pixel_loss = tf.losses.mean_squared_error(generated_hr, img_hr)            
            
            return [ grad_loss, gen_loss, pixel_loss, phy_loss, ens_loss]

        # Input LR images
        img_lr = Input(shape=self.shape_lr)
        img_hr = Input(shape=self.shape_hr)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        
        # In the combined model we only train the generator
        self.discriminator.trainable = False
        self.RaGAN.trainable = False
              
        # Output tensors to a Model must be the output of a Keras `Layer`
        total_loss = Lambda(comput_loss, name='comput_loss')([img_hr, generated_hr])
        grad_loss = Lambda(lambda x: self.loss_weights['percept'] * x, name='grad_loss')(total_loss[0])
        gen_loss = Lambda(lambda x: self.loss_weights['gen'] * x, name='gen_loss')(total_loss[1])
        pixel_loss = Lambda(lambda x: self.loss_weights['pixel'] * x, name='pixel_loss')(total_loss[2])
        phy_loss = Lambda(lambda x:  self.loss_weights['phy'] *x, name='phy_loss')(total_loss[3])
        ens_loss = Lambda(lambda x:  self.loss_weights['ens'] *x, name='ens_loss')(total_loss[4])
#        loss = Lambda(lambda x: x[0]+x[1]+x[2]+0.0*x[3], name='total_loss')(total_loss)
       
        # Create model
        model = Model(inputs=[img_lr, img_hr], outputs=[grad_loss, gen_loss, pixel_loss, phy_loss, ens_loss])
        
        # Add the loss of model and compile
#        model.add_loss(loss)
        model.add_loss(grad_loss)
        model.add_loss(gen_loss)
        model.add_loss(pixel_loss)
        model.add_loss(phy_loss)
        model.add_loss(ens_loss)
        model.compile(optimizer=Adam(self.gen_lr),loss_weights=[grad_loss, gen_loss, pixel_loss, phy_loss, ens_loss])

        # Create metrics of PIESRGAN
        model.metrics_names.append('grad_loss')
        model.metrics_tensors.append(grad_loss)
        model.metrics_names.append('gen_loss')
        model.metrics_tensors.append(gen_loss)
        model.metrics_names.append('pixel_loss')
        model.metrics_tensors.append(pixel_loss) 
        model.metrics_names.append('phy_loss')
        model.metrics_tensors.append(phy_loss) 
        model.metrics_names.append('ens_loss')
        model.metrics_tensors.append(ens_loss) 
#        model.summary()
        return model
        
    def build_RaGAN(self):
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        def comput_loss(x):
            real, fake = x
            fake_logit = fake - K.mean(real)
#            fake_logit = K.sigmoid(fake - K.mean(real))
            real_logit = real - K.mean(fake)
#            real_logit = K.sigmoid(real - K.mean(fake))
            return [fake_logit, real_logit]

        # Input HR images
        imgs_hr = Input(self.shape_hr)
        generated_hr = Input(self.shape_hr)
        
        # Create a high resolution image from the low resolution one
        real_discriminator_logits = self.discriminator(imgs_hr)
        fake_discriminator_logits = self.discriminator(generated_hr)

        total_loss = Lambda(comput_loss, name='comput_loss')([real_discriminator_logits, fake_discriminator_logits])
        # Output tensors to a Model must be the output of a Keras `Layer`
        fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
        real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])

#        dis_loss = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
#                          K.binary_crossentropy(K.ones_like(real_logit), real_logit))
        
        epsilon = 0.000001
        dis_loss = -(K.mean(K.log(K.sigmoid(real_logit)+epsilon))+K.mean(K.log(1-K.sigmoid(fake_logit)+epsilon)))
        # dis_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit) +
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_likes(real_logit), logits=real_logit))
        # dis_loss = K.mean(- (real_logit - fake_logit)) + 10 * K.mean((grad_norms - 1) ** 2)

        model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit])

        model.add_loss(dis_loss)
        model.compile(optimizer=Adam(self.dis_lr))

        model.metrics_names.append('dis_loss')
        model.metrics_tensors.append(dis_loss)
        
#        model.summary()
        return model

  
    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        def pixel_loss(y_true, y_pred):
             loss1=tf.losses.mean_squared_error(y_true,y_pred)
             loss2=tf.losses.absolute_difference(y_true,y_pred)
 
             return 1*loss1+0.005*loss2
         
        def mae_loss(y_true, y_pred):
            loss=tf.losses.absolute_difference(y_true,y_pred)
            return loss*0.01
        def grad_loss(y_true,y_pred):
            grad_hr_1 = tf.py_func(grad1,[y_true],tf.float32)
            grad_hr_2 = tf.py_func(grad2,[y_true],tf.float32)
#            grad_hr_3 = tf.py_func(grad3,[y_true],tf.float32)
            grad_sr_1 = tf.py_func(grad1,[y_pred],tf.float32)
            grad_sr_2 = tf.py_func(grad2,[y_pred],tf.float32)
#            grad_sr_3 = tf.py_func(grad3,[y_pred],tf.float32)
            grad_loss = K.mean( 
                tf.losses.mean_squared_error(grad_hr_1,grad_sr_1)+
                tf.losses.mean_squared_error(grad_hr_2,grad_sr_2))
#                tf.losses.mean_squared_error(grad_hr_3,grad_sr_3))
            return grad_loss        

        model.compile(
            loss=pixel_loss,
            optimizer=Adam(self.gen_lr, 0.9,0.999),
            metrics=[self.PSNR]
        )

    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.dis_lr, 0.9, 0.999),
            metrics=['accuracy']
        )

    def compile_piesrgan(self, model):
        """Compile the PIESRGAN with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.gen_lr, 0.9, 0.999)
        )

    def PSNR(self, y_true, y_pred):
        """
        Peek Signal to Noise Ratio
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    def save_weights(self, filepath, e=None):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}generator_{}X_epoch_{}.h5".format(filepath, self.upscaling_factor, e))
        self.discriminator.save_weights("{}discriminator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)                 
            
            
    def train_generator(self,
        epochs, max_it, batch_size,
        dataname='pre_gen',
        lr_train = None,
        hr_train = None,
        lr_test  = None,
        hr_test  = None,
        steps_per_epoch=1,
        steps_per_validation=4,
        log_weight_path='./data/weights/pre/'
        ):
        """Trains the generator part of the network with MSE loss"""

#        self.gen_lr = 3.2e-5
        for step in range(epochs // 10):

            self.compile_generator(self.generator)             
            callbacks = []
 #             Callback: save weights after each epoch
            modelcheckpoint = ModelCheckpoint(
                    os.path.join(log_weight_path, dataname + 
                                 '_{}X.h5'.format(self.upscaling_factor)),
                    monitor='PSNR',
                    save_best_only=True,
                    save_weights_only=True
            )
            callbacks.append(modelcheckpoint)
            csv_logger = CSVLogger("model_history_log.csv", append=True)

            # Fit the model  --int(16*16*31/batch_size)
            for idx in range (0,max_it+1):
                
                print("\nIter {}/{}".format(idx, max_it))
                
                if idx%(max_it/5)==0:
                    self.generator.save_weights("{}pre_gen_idx{}.h5".format(log_weight_path,idx))
#
                rand_nums = np.random.randint(0, hr_train.shape[0],size=batch_size)
                train_batch_hr = hr_train[rand_nums]
                train_batch_lr = lr_train[rand_nums]  
                
                rand_nums = np.random.randint(0, hr_test.shape[0],size=batch_size)
                test_batch_hr = hr_test[rand_nums]
                test_batch_lr = lr_test[rand_nums]   
                test_loader = test_batch_lr, test_batch_hr

                history = self.generator.fit(
                 train_batch_lr,train_batch_hr,
                 steps_per_epoch=4,
                 epochs=10,
                 validation_data=test_loader,
                 validation_steps=steps_per_validation,
                 callbacks=[csv_logger],
                )

            self.generator.save( log_weight_path+'pre_gen_final_model.h5')
            self.gen_lr /= 1.149
            print(step, self.gen_lr)   
            
        return history            
            
    def train_piesrgan(self,
        epochs, batch_size,
        dataname='post_',
        lr_train = None,
        hr_train = None,
        lr_rf  = None,
        hr_rf  = None, 
        lr_test  = None,
        snc = None, wnc = None,        
        steps_per_validation=10,
        first_epoch=0,
        print_frequency=2,
        log_weight_frequency=2,
        log_img_frequency=2,
        log_weight_path='./data/weights/post/',
        ):
        # Each epoch == "update iteration" as defined in the paper
#        print_losses = {"G": [], "D": []}
        
        gen_loss = np.empty((0,6))
        dis_loss = np.empty((0,2))
#        adv_loss = np.empty((0,2))
#        gen_loss = np.empty()

        # Random images to go through
        #idxs = np.random.randint(0, len(loader), epochs)
#        print(">> Pretraining Discriminator")
#
#        for idx in range(0,50):
#            
#            rand_nums = np.random.randint(0, hr_train.shape[0],size=batch_size)
#            imgs_hr = hr_train[rand_nums]
#            imgs_lr = lr_train[rand_nums]
#            
#            generated_hr = self.generator.predict(imgs_lr,steps=1) 
#            discriminator_loss = self.RaGAN.train_on_batch([imgs_hr, generated_hr], None)
            
        print(">> Training Started")
            
        # Loop through epochs / iterations
        for epoch in range(first_epoch, epochs + first_epoch):
            # Start epoch time
#            if epoch % print_frequency == 0:
#                start_epoch = datetime.datetime.now()

            rand_nums = np.random.randint(0, hr_train.shape[0],size=batch_size)
            imgs_hr = hr_train[rand_nums]
            imgs_lr = lr_train[rand_nums]  
            generated_hr = self.generator.predict(imgs_lr,steps=1)             

            
#            for step in range(10):
            # SRGAN's loss (don't use them)
#            real_loss = self.discriminator.train_on_batch(imgs_hr, real)
#            fake_loss = self.discriminator.train_on_batch(generated_hr, fake)
#            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
                #print("step: ",step+1)
                
            # Train Relativistic Discriminator
#            fake_logit, real_logit = self.RaGAN([imgs_hr,generated_hr])
#            dis_adv_loss = [K.mean(fake_logit), K.mean(real_logit)]
            discriminator_loss = self.RaGAN.train_on_batch([imgs_hr, generated_hr], None)    
            # Train generator
            # features_hr = self.vgg.predict(self.preprocess_vgg(imgs_hr))
            generator_loss = self.piesrgan.train_on_batch([imgs_lr, imgs_hr], None)

            # Callbacks
            # Save losses
#            print_losses['G'].append(generator_loss)
#            print_losses['D'].append(discriminator_loss)            
            gen_loss = np.vstack((gen_loss, generator_loss))
            dis_loss = np.vstack((dis_loss, discriminator_loss))
#            adv_loss = np.vstack((adv_loss, dis_adv_loss))
            
                # Show the progress
            if epoch % print_frequency == 0:
#            g_avg_loss = np.array(print_losses['G']).mean(axis=0)
#            d_avg_loss = np.array(print_losses['D']).mean(axis=0)
            #print(self.piesrgan.metrics_names)
            #print(g_avg_loss)
#            print(self.piesrgan.metrics_names, g_avg_loss)
#            print(self.RaGAN.metrics_names, d_avg_loss)
                print("\nEpoch {}/{}".format(epoch, epochs + first_epoch))
             
                print('Generator',
                  self.piesrgan.metrics_names[0], '=', '%.4f' %gen_loss[epoch,0],                           
                  self.piesrgan.metrics_names[1], '=', '%.4f' %gen_loss[epoch,1],
                  self.piesrgan.metrics_names[2], '=', '%.4f' %gen_loss[epoch,2],
                  self.piesrgan.metrics_names[3], '=', '%.4f' %gen_loss[epoch,3],
                  self.piesrgan.metrics_names[4], '=', '%.7f' %gen_loss[epoch,4],
                  self.piesrgan.metrics_names[5], '=', '%.7f' %gen_loss[epoch,5])
            
                print('Discriminator',
                  self.RaGAN.metrics_names[0], '=', '%.8f' %dis_loss[epoch,0],                           
                  self.RaGAN.metrics_names[1], '=', '%.8f' %dis_loss[epoch,1])            
             
#            print_losses = {"G": [], "D": []}
            # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:
            # Save the network weights
                print(">> Saving the network weights")
                self.save_weights(os.path.join(log_weight_path, dataname), epoch)                         
                
            if log_weight_frequency and epoch % log_img_frequency == 0:
                # Save the network weights
                print(">> Saving the images")
#                self.save_weights(os.path.join(log_weight_path, dataname), epoch)                         
                
                img_sr = self.generator.predict(lr_test,steps=1)        
                 
        #        print(img_sr.shape)
                sn_sr = img_sr[0,:,:,0]
                wn_sr = img_sr[0,:,:,1]
                
                s_sr = sn_sr*snc[0] + snc[1]
                w_sr = wn_sr*wnc[0] + wnc[1]
                
                #create image slice for visualization
                img_shr = hr_rf[:,:,0]
                img_whr = hr_rf[:,:,1]
                
                img_slr = lr_rf[:,:,0]
                img_wlr = lr_rf[:,:,1]
                
                print(">> Ploting test images")
                en_sp(img_wlr,w_sr,img_whr,'epoch_'+str(epoch))
                w_generator(img_wlr,w_sr,img_whr,'epoch_'+str(epoch))        
                s_generator(img_slr,s_sr,img_shr,'epoch_'+str(epoch)) 
                
        return gen_loss, dis_loss
                
    def test(self,            
        refer_model=None,
        batch_size=1,
        lr_rf    = None,
        lr_test  = None,
        hr_rf  = None, 
        snc = None,
        wnc = None,           
        datapath_test=None,
        output_name=None
        ):
        """Trains the generator part of the network"""
        
        img_sr = self.generator.predict(lr_test,steps=1)        
         
#        print(img_sr.shape)
        sn_sr = img_sr[0,:,:,0]
        wn_sr = img_sr[0,:,:,1]
        
        s_sr = sn_sr*snc[0] + snc[1]
        w_sr = wn_sr*wnc[0] + wnc[1]
        
        #create image slice for visualization
        img_shr = hr_rf[:,:,0]
        img_whr = hr_rf[:,:,1]
        
        img_slr = lr_rf[:,:,0]
        img_wlr = lr_rf[:,:,1]
        
        print(">> Ploting test images")
        en_sp(img_wlr,w_sr,img_whr,'epoch_'+output_name)
        s_generator(img_slr,s_sr,img_shr,output_name)          
        w_generator(img_wlr,w_sr,img_whr,output_name)        

        return s_sr, w_sr


#here starts python execution commands
# Run the PIESRGAN network
if __name__ == '__main__':
    
    nf = 800
    nx = 256
    ny = 256  
    nxc = 32
    nyc = nxc
    
    path = './data/2d_turb_'+str(nxc)+'_'+str(nx)+'_'+str(nf)+'/'
    pathout = 'output/'
    
    data_  = np.load(path+'data_'+ str(nxc)+'_'+ str(nx)+'_'+str(nf)+'.npz')
    
    data_hr = data_['data_hr']
    data_lr = data_['data_lr']
    
    nf_train = 600
    nf_val   = 750 
    
    s_hr = data_hr[:,:nx,:ny,0] 
    w_hr = data_hr[:,:nx,:ny,1]
    
    s_lr = data_lr[:,:nxc,:nyc,0] 
    w_lr = data_lr[:,:nxc,:nyc,1]
    
    sn_hr = (s_hr - np.min(s_hr))/(np.max(s_hr) - np.min(s_hr))
    wn_hr = (w_hr - np.min(w_hr))/(np.max(w_hr) - np.min(w_hr))
    
    sn_lr = (s_lr - np.min(s_lr))/(np.max(s_lr) - np.min(s_lr))
    wn_lr = (w_lr - np.min(w_lr))/(np.max(w_lr) - np.min(w_lr))
    
    data_nhr = np.stack((sn_hr, wn_hr),axis=-1)
    data_nlr = np.stack((sn_lr, wn_lr),axis=-1)
    
    ## Load Stream Function ########
    train_hr = data_nhr[0:nf_train,:nx,:ny,:]
    train_lr = data_nlr[0:nf_train,:nxc,:nyc,:]  
    
    #
    val_hr = data_nhr[nf_train:nf_val,:nx,:ny,:]
    val_lr = data_nlr[nf_train:nf_val,:nxc,:nyc,:]    
    
    #x_test_hr = data_hr[nf_train:nf_test,:nx,:ny,0].reshape((nf,nx,ny,1))
    #x_test_lr = data_lr[nf:,:nxc,:nyc,0].reshape((nf,nxc,nyc,1))   
    
    image_shape = (train_hr.shape[1],train_hr.shape[2],train_hr.shape[3])   
    
    print("data loaded")    
    
    print(">> Creating the PIESRGAN network")
    gan = PIESRGAN(height_lr=nxc, width_lr=nxc, upscaling_factor=int(nx/nxc),training_mode=True,gen_lr=1e-4, dis_lr=1e-5,loss_weights={'percept':1e-1,'gen':1e-5, 'pixel':5.0, 'phy':0.001, 'ens':0.01})
#loss_weights={'percept':1e-1,'gen':1e-3, 'pixel':5.0, 'phy':0.001, 'ens':0.1}
#%% 
#    print(">> Loading pre-trained generator network") 
#    print('>> changing directories')
#    os.chdir(r'./data/weights/pre/') 
#    gan.generator.load_weights('Pre_generator_8X_epoch_100.h5')
#    print('>> changing back directories')
#    os.chdir(r'../../../')
#    
    print(">> Loading post-trained generator network") 
    print('>> changing directories')
    os.chdir(r'./data/weights/w1/post/') 
    gan.generator.load_weights('post_generator_8X_epoch_1000.h5')
    print('>> changing back directories')
    os.chdir(r'../../../')     
#%%    
 # Stage1: Train the generator w.r.t RRDB first
#    max_it= 100
#    print(">> Start training generator")
#    history = gan.train_generator(
#         epochs=10,max_it= max_it,
#         lr_train = train_lr,
#         hr_train = train_hr,
#         lr_test  = val_lr,
#         hr_test  = val_hr,         
#         batch_size=32,
#    )
#    
#    print(">> Generator trained based on MSE")
#    
#    '''Save pretrained generator'''
#    folder = './data/weights/pre/'
#    gan.generator.save(folder+'Pretrained_gen_model.h5')
#    gan.save_weights(folder+'Pre_',e=max_it)   
#  

#%%
#  #Stage2: Train the PIESRGAN with percept_loss, gen_loss and pixel_loss
    hr_ref = data_hr[-1,:nx,:ny,:]
    lr_ref = data_lr[-1,:nxc,:nyc,:]
    
    test_lr = data_nlr[-1,:nxc,:nyc,:]
    test_lr = test_lr.reshape(1,test_lr.shape[0],test_lr.shape[1],2)
    
    asn = (np.max(s_hr) - np.min(s_hr))
    bsn = np.min(s_hr)
    scf = np.array([asn, bsn])
    awn = (np.max(w_hr) - np.min(w_hr))
    bwn = np.min(w_hr)
    wcf = np.array([awn, bwn])
    
    
    max_it= 50000
    print(">> Start training PIESRGAN")
    gen_loss,dis_loss = gan.train_piesrgan(    
         epochs=max_it,
         first_epoch=0,
         batch_size=32,
         lr_train = train_lr,
         hr_train = train_hr,
         lr_rf  = lr_ref,
         hr_rf  = hr_ref, 
         lr_test  = test_lr,
         snc = scf, wnc = wcf,
         #datapath_train='../datasets/DIV2K_224/',  
         print_frequency=5,
         log_weight_frequency=max_it/20,
         log_img_frequency=max_it/50
         )
    print(">> Done with PIESRGAN training")
    
#%%    
    print(">> Saving Model and Weights Post training")
    file_name ='./data/weights/post/'
    gan.generator.save(file_name+'Post_gen_model.h5')
    gan.discriminator.save(file_name+'Post_dis_model.h5')    
    file_name=os.path.join(file_name,'post_')
    gan.save_weights(file_name,e=max_it)
#
    print(">> Saving Losses Post training")

    folder='./results/'
    np.savez(folder+'loss_p.npz', gen_loss=gen_loss, dis_loss=dis_loss)
    loss = np.load(folder+'loss_p.npz')
    gen_loss = loss['gen_loss']
    dis_loss = loss['dis_loss']

    print(np.min(s_hr))
#    ashn = np.max(s_hr)
#    bshn = np.min(s_hr)
#    sc00 = np.array([ashn, bshn])
#    awhn = np.max(w_hr)
#    bwhn = np.min(w_hr)
#    wc00 = np.array([awhn, bwhn])
#
#    asln = np.max(s_lr)
#    bsln = np.min(s_lr)
#    sc01 = np.array([asln, bsln])
#    awln = np.max(w_lr)
#    bwln = np.min(w_lr)
#    wc01 = np.array([awln, bwln])
#    
#    folder='./results/'
#    np.savez(folder+'scaling.npz',sh=sc00, wh=wc00, sl= sc01,wl = wc01)
    
#    print(">> Plotting training training")
#    
#    loss_plot(gen_loss, dis_loss)       
#%%
# Stage 3: Testing    

    print(">> Start testing PIESRGAN")       
    s_sr, w_sr = gan.test(lr_rf = lr_ref,
             lr_test  = test_lr,
             hr_rf    = hr_ref,
             snc = scf, wnc = wcf,
             output_name='epoch_'+str(max_it))
    print(">> Test finished, img file saved at: <test_1.png>")

#%%
#os.chdir(r'./data/weights')    
#gan.generator.load_weights('generator_idx0.h5')
#
#gan.test(lr_rf  = lr_ref,
#             lr_test  = test_lr,
#             hr_rf    = hr_ref,
#             snc = scf, wnc = wcf,
#             output_name='test_1.png')

