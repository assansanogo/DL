import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler



class CCPminiBlock(tf.keras.layers.Layer):
    """
    How to define a combination of layers (conv2D & Pooling) 
    """
    def __init__(self, k_size, activ):
        super(CCPminiBlock, self).__init__()
        
        self.conv =  Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')
        
        self.pool =   MaxPooling2D(pool_size=(2, 2))
    def call(self, inputs):
        x = self.conv(inputs)
        #x = self.conv(x)
        pooled_x = self.pool(x)
        return x, pooled_x

    

class CCPBlock(tf.keras.layers.Layer):
    """
    How to define a combination of complex layers (Many Conv2D & Pooling)  
    """
    
    def __init__(self, k_size, activ):
        super(CCPBlock, self).__init__()
        # creation of a Mini network inside the block
        self.net = tf.keras.Sequential()
        
        # add double convolution
        self.net.add(keras.layers.Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal'))
        self.net.add(keras.layers.Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal'))
        
        # add pooling
        self.pool = keras.layers.MaxPooling2D(pool_size=(2, 2))
        
    def call(self, inputs):
        

        return self.net(inputs), self.pool(self.net(inputs))



def block_ccp(inputs, k_size = 1, activ= 'relu'):
    """
    functional approach : block of 2 convolutions + 1 Max Pooling
    
    """
    conv = Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    return conv1, pool


def block_ccd(inputs, k_size = 1, activ= 'relu', drop_factor =0.5):
    """
    functional approach : block of 2 convolutions + 1 Dropout
    
    """
    conv = Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv = Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(conv)
    drop = Dropout(0.5)(conv)
    return drop


def block_mccuc(input_branch_down,input_branch_up, k_size = 1, activ= 'relu', up_size = 2):
    """
    functional approach : block of 1 merge + 2 convolutions + [1 UpSampling + conv]
    
    """
    merge = concatenate([input_branch_down, input_branch_up], axis = 3)
    
    conv = Conv2D(64*k_size, 
                  3, 
                  activation = activ, 
                  padding = 'same', 
                  kernel_initializer = 'he_normal')(merge)
    conv = Conv2D(64*k_size, 
                  3, 
                  activation = activ, 
                  padding = 'same', 
                  kernel_initializer = 'he_normal')(conv)
    
    upsampled = UpSampling2D(size = (up_size, up_size))(conv)
                             
    conv = Conv2D(64*k_size/2, 
                  2, 
                  activation = activ, 
                  padding = 'same', 
                  kernel_initializer = 'he_normal')(upsampled)
                             
    return conv


    
def block_mcccc(input_branch_down, input_branch_up, k_size = 1, activ= 'relu', up_size = 2):
    """

    functional approach : block of [64 - 64] conv + [2 -1] conv

    """
                             
                             
    merge = concatenate([input_branch_down, input_branch_up], axis = 3)
    conv = Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(merge)
    conv = Conv2D(64*k_size, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = Conv2D(2, 3, activation = activ, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv = Conv2D(1, 1, activation = 'sigmoid')(conv)
    return conv




def unet(pretrained_weights = None,input_size = (256,256,1), model_loss = 'binary_crossentropy'):
    """
    model creation : 
    - first layer defined as a class (learning purposes)
    - other layers defined by functions (learning purposes) (less elegant)
    
    """
    
    
    ccpblock = CCPBlock(1,'relu')
    inputs = Input(input_size)
    
    # input branch down
    # conv1, pool1 = block_ccp(inputs, k_size = 1, activ= 'relu')
    
    conv1, pool1 = ccpblock(inputs)
    conv2, pool2 = block_ccp(pool1, k_size = 2, activ= 'relu')
    conv3, pool3 = block_ccp(pool2, k_size = 4, activ= 'relu')
    
    # Bottom of the U
    drop4 = block_ccd(pool3, k_size = 8, activ= 'relu', drop_factor =0.5)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    drop5 = block_ccd(pool4, k_size = 8, activ= 'relu', drop_factor =0.5)
    up5 = UpSampling2D(size = (2,2))(drop5)                      
    
    # Output branch up
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up5)                          
    up7 = block_mccuc(drop4, up6, k_size = 8, activ= 'relu', up_size = 2)
    up8 = block_mccuc(conv3, up7, k_size = 4, activ= 'relu', up_size = 2)
    up9 = block_mccuc(conv2, up8, k_size = 2, activ= 'relu', up_size = 2)
    
    conv10 = block_mcccc(conv1, up9, k_size = 1, activ= 'relu', up_size = 2)
        
    
    # Functional definition of the model 
    model = Model(inputs, conv10)

    # Compilation of the model with parameters
    model.compile(optimizer = Adam(lr = 1e-4), 
                  loss = model_loss, 
                  metrics = ['accuracy'])
    
    # Display the model summary for structural info
    model.summary()

    # Load weights if prexisting weeights
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


