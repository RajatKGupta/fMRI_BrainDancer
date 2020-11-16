# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: December 16, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Script for CNN denoising. 

"""

# %% All imports
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, Conv1D,PReLU, Dropout,Activation, Input, Subtract
import keras.backend as K
from keras.callbacks import ModelCheckpoint,Callback, LearningRateScheduler
from keras import optimizers


# %%
# =============================================================================
# CNN Denoiser
# =============================================================================

def cnn(lwindow):
    length = int(lwindow)
    
    def CNN(length):
    
        inpt = Input(shape=(length,1))
        # 1st layer, Conv+relu
        x = Conv1D(filters=18, kernel_size=9, strides=1,data_format="channels_last", padding='same')(inpt)
        x = Activation('sigmoid')(x)
        # 6 layers, Conv+BN+relu
        for i in range(6):
            x = Conv1D(filters=18, kernel_size=9, strides=1,data_format="channels_last", padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('sigmoid')(x)  
        # last layer, Conv
        x = Dropout(0.2)(x)
        x = Conv1D(filters=1, kernel_size=9, strides=1, data_format="channels_last",padding='same')(x)
     
           # input - noise
        model = Model(inputs=inpt, outputs=x)
    
        return model


    # Just for future use! 
    #def step_decay(epoch):
        #initial_lrate = 0.0001
        #drop = 0.5
        #epochs_drop = 50
        #lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        #print(lrate)
        #return lrate

    # learning schedule callback, just for future use
    #lrate = LearningRateScheduler(step_decay)

    def custom_loss(y_true,y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        loss2 =  ( 1 - SS_res/(SS_tot + K.epsilon()) )
        return -loss2   
    
    ml = CNN(length)
    ml.compile(optimizer=optimizers.adam(), loss=[custom_loss], metrics=[custom_loss])
    ml.summary()
    
    return ml

# %%