# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 17:35:47 2021

@author: vwang
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:41:39 2021
@author: vwang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:36:44 2021
@author: vwang
"""

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import skimage
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import numpy as np
import os
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Reshape, Conv2DTranspose
#from tensorflow.keras.utils import np_utils
#from keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
#from tensorflow import keras



code = "00"



def load_data(path):
    addr1 = "C:/microdata/BBBC006_v1_images_z_" + code
    addr2 = "C:/microdata/BBBC006_v1_images_z_17"
    #saveAddr = "C:/Users/jwang/Desktop/Unet"
    img_rows, img_cols = 260, 348
    #for secments
    upleft = []
    upright = []
    downleft = []
    downright = []
    train1 = []
    train2 = []
    truth1 = []
    truth2 = []
    print(addr1)
    #生成to be trained array
    for filename in os.listdir(addr1)[::2]:
        realad = addr1 + "/" + filename 
        img = cv2.imread(realad,-1)
        if img is not None:
            upleft = img[:260, :348]
            upright = img[:260, 348:]
            downleft = img[260:, :348]
            downright = img[260:, 348:]
            train1.append(upleft)
            train1.append(upright)
            train1.append(downleft)
            train1.append(downright)
        
        
    for filename in os.listdir(addr1)[1::2]:
        realad = addr1 + "/" + filename 
        img = cv2.imread(realad,-1)
        if img is not None:
            upleft = img[:260, :348]
            upright = img[:260, 348:]
            downleft = img[260:, :348]
            downright = img[260:, 348:]
            train2.append(upleft)
            train2.append(upright)
            train2.append(downleft)
            train2.append(downright)
            
            #生成ground truth array
    for filename in os.listdir(addr2)[::2]:
        realad = addr2 + "/" + filename 
        img = cv2.imread(realad,-1)
        if img is not None:
            upleft = img[:260, :348]
            upright = img[:260, 348:]
            downleft = img[260:, :348]
            downright = img[260:, 348:]
            truth1.append(upleft)
            truth1.append(upright)
            truth1.append(downleft)
            truth1.append(downright)
        
        
    for filename in os.listdir(addr2)[1::2]:
        realad = addr2 + "/" + filename 
        img = cv2.imread(realad,-1)
        if img is not None:
            upleft = img[:260, :348]
            upright = img[:260, 348:]
            downleft = img[260:, :348]
            downright = img[260:, 348:]
            truth2.append(upleft)
            truth2.append(upright)
            truth2.append(downleft)
            truth2.append(downright)
        
    train1 = np.array(train1)   
    train2 = np.array(train2)         
    truth1 = np.array(truth1)
    truth2 = np.array(truth2)
    train = np.stack((train1,train2),axis = 3)
    truth = np.stack((truth1,truth2),axis = 3)
    converge = np.amax(train)
    test_x = train[2700:]
    test_y = truth[2700:]
    train = train[:2700]
    truth = truth[:2700]
    train = train.astype('float16')/converge
    test_x = test_x.astype('float16')/converge
    test_y = test_y.astype('float16')/converge
    truth = truth.astype('float16')/converge
    train = np.pad(train, ((0,0), (6,6), (2,2), (0,0)), 'edge')
    truth = np.pad(truth, ((0,0), (6,6), (2,2), (0,0)), 'edge')
    test_x = np.pad(test_x, ((0,0), (6,6), (2,2), (0,0)), 'edge')
    test_y = np.pad(test_y, ((0,0), (6,6), (2,2), (0,0)), 'edge')
    return (train,test_x,truth,test_y)

def mod():
    input4 = tf.keras.Input(shape = (34,44,2))
    input0 = tf.keras.Input(shape = (68,88,2))
    inputA = tf.keras.Input(shape = (136,176,2))
    inputB = tf.keras.Input(shape = (272,352,2))
    
    layer01, output4 = unet4(input4)
    output4in = UpSampling2D(size = (2,2))(output4)
    
    inputA4 = tf.keras.layers.concatenate([input0,output4in],axis=3)
    
    layer01, layer02, layer03 = unet0(inputA4, layer01)
    output0in = UpSampling2D(size = (2,2))(layer03)
    
    inputA0 = tf.keras.layers.concatenate([inputA,output0in],axis=3)
    #unet for A
    
    layer1,layer2, layer3,outputA = unetA(inputA0, layer01, layer02)
    outputAin = UpSampling2D(size = (2,2))(outputA)
    #inputB = tf.concat((inputB,outputAin),3)
    
    outputB = tf.keras.layers.concatenate([inputB,outputAin],
                            axis=3)
    #unet for B
    outputB = unetB(layer1,layer2, layer3, inputB)

    
    #model = tf.keras.Model(inputs = inputA, outputs = outputA)
    
    model = Model(inputs = [input4,input0,inputA,inputB],outputs = [output4,layer03,outputA,outputB])
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse')
    
    return model

def unet4(X):
    
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(X)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    up6 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
    
    merge6 = concatenate([conv1,up6], axis = 3)
   
    
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    conv7 = Conv2D(2, 1, activation = 'linear')(conv6)
    
    return merge6, conv7
    
def unet0(X, layer01):
    
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(X)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    conv2 = tf.keras.layers.Add()([conv2, layer01])
    #conv2 = tf.keras.layers.Concatenate(axis=3)([conv2, layer01])  
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4, training=True)
    
    up5 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
    merge5 = concatenate([conv2,up5], axis = 3)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #print(tf.shape(conv5))
    
    up6 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv1,up6], axis = 3)
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv7 = Conv2D(2, 1, activation = 'linear')(conv6)
    
    return merge5,merge6, conv7
    
def unetA(X, layer01, layer02):
    
    
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(X)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    conv2 = tf.keras.layers.Add()([conv2, layer02])
    #conv2 = tf.keras.layers.Concatenate(axis=3)([conv2, layer02])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    conv3 = tf.keras.layers.Add()([conv3, layer01])
    #conv3 = tf.keras.layers.Concatenate(axis=3)([conv3, layer01])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    up5 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv3,up5], axis = 3)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #print(tf.shape(conv5))
    
    up6 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 3)
    conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #print(tf.shape(conv6))

    up7 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 3)
    conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    conv8 = Conv2D(2, 1, activation = 'linear')(conv7)
    
    return merge5,merge6,merge7,conv8
    
def unetB(layer1,layer2,layer3, X): 
    ##### removed one depth because I was running out of memory somehow. Although it was solved, the original script was replaced. Please use your orginal UnetB (with one more depth) if possible
    
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(X)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2,padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', dilation_rate=2, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    conv2 = tf.keras.layers.Add()([conv2, layer3])
   # conv2 = tf.keras.layers.Concatenate(axis=3)([conv2, layer3])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
  
    conv3 = tf.keras.layers.Add()([conv3, layer2])
   #conv3 = tf.keras.layers.Concatenate(axis=3)([conv3, layer2])
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    #
    conv4 = tf.keras.layers.Add()([conv4, layer1])
    #conv4 = tf.keras.layers.Concatenate(axis=3)([conv4, layer1])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
   
    conv10 = Conv2D(2, 1, activation = 'linear')(conv9)
    
    return conv10
    

(train,test_x,truth,test_y) = load_data("")
saveAddr = "C:/Users/vwang/OneDrive/Desktop/4layeradd/"

trainA = resize(train, (train.shape[0] // 1, train.shape[1] // 2, train.shape[2] // 2, train.shape[3] // 1),
                       anti_aliasing=True)
truthA = resize(truth, (truth.shape[0] // 1, truth.shape[1] // 2, truth.shape[2] // 2, truth.shape[3] // 1),
                       anti_aliasing=True)
test_xA = resize(test_x, (test_x.shape[0] // 1, test_x.shape[1] // 2, test_x.shape[2] // 2, test_x.shape[3] // 1),
                       anti_aliasing=True)
test_yA = resize(test_y, (test_y.shape[0] // 1, test_y.shape[1] // 2, test_y.shape[2] // 2, test_y.shape[3] // 1),
                       anti_aliasing=True)

train0 = resize(trainA, (trainA.shape[0] // 1, trainA.shape[1] // 2, trainA.shape[2] // 2, trainA.shape[3] // 1),
                       anti_aliasing=True)
truth0 = resize(truthA, (truthA.shape[0] // 1, truthA.shape[1] // 2, truthA.shape[2] // 2, truthA.shape[3] // 1),
                       anti_aliasing=True)
test_x0 = resize(test_xA, (test_xA.shape[0] // 1, test_xA.shape[1] // 2, test_xA.shape[2] // 2, test_xA.shape[3] // 1),
                       anti_aliasing=True)
test_y0 = resize(test_yA, (test_yA.shape[0] // 1, test_yA.shape[1] // 2, test_yA.shape[2] // 2, test_yA.shape[3] // 1),
                       anti_aliasing=True)

train4 = resize(train0, (train0.shape[0] // 1, train0.shape[1] // 2, train0.shape[2] // 2, train0.shape[3] // 1),
                       anti_aliasing=True)
truth4 = resize(truth0, (truth0.shape[0] // 1, truth0.shape[1] // 2, truth0.shape[2] // 2, truth0.shape[3] // 1),
                       anti_aliasing=True)
test_x4 = resize(test_x0, (test_x0.shape[0] // 1, test_x0.shape[1] // 2, test_x0.shape[2] // 2, test_x0.shape[3] // 1),
                       anti_aliasing=True)
test_y4 = resize(test_y0, (test_y0.shape[0] // 1, test_y0.shape[1] // 2, test_y0.shape[2] // 2, test_y0.shape[3] // 1),
                       anti_aliasing=True)



model = mod()
model.summary()
#model.fit(trainA, truthA, epochs=1, batch_size=2, validation_data=(test_xA,test_yA))
model.fit([train4, train0, trainA, train], [truth4, truth0, truthA, truth], epochs=80, batch_size=1)#, validation_data=([test_xA,test_x],[test_yA,test_y]))
model.save(saveAddr + code) ## please change it back to your way of saving

'''
result = []
for i in range(len(test_x)):
    tested = model.predict(np.reshape((test_xA[i] , test_x[i]),((1,272,352,2))))
    tested = np.reshape(tested,((272,352,2)))
    tested = tested.tolist()
    result.append(tested)
tested = np.array(result)
display = -2
test_input = test_x[display]
test_input  = np.concatenate((test_input , test_input [:, :, 0:1]), axis=2)
plt.imshow(test_input)
test_input *= 255
groundTruth = test_y[display]
groundTruth = np.concatenate((groundTruth, groundTruth[:, :, 0:1]), axis=2)
plt.imshow(groundTruth)
#cv2.imwrite(saveAddr + "/truth.tif", groundTruth)
groundTruth *= 255
tested = tested[display]
tested = np.concatenate((tested, tested[:, :, 0:1]), axis=2)
plt.imshow(tested)
tested *= 255
cv2.imwrite(saveAddr + "/input" + code + ".png", test_input)
cv2.imwrite(saveAddr + "/truth" + code + ".png", groundTruth)
cv2.imwrite(saveAddr + "/tested" + code + ".png", tested)
'''