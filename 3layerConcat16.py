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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, AveragePooling2D, Reshape, Conv2DTranspose
from keras.utils import np_utils
from keras import backend as K
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
#from tensorflow import keras



code = "16"



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
    
    inputB = tf.keras.Input(shape = (272,352,2))
    outputB = unetA(inputB)
 
    model = Model(inputs = [inputB],outputs = [outputB])
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse')
    
    return model
    

    
def unetA(X):
    
    return X

    

(train,test_x,truth,test_y) = load_data("")
saveAddr = "C:/Users/vwang/OneDrive/Desktop/water/"






model = mod()
model.summary()
#model.fit(trainA, truthA, epochs=1, batch_size=2, validation_data=(test_xA,test_yA))
model.fit([train], [truth], epochs=1, batch_size=1)#, validation_data=([test_xA,test_x],[test_yA,test_y]))
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