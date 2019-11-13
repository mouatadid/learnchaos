# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:24:18 2019

@author: pocse
"""

from keras.models import Model
from keras.layers import Input, Cropping2D, Reshape, Conv1D, MaxPooling1D, Add, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, CSVLogger#, EarlyStopping
import numpy as np
from utils.general_util import r2, get_data_generators
import time
import tensorflow as tf
import os

class my_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('val_r2')>0.95:
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training =True


def resconv1d(job_id, data_dir, j='all', weighted=True, num_epochs=10, pretrained=False, 
        best_job_id=None, test_mode=False, data_dir_test=None, l96_var = 'XY',
        num_layers = 1):
    
    time_start = time.time()
    np.random.seed(11)
    alpha = 0.001
    
    model = Sequential()
    
    if l96_var == 'Y':
        model_dim = 16
        left_crop = 4
        right_crop = 0
    elif l96_var == 'Z':
        model_dim =4
        left_crop = 0
        right_crop = 16
    else:
        model_dim = 20
    
    inputs = Input(shape=(20, 20,1))
    
    #model.add(InputLayer(input_shape=(20,20,1)))
    
    if l96_var != 'XY':
        #model.add(Cropping2D(cropping=((0, 0), (left_crop, right_crop))))
        inputs = Cropping2D(cropping=((0,0), (left_crop, right_crop)))(inputs)
        
    #model.add(Reshape(target_shape = (20,model_dim)))
    x0 = Reshape(target_shape=(20, model_dim))(inputs)
    
    x1 = Conv1D(filters = 32, kernel_size = 3)(x0)
    x1lr = LeakyReLU(alpha=alpha)(x1)
    x1mp = MaxPooling1D(pool_size=2)(x1lr)
    
    x2 = Conv1D(filters = 32, kernel_size = 3)(x1mp)
    xres1 = Add()([x0, x2])
    x2lr = LeakyReLU(alpha=alpha)(xres1)
    x2mp = MaxPooling1D(pool_size=2)(x2lr)
    
    x3 = Conv1D(filters = 32, kernel_size = 3)(x2mp)
    xres2 = Add()([x1mp, x3])
    x3lr = LeakyReLU(alpha=alpha)(xres2)
    x3mp = MaxPooling1D(pool_size=2)(x3lr)
    
    x4 = Flatten()(x3mp)
           
    x5 = Dense(units=128)(x4)
    x5lr = LeakyReLU(alpha=alpha)(x5)
    x6 = Dense(units=60)(x5lr)
    x6lr = LeakyReLU(alpha=alpha)(x6)
    x7 = Dense(units=3)(x6lr)
    
    model = Model(inputs=inputs, outputs = x7)

    if weighted==False:
        model.compile(loss='mse',optimizer='adam', metrics=[r2])
        loss_weights='none'
    elif weighted==True:
            if j ==0:
                loss_weights=1/0.1
            else:
                loss_weights=1/5
            model.compile(loss='mse',optimizer='adam', metrics=[r2], loss_weights=[loss_weights])
            
    
            
    #load data
    train_generator, val_generator, test_generator = get_data_generators(data_dir, j = j)
    
    N = data_dir[data_dir.index('_N')+2:-3]   
    #callbacks
    path_log = 'experiments\\results_conv1d\\N'+str(N)+'_conv1d_'+str(num_layers)+'_checkpoints_'+str(test_mode)+'_'+str(l96_var)+'_'
    os.makedirs(path_log)
    checkpoints = ModelCheckpoint(path_log+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    early_stopping = my_callback() #EarlyStopping(monitor='val_r2', patience=3, baseline= 0.95)
    csv_logger = CSVLogger(path_log+'training_conv1d_'+str(num_layers)+'.log', separator=',', append=True)
    
    
    if pretrained ==True:
        path_weights = 'experiments\\results_conv1d\\N'+str(N)+'_conv1d_'+str(num_layers)+'_checkpoints_'+str(best_job_id)+'.hdf5'#+str(data_dir[17:-1])+str(best_job_id)+'\\'
#        weights_id = [f for f in os.listdir(path_weights) if f.endswith('.hdf5')]
#        weights_id = str(weights_id[0])
        model.load_weights(path_weights)
    
    if j!='all':
        model.add(Dense(1))
    if pretrained == False:
        model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n//train_generator.batch_size,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.n//val_generator.batch_size,
                                  epochs=num_epochs,
                                  callbacks=[checkpoints, early_stopping, csv_logger])
        
    #evaluate
    print('\nevaluating the train generator')
    loss_train, metric_train = model.evaluate_generator(generator=train_generator,
                                                            steps=train_generator.n//train_generator.batch_size)#len(train_generator))

    if test_mode ==True:
        _, _, test_generator = get_data_generators(data_dir_test, j = j)   
    print('\nevaluating the test generator')
    loss_test, metric_test = model.evaluate_generator(generator=test_generator,
                                                        steps=test_generator.n//test_generator.batch_size)#len(test_generator))
    
    #get r2 values
    print('r2_train = '+str(metric_train)+'\nr2_test = '+str(metric_test))
    
    #keep track of time
    time_exp = time.time()-time_start
    
    model_out = {'time_exp'     : time_exp/60,
                 'loss_train'   : loss_train,
                 'loss_test'    : loss_test,
                 'r2_train'     : metric_train,
                 'r2_test'      : metric_test
                     
                 }
    
    
    model_params = {'model_name'   : 'resconv1D_'+str(num_layers),
                    'job_id'       : job_id,
                    'pretrained'   : pretrained,
                    'best_job_id'  : best_job_id,
                    'data_dir'     : data_dir[17:],
                    'test_mode'    : test_mode,
                    'data_dir_test': data_dir_test[22:],
                    #'N'         : N,
                    #'input_shape': x_train.shape,
                    'target'       : j,
                    'weighted'     : weighted,
                    'num_epochs'   : num_epochs,
                    'l96_var'      : l96_var
                    }
    
    return model, model_out, model_params

                    