

def make_lr_1(weights_name):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Cropping2D, InputLayer
    from utils.general_util import r2
    
    model = Sequential()
    model.add(InputLayer(input_shape=(20, 20, 1)))
    if '_Y_' in weights_name:
        left_crop, right_crop = 4, 0
        model.add(Cropping2D(cropping=((0, 0), (left_crop, right_crop))))#,  
    model.add(Flatten())            
    model.add(Dense(3))   
    
    model.compile(loss='mse',optimizer='adam', metrics=[r2], loss_weights=[1/5])      
    path_weights = 'experiments\\results_lr_1\\'+str(weights_name)+'.hdf5'
    model.load_weights(path_weights)
    
    return model


def make_fc_3(weights_name):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Cropping2D, InputLayer
    from utils.general_util import r2
    from keras.layers.advanced_activations import LeakyReLU
    
    model = Sequential()
    model.add(InputLayer(input_shape=(20, 20, 1)))
    if '_Y_' in weights_name:
        left_crop, right_crop = 4, 0
        model.add(Cropping2D(cropping=((0, 0), (left_crop, right_crop))))   
    model.add(Flatten())
    model.add(Dense(units = 400))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(units = 200))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(units = 60))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(3))
    
    model.compile(loss='mse',optimizer='adam', metrics=[r2], loss_weights=[1/5])
    path_weights = 'experiments\\results_fc_3\\'+str(weights_name)+'.hdf5'
    model.load_weights(path_weights)
            
    return model


def make_conv1d_2(weights_name):
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D, Reshape, Flatten, Dense, Cropping2D, InputLayer
    from utils.general_util import r2
    from keras.layers.advanced_activations import LeakyReLU
        
    alpha = 0.001
    num_layers = 2
    model_dim = 20
    
    model = Sequential()    
    model.add(InputLayer(input_shape=(20,20,1)))
    if '_Y_' in weights_name:
        model_dim, left_crop, right_crop = 16, 4, 0
        model.add(Cropping2D(cropping=((0, 0), (left_crop, right_crop))))     
    model.add(Reshape(target_shape = (20,model_dim)))
    for l in range(num_layers):
        model.add(Conv1D(32, (3)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(units = 128))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(units = 60))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(3))
    
    model.compile(loss='mse',optimizer='adam', metrics=[r2], loss_weights=[1/5])
    path_weights = 'experiments\\results_conv1d_2\\'+str(weights_name)+'.hdf5'
    model.load_weights(path_weights)

    return model

def make_conv1d_fs5_2(weights_name):
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D, Reshape, Flatten, Dense, Cropping2D, InputLayer
    from utils.general_util import r2
    from keras.layers.advanced_activations import LeakyReLU
        
    alpha = 0.001
    num_layers = 2
    model_dim = 20
    
    model = Sequential()    
    model.add(InputLayer(input_shape=(20,20,1)))
    if '_Y_' in weights_name:
        model_dim, left_crop, right_crop = 16, 4, 0
        model.add(Cropping2D(cropping=((0, 0), (left_crop, right_crop))))     
    model.add(Reshape(target_shape = (20,model_dim)))
    for l in range(num_layers):
        model.add(Conv1D(32, (5)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(units = 128))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(units = 60))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(3))
    
    model.compile(loss='mse',optimizer='adam', metrics=[r2], loss_weights=[1/5])
    path_weights = 'experiments\\results_conv1d_fs5_2\\'+str(weights_name)+'.hdf5'
    model.load_weights(path_weights)

    return model

def make_conv2d_1(weights_name):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Cropping2D, InputLayer
    from utils.general_util import r2
    from keras.layers.advanced_activations import LeakyReLU

    alpha = 0.001   

    model = Sequential()    
    model.add(InputLayer(input_shape=(20,20,1)))
    if '_Y_' in weights_name:
        left_crop, right_crop = 4, 0
        model.add(Cropping2D(cropping=((0, 0), (left_crop, right_crop))))  
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size = (2, 2)))    
    # Step 3 - Flattening
    model.add(Flatten())
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(units = 128))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(units = 60))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(3))
    
    model.compile(loss='mse',optimizer='adam', metrics=[r2], loss_weights=[1/5])
    path_weights = 'experiments\\results_conv2d_1\\'+str(weights_name)+'.hdf5'
    model.load_weights(path_weights)
    
    return model


def make_model(model_name, model_weights):
    if model_name =='lr_1':
        model = make_lr_1(model_weights)
    elif model_name == 'fc_3':
        model = make_fc_3(model_weights)
    elif model_name == 'conv1d_2':
        model = make_conv1d_2(model_weights)
    elif model_name == 'conv1d_fs5_2':
        model = make_conv1d_fs5_2(model_weights)
    elif model_name == 'conv2d_1':
        model = make_conv2d_1(model_weights)
    else:
        raise ValueError('Model name is invalid')
    return model

