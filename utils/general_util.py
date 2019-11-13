
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    from scipy.stats import truncnorm
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def l96_truth_step(X, Y, h, F, b, c):
    import numpy as np
    """
    Code for this function was based on https://github.com/djgagne/lorenz_gan
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Y[k * J: (k + 1) * J])
    for j in range(J * K):
        dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1]) - c * Y[j] + h * c / b * X[int(j / J)]
    return dXdt, dYdt


def run_lorenz96_truth(x_initial, y_initial, h, f, b, c, time_step, num_steps, burn_in, skip):
    import numpy as np
    """
    Code for this function was based on https://github.com/djgagne/lorenz_gan
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        f (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival

    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    # Calculate total Y forcing over archive period using trapezoidal rule
    y_trap = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    y_trap[:] = y_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        #if (n * time_step) % 1 == 0:
            #print(n, n * time_step)
        k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, f, b, c)
        k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                                y + k1_dydt * time_step / 2,
                                                h, f, b, c)
        k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                                y + k2_dydt * time_step / 2,
                                                h, f, b, c)
        k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                                y + k3_dydt * time_step,
                                                h, f, b, c)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = (y + y_trap) / skip
            i += 1
        elif n % skip == 1:
            y_trap[:] = y
        else:
            y_trap[:] += y
    return x_out, y_out, times, steps


def generate_L96(num_images = 10, K = 4, J = 4, time_step = 0.01, num_steps = 50000, 
                 test_mode=False, l96_var = 'XY'):
    '
    import numpy as np
    import os   
    
    path_out = 'data\\out_l96_'+str(l96_var)+'_N'+str(num_images)+'\\'
    if test_mode == True:
        path_out = 'data\\test_out_l96_'+str(l96_var)+'_N'+str(num_images)+'\\'
    os.makedirs(path_out)
    
    X = np.zeros(K)
    Y = np.zeros(J * K)
    burn_in=50
    skip=1
    X[0] = 1
    Y[0] = 1
    #h = 1
    #b = 10.0
    #c = 10.0
    
    F = 8
    #outputAll = np.zeros([1,8])
    b = get_truncated_normal(mean=11, sd=5, low=0, upp=22)
    c = get_truncated_normal(mean=11, sd=5, low=0, upp=22)
    h = get_truncated_normal(mean=1, sd=0.1, low=0, upp=2)
    theta_b, theta_c, theta_h = b.rvs(num_images), c.rvs(num_images), h.rvs(num_images)
    #plt.hist(b.rvs(100), normed=True)
    #plt.hist(c.rvs(100), normed=True)
    #plt.hist(h.rvs(100), normed=True)
    theta = np.concatenate((np.reshape(theta_b, (-1,1)), np.reshape(theta_c, (-1,1)), np.reshape(theta_h, (-1,1))), axis=1)  
    np.savetxt(path_out+'theta.out', theta, delimiter=',')
    
    print('Generating L96 output for '+str(num_images)+' images')
    for run in range(0, num_images):
        
        #b, c, h = round(theta_b[run]), round(theta_c[run]), round(theta_h[run])
        b, c, h = theta_b[run], theta_c[run], theta_h[run]
        X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps, burn_in, skip)
        output = np.concatenate((np.reshape(steps, (-1,1)), np.reshape(times, (-1,1)), X_out, Y_out), axis=1)
    
        #filename = 'l96_'+str(run)+'_b='+str(b)+'_c='+str(c)+'_h='+str(h)+'.out'
        filename = 'l96_'+str(run)+'.out'
        np.savetxt(path_out+filename, output, delimiter=',')
        print('run '+str(run)+' of '+str(num_images)+' completed.')

    return path_out

def plot3d_obs(N= 200, out_dir = 'data\\out_l96_XY_N200\\'):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    #for n in tqdm.tnrange(N):
    for n in tqdm(range(N)):
        out = np.loadtxt(out_dir+'l96_'+str(n)+'.out', delimiter = ',')
        # plot first three slow variables
        x = out[:,2:5]
        plt.rcParams.update({'font.size': 18})
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.plot(x[:,0],x[:,1],x[:,2], 'g')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        ax.set_title('Observed X phase diagram, run_id = '+str(n))
        figname = (out_dir+'\\vis_X_l96_'+str(n)+'.jpg')
        plt.savefig(figname, bbox_inches='tight')
        #plt.show()
        plt.close()
        # plot first three slow variables
        y = out[:,6:9]
        plt.rcParams.update({'font.size': 18})
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.plot(y[:,0],y[:,1],y[:,2], 'g')
        ax.set_xlabel('$y_1$')
        ax.set_ylabel('$y_2$')
        ax.set_zlabel('$y_3$')
        ax.set_title('Observed Y phase diagram, run_id = '+str(n))
        figname = (out_dir+'vis_Y_l96_'+str(n)+'.jpg')
        plt.savefig(figname, bbox_inches='tight')
        #plt.show()
        plt.close()

def gen_pixel_imgs(num_images, color, path_out, save=True, l96_var='XY', deltaT='all', test_mode=False):
    '''color: 'RGB' for color or 'L' for greyscale'''
    from PIL import Image
    import numpy as np
    import os
    
    if save == True:
        path_imgs = 'data\\imgs_l96_'+str(l96_var)+'_N'+str(num_images)+'_'+str(color)
        if test_mode==True:
            path_imgs = 'data\\test_imgs_l96_'+str(l96_var)+'_N'+str(num_images)+'_'+str(color)
        os.makedirs(path_imgs)
        
    #generate four pixel images
    print('Generating pixel imgs for '+str(num_images)+' images')
    for run in range(0, num_images):
        print(run)
        filename = path_out+'l96_'+str(run)+'.out'
        if l96_var == 'X':
            out = np.loadtxt(filename, delimiter=',')[:,2:6]
        elif l96_var == 'Y':
            out = np.loadtxt(filename, delimiter=',')[:,6:]
        elif l96_var == 'XY':
            out = np.loadtxt(filename, delimiter=',')[:,2:]
        if type(deltaT) != str:
            out = out[:deltaT, :]
            
        img = Image.fromarray(out, color)
        if save == True:
            img.save(path_imgs+'\\l96_'+str(l96_var)+'_'+str(run)+'.jpg', 'JPEG')
    
    labels_raw = np.loadtxt(path_out+'theta.out', delimiter=',')[:num_images,:]
    np.save(path_imgs+'\\labels', labels_raw)
            
    return path_imgs


def generate_L96_theta( num_images= 1, K = 4, J = 4, time_step = 0.01, num_steps = 50000,
                 test_mode=False, l96_var = 'XY', theta= None, theta_index =None,
                 plot=False, truth =None, job_id=None):
    import numpy as np
    import os   
    
    
    
    X = np.zeros(K)
    Y = np.zeros(J * K)
    burn_in=50
    skip=1
    X[0] = 1
    Y[0] = 1
    #h = 1
    #b = 10.0num_images
    #c = 10.0
    
    F = 8
    #outputAll = np.zeros([1,8])
    if theta.all() == None:
        b = get_truncated_normal(mean=11, sd=5, low=0, upp=22)
        c = get_truncated_normal(mean=11, sd=5, low=0, upp=22)
        h = get_truncated_normal(mean=1, sd=0.1, low=0, upp=2)
        theta_b, theta_c, theta_h = b.rvs(num_images), c.rvs(num_images), h.rvs(num_images)
    else:
        if num_images ==1:
            num_images = theta.shape[0]
        theta_b, theta_c, theta_h = theta[:num_images,0], theta[:num_images,1], theta[:num_images,2]
    
    path_out = 'experiments\\analyze_results_cnn\\'+str(job_id)+'_out_l96_'+str(l96_var)+'_N'+str(num_images)+'\\'
    if test_mode == True:
        path_out = 'experiments\\analyze_results_cnn\\'+str(job_id)+'_test_out_l96_'+str(l96_var)+'_N'+str(num_images)+'\\'
    if os.path.isdir(path_out)==False:
        os.makedirs(path_out)
        
        
    theta = np.concatenate((np.reshape(theta_b, (-1,1)), np.reshape(theta_c, (-1,1)), np.reshape(theta_h, (-1,1))), axis=1)  
    
    np.savetxt(path_out+str(truth)+'_theta.out', np.hstack((theta_index.reshape((theta_index.shape[0], 1)), theta)), delimiter=',')
    
    print('Generating L96 output for '+str(num_images)+' images')
    for run in range(0, num_images):
        
        #b, c, h = round(theta_b[run]), round(theta_c[run]), round(theta_h[run])
        b, c, h = theta_b[run], theta_c[run], theta_h[run]
        X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps, burn_in, skip)
        output = np.concatenate((np.reshape(steps, (-1,1)), np.reshape(times, (-1,1)), X_out, Y_out), axis=1)
    
        #filename = 'l96_'+str(run)+'_b='+str(b)+'_c='+str(c)+'_h='+str(h)+'.out'
        filename = str(truth)+'_l96_'+str(int(theta_index[run]))+'.out'
        np.savetxt(path_out+filename, output, delimiter=',')
        print('run '+str(run)+' of '+str(num_images)+' completed.')
        
        if plot == True:
            import matplotlib.pyplot as plt

            print(X_out.max(), X_out.min())
            plt.figure(figsize=(8, 10))
            plt.pcolormesh(np.arange(K + 1), times, X_out, cmap="RdBu_r")
            plt.title("Lorenz '96 X "+str(truth))
            plt.colorbar()
            figname = path_out +str(int(theta_index[run]))+'_'+str(truth)+'_l96_Xout_b='+str(b)+'_c='+str(c)+'_h='+str(h)+'.jpg'
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
                
            plt.figure(figsize=(8, 10))
            plt.pcolormesh(np.arange(J * K + 1), times, Y_out, cmap="RdBu_r")
            plt.xticks(np.arange(0, J * K, J))
            plt.title("Lorenz '96 Y "+str(truth))
            plt.colorbar()
            figname = path_out +str(int(theta_index[run]))+'_'+ str(truth)+'_l96_Yout_b='+str(b)+'_c='+str(c)+'_h='+str(h)+'.jpg'
            plt.savefig(figname, bbox_inches='tight')
            plt.close()

    return path_out





def load_ds(model_name, N=10, j='all', trp=0.9, trv = 0.05, path_imgs = 'data\\imgs_l96_XY_N101_L', path_out='data\\out_l96\\', l96_var = 'XY'):
    import numpy as np
    from PIL import Image
        
    #generate sequences from L96 outputs
    if model_name =='transformer':
        print('Generating chunked sequences data for '+str(N)+' L96 outputs')
        #transformer
        seqAll = []
        for run in range(0,N):
            seq_path = path_out+'l96_'+str(run)+'.out'
            seq = np.loadtxt(seq_path, delimiter = ',')[:,2:]
            l, w = seq.shape[-2], seq.shape[-1]
            seq = seq[:((l//w)*w),:].reshape(seq.shape[0]//w, w, w)
            seqAll.append(seq)
        sequences_raw = np.stack(seqAll, axis=0)
        sequences = sequences_raw.reshape(sequences_raw.shape[0]*sequences_raw.shape[1], w, w)
            
        labels_raw = np.loadtxt(path_out, delimiter=',')[:N,:]
        labels = 0*labels_raw[1,:]
        for i in range(len(labels_raw)):
            te = np.ones((sequences_raw.shape[1], labels_raw.shape[1]))*labels_raw[i,:]
            labels = np.vstack((labels, te))
        labels = labels[1:,:]  
        inputs = sequences
   #generate images from L96 outputs     
    else:
        print('Generating chunked image data for '+str(N)+' grayscale images')
        imgAll = []
        for img in range(0,N):
            image_path = path_imgs+'\\l96_'+str(l96_var)+'_'+str(img)+'.jpg'
            img_new = Image.open(image_path)
            img_new = np.array(img_new)
            imgAll.append(img_new)
        images = np.stack(imgAll, axis=0)#/255.0
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
        images_raw = images    
        labels_raw = np.load(path_imgs+'\\labels.npy').astype(float)[:N, :]
        
        #cnn
        [n, l, w, c] = images_raw.shape
        len_stop = ((n*l)//w)*w
        images = images.reshape(n*l, w, c)[:len_stop, :, :]
        images = images.reshape(images.shape[0]//w, w, w, c)
        labels = 0*labels_raw[1,:]
        for i in range(len(labels_raw)):
            te = np.ones((images.shape[0]//n, labels_raw.shape[1]))*labels_raw[i,:]
            labels = np.vstack((labels, te))
        labels = labels[1:,:]
        #conv_lstm
        if model_name == 'conv_lstm':
            images = np.reshape(images, (images.shape[0],images.shape[1], 
                                         1, images.shape[2], images.shape[3]))
        #resnet
        if model_name == 'resnet':
            images = np.reshape(images, (images.shape[0],images.shape[3], 
                                         images.shape[1], images.shape[2]))
        inputs = images    
            
    #shuffle dataset
    ds = list(zip(inputs, labels))
    np.random.shuffle(ds)
    inputs, labels = zip(*ds)
    inputs, labels = np.array(inputs), np.array(labels)         
    
    #split data into train, val and test
    tr = int(trp*labels.shape[0])
    tv = int(trv*tr)
    x_train, x_val, x_test = inputs[:tv], inputs[tv:tr], inputs[tr:]
    if j == 'all':
        y_train, y_val, y_test = labels[:tv], labels[tv:tr], labels[tr:]
    else:
        y_train, y_val, y_test = labels[:tv, j], labels[tv:tr], labels[tr:, j]    
    
    
            
      
    return x_train, x_val, x_test, y_train, y_val, y_test



def save_chunked_imgs(num_images, path_out, path_imgs, save = True, color='L',  l96_var='XY', test_mode=False):
    
   
    from PIL import Image
    import os
    import numpy as np
    import pandas as pd
    
    x_train, x_val, x_test, y_train, y_val, y_test = load_ds('cnn', num_images, 'all', 0.9, 0.95, path_imgs, path_out, l96_var = l96_var)

    data_dir = 'data\\chunked_l96_'+str(l96_var)+'_N'+str(num_images)+'_'+str(color)+'\\'
    if test_mode == True:
        data_dir = 'data\\test_chunked_l96_'+str(l96_var)+'_N'+str(num_images)+'_'+str(color)+'\\'
    if save == True:
        path_train = data_dir+'train\\'
        path_val = data_dir+'val\\'
        path_test = data_dir+'test\\'
        
    
        os.makedirs(path_train)
        os.makedirs(path_val)
        os.makedirs(path_test)
        
    #generate four pixel images
    print('Generating pixel imgs for '+str(num_images)+' images')
    filenames_train = []
    for run in range(x_train.shape[0]):
        out = x_train[run, :, :, 0]
        img = Image.fromarray(out, color)
        if save == True:
            img.save(path_train+'\\l96_'+str(l96_var)+'_'+str(run)+'.jpg', 'JPEG')
            filenames_train.append('l96_'+str(l96_var)+'_'+str(run)+'.jpg')
    run_next = run
    filenames_train = np.stack(filenames_train, axis=0)
    labels_train = {'id' : filenames_train, 
                    'b' : y_train[:,0], 
                    'c' : y_train[:,1],
                    'h' :y_train[:,2]}
    labels_train = pd.DataFrame(labels_train)
    labels_train.to_csv(path_train+'\\labels_train.csv', index = False)   
    
    filenames_val = []
    i=0
    for run in range(run_next, run_next+x_val.shape[0]):
        out = x_val[i, :, :, 0]
        img = Image.fromarray(out, color)
        if save == True:
            img.save(path_val+'\\l96_'+str(l96_var)+'_'+str(run+1)+'.jpg', 'JPEG')
            filenames_val.append('l96_'+str(l96_var)+'_'+str(run+1)+'.jpg')
        i+=1
    run_next = run+1
    filenames_val = np.stack(filenames_val, axis=0)
    labels_val = {'id' : filenames_val, 
                    'b' : y_val[:,0], 
                    'c' : y_val[:,1],
                    'h' :y_val[:,2]}
    labels_val = pd.DataFrame(labels_val)
    labels_val.to_csv(path_val+'\\labels_val.csv', index = False) 
    
    filenames_test = []
    i=0
    for run in range(run_next, run_next+x_test.shape[0]):
        out = x_test[i, :, :, 0]
        img = Image.fromarray(out, color)
        if save == True:
            img.save(path_test+'\\l96_'+str(l96_var)+'_'+str(run+1)+'.jpg', 'JPEG')
            filenames_test.append('l96_'+str(l96_var)+'_'+str(run+1)+'.jpg')
        i+=1
    filenames_test = np.stack(filenames_test, axis=0)
    labels_test = {'id' : filenames_test, 
                    'b' : y_test[:,0], 
                    'c' : y_test[:,1],
                    'h' :y_test[:,2]}
    labels_test = pd.DataFrame(labels_test)
    labels_test.to_csv(path_test+'\\labels_test.csv', index = False) 
    return data_dir

def build_dataset(N=1, N_test=1, l96_var='XY', data_dir=None, data_dir_test=None):
    #from utils import generate_L96, gen_pixel_imgs, save_chunked_imgs

    if N>1:
        print('\nBuilding train dataset')
        #generate L96 runs
        out_path =  generate_L96(num_images = N, l96_var = l96_var)
            
        #generate pix images from l96 runs
        imgs_path = gen_pixel_imgs(num_images=N, color='L', l96_var=l96_var, save=True, path_out=out_path)
        #load ds
          
        #convert to chunked images labels, saved in train, val test
        data_dir = save_chunked_imgs(num_images=N, path_out = out_path, path_imgs=imgs_path, l96_var=l96_var)
        data_dir_test = None
    
    if N_test>1:
        print('\nBuilding test dataset')
        #generate L96 runs
        out_path_test =  generate_L96(num_images = N_test, test_mode=True, l96_var = l96_var)        
        #generate pix images from l96 runs
        imgs_path_test = gen_pixel_imgs(num_images=N_test, color='L', l96_var=l96_var, save=True, path_out=out_path_test, test_mode=True)  
        #convert to chunked images labels, saved in train, val test
        data_dir_test = save_chunked_imgs(num_images=N_test, path_out = out_path_test, path_imgs=imgs_path_test, test_mode=True, l96_var=l96_var)
    
    return data_dir, data_dir_test


def get_data_generators(data_dir = 'data\\chunked_l96_XY_N2_L\\', j = 'all', l96_var = 'XY', normalized_target =False):
    
    from keras_preprocessing.image import ImageDataGenerator
    import pandas as pd   
    
    path_train = data_dir+'train\\labels_train.csv' 
    dir_train = data_dir+'train' 
    
    path_val = data_dir+'val\\labels_val.csv' 
    dir_val = data_dir+'val' 
    
    path_test = data_dir+'test\\labels_test.csv' 
    dir_test = data_dir+'test' 
    
    #build dataframe
    traindf=pd.read_csv(path_train, dtype=str)
    valdf=pd.read_csv(path_val, dtype=str)
    testdf=pd.read_csv(path_test, dtype=str)
    
    traindf[['b', 'c', 'h']] = traindf[['b', 'c', 'h']].apply(pd.to_numeric, errors='coerce')
    valdf[['b', 'c', 'h']] = valdf[['b', 'c', 'h']].apply(pd.to_numeric, errors='coerce')
    testdf[['b', 'c', 'h']] = testdf[['b', 'c', 'h']].apply(pd.to_numeric, errors='coerce')
    
    if normalized_target == True:
        traindf[['b', 'c']]=(traindf[['b', 'c']]-11)/5
        traindf[['h']]=(traindf[['h']]-1)/0.1
        valdf[['b', 'c']]=(valdf[['b', 'c']]-11)/5
        valdf[['h']]=(valdf[['h']]-1)/0.1
        testdf[['b', 'c']]=(testdf[['b', 'c']]-11)/5
        testdf[['h']]=(testdf[['h']]-1)/0.1

    #This is the augmentation configuration we will use for training
    datagen=ImageDataGenerator(rescale=1./255.)#,
                               #shear_range=0.2,
                               #zoom_range=0.2,
                               #horizontal_flip=True,
                               #validation_split=0.25)
    
    #These are the generators that will read images found in 'data/l96/train'
    if j=='all':
        y_cols = ['b', 'c', 'h']
    elif j == 0:
        y_cols = ['b']
    elif j ==1:
        y_cols = ['c']
    elif j == 2:
        y_cols = ['h']
        
    if l96_var == 'XY':
        img_w = 20
    elif l96_var == 'Y':
        img_w = 16
    elif l96_var == 'Z':
        img_w = 4
        
    print('\ntrain_generator: ')
    train_generator=datagen.flow_from_dataframe(
                                dataframe=traindf,
                                directory=dir_train,
                                x_col="id",
                                y_col=y_cols,
                                #subset="training",
                                batch_size=128,
                                seed=42,
                                shuffle=True,
                                class_mode="other",
                                target_size=(20, img_w),
                                color_mode = 'grayscale')
    print('\nval_generator: ')
    val_generator=datagen.flow_from_dataframe(
                                dataframe=valdf,
                                directory=dir_val,
                                x_col="id",
                                y_col=y_cols,
                                #subset="validation",
                                batch_size=128,
                                seed=42,
                                shuffle=True,
                                class_mode="other",
                                target_size=(20,img_w),
                                color_mode='grayscale')
    print('\ntest_generator: ')                          
    test_generator=datagen.flow_from_dataframe(
                                dataframe=testdf,
                                directory=dir_test,
                                x_col="id",
                                y_col=y_cols,
                                #subset="validation",
                                batch_size=128,
                                seed=42,
                                shuffle=False,
                                class_mode="other",
                                target_size=(20,img_w),
                                color_mode='grayscale')
                                
    return train_generator, val_generator, test_generator


def r2(y_true, y_pred):
    import keras.backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def run_model(model_name, job_id, data_dir, N=0, j='all', weighted=True, num_epochs=10, pretrained=False):
    from models.cnn import cnn
#    from model.conv_lstm import conv_lstm
#    from model.resnet import make_resnet18
#    from model.transformer import transformer

    if model_name == 'cnn':
        model, model_out, model_params = cnn(job_id, data_dir, j, weighted, num_epochs, pretrained)
#    elif model_name =='conv_lstm':
#        model, model_out, model_params = conv_lstm(N, j, weighted, num_epochs)
#    elif model_name =='resnet':
#        model, model_out, model_params = make_resnet18(N, j,  num_epochs, pretrained=False)
#    elif model_name =='resnet_pretrained':
#        model, model_out, model_params = make_resnet18(N, j,  num_epochs, pretrained=True)
#    elif model_name == 'transformer':
#        model, model_out, model_params = transformer(N, j, num_epochs)
    return model, model_out, model_params


def add_to_log_exp9(job_id,  model_out, model_params, log='new'):    
    import pandas as pd
    if type(log)  == type('new'):
        log = pd.DataFrame(columns =['job_id', 'time_exp_mins',
                                     'model_name', 'l96_var', 'pretrained', 'best_job_id',
                                     'data_dir', 'test_mode', 'data_dir_test', 
                                     'target', 'weighted_loss',# 'normalized_target', 
                                     'num_epochs',
                                     'loss_train', 'loss_test', 'r2_train', 'r2_test'])
        
    else:    
        log = log.append({'job_id'       : job_id,
                          'time_exp_mins': model_out['time_exp'], 
                          'model_name'   : model_params['model_name'],
                          'l96_var'      : model_params['l96_var'],
                          'pretrained'   : model_params['pretrained'],
                          'data_dir'     : model_params['data_dir'],
                          'best_job_id'  : model_params['best_job_id'],
                          'test_mode'   : model_params['test_mode'],
                          'data_dir_test' : model_params['data_dir_test'],
                          #'N'            : model_params['N'],
                          #'input_shape'  : model_params['input_shape'],
                          'target'       : model_params['target'],
                          'weighted_loss': model_params['weighted'],
                          #'normalized_target': model_params['normalized_target'],
                          'num_epochs'   : model_params['num_epochs'],                          
                          'loss_train'   : model_out['loss_train'], 
                          'loss_test'    : model_out['loss_test'],
                          'r2_train'     : model_out['r2_train'],
                          'r2_test'      : model_out['r2_test']}, ignore_index=True)
    
    
    return log



 
