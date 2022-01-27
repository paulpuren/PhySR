''' Downsample the 2DGS dataset '''

import numpy as np
import scipy.io as scio
import pyrtools as pt

if __name__ == '__main__':

    # downsample factors
    s_ds, t_ds = 8, 4 
    n_levels = 3
    noise = 0.05
    
    print('Downsample the high-res data of 2D GS...')
    print('Downsample size in time and space are: %d, %d' % (t_ds, s_ds))
    
    # load data
    data = scio.loadmat('./2DGS_IC1_2x3001x256x256.mat')
    data = data['uv'] # [2,3001,256,256]       
    print('hres shape: ', data.shape)
    
    ds = np.zeros([2,751,32,32])
    for ii in range(ds.shape[1]):
        ds[0,ii,:,:] = pt.blurDn(data[0,t_ds*ii,:,:], n_levels, filt='binom5')
        ds[1,ii,:,:] = pt.blurDn(data[1,t_ds*ii,:,:], n_levels, filt='binom5')

    print('lres shape: ', ds.shape)
    scio.savemat('./2DGS_IC1_2x751x32x32.mat', {'uv': ds})  
        
