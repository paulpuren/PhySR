'''quadlinear for 3D GS'''

import torch
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

if __name__ == '__main__':
    # load test data
    data_dir = './model/3DGS/'
    data = torch.load(data_dir + 'output_0.pt')
    
    lres = data['lres'].numpy()  # [50,2,2,12,12,12], [t,b,c,d,h,w]
    hres = data['hres'].numpy()  # [100,2,2,48,48,48]
    pred_physr = data['pred'].numpy()
    
    # quadlinear on lres data, [t,b,c,d,h,w]
    # lres coords
    x = np.linspace(-50, 50, 12+1)
    x = x[:-1]
    # x_lres, y_lres, z_lres = np.meshgrid(x, x, x)
    t = np.linspace(0,100,50+1)
    t = t[:-1]
    u_lres = lres[:,0,0,...].ravel() # b=0, c=0
    v_lres = lres[:,0,1,...].ravel() # b=0, c=1
    
    # build the coordinates
    lres_coord = np.zeros((len(t), len(x), len(x), len(x), 4))
    lres_coord[...,0] = t.reshape((len(t),1,1,1))
    lres_coord[...,1] = x.reshape((1,len(x),1,1))
    lres_coord[...,2] = x.reshape((1,1,len(x),1))
    lres_coord[...,3] = x.reshape((1,1,1,len(x)))
    lres_coord = lres_coord.reshape((u_lres.size,4))
    
    # build the interpolator
    interp = LinearNDInterpolator(lres_coord, u_lres)   
                          
    # hes coords to be predicted
    X = np.linspace(-50, 50, 48+1)
    X = X[:-1]
    T = np.linspace(0, 100, 100+1)
    T = T[:-1]

    t_hres, x_hres, y_hres, z_hres = np.meshgrid(X,T,X,X)

    # build the interpolator
    u_hres = interp(t_hres, x_hres, y_hres, z_hres)
    torch.save({"pred": u_hres}, data_dir + '3DGS[quadlinear,u].pt')
    
    # v_lres
    interp = LinearNDInterpolator(lres_coord, v_lres)
    # build the interpolator
    v_hres = interp(t_hres, x_hres, y_hres, z_hres)
    torch.save({"pred": v_hres}, data_dir + '3DGS[quadlinear,v].pt')    
    
    # compare the point
    plt.figure()
    plt.plot(t, lres[:,0,0,6,6,6], label='lres')
    plt.plot(T, hres[:,0,0,24,24,24], label='hres')
    plt.plot(T, u_hres[:,24,24,24], label='hres_quadlinear')
    




