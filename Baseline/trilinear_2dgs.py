'''trilinear for 2D GS'''

import torch
import torch.nn as nn
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

if __name__ == '__main__':
    # load test data
    data_dir = './model/2DGS/'
    data = torch.load(data_dir + 'output_IC10.pt')
    lres = data['lres'].cuda() # [25,4,2,32,32], [t,b,c,h,w]
    hres = data['hres'] # [100,4,2,256,256]
    pred_physr = data['pred'].numpy()
    
    # trilinear on lres data, [t,b,c,h,w] -> [b,c,d,h,w], [b,c,t,h,w]
    lres = lres.permute(1,2,0,3,4)
    pred_trilinear = nn.functional.interpolate(lres, scale_factor=[4,8,8], mode='trilinear', align_corners=True).cuda()
    pred_trilinear = pred_trilinear.permute(2,0,1,3,4) # [b,c,t,h,w] -> [t,b,c,h,w]
    
    # save trilinear results
    torch.save({"pred": pred_trilinear.cpu()}, data_dir + '2DGS[IC10,trilinear].pt')    

    # compute error
    hres = hres.numpy()
    pred_trilinear = pred_trilinear.cpu().numpy()
    pred_error = frobenius_norm(pred_trilinear - hres) / frobenius_norm(hres)
    print('The predictive error of trilinear is: ', pred_error)

#    pred_error = frobenius_norm(pred_physr - hres) / frobenius_norm(hres)
#    print('The predictive error of PhySR is: ', pred_error)    





