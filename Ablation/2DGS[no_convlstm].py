""" WDSR method for super-resolution with scientific data """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float32)


lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

def initialize_weights(module):

    c = 1
    if isinstance(module, nn.Conv2d):
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))

    if isinstance(module, nn.Conv1d):
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ShiftMean(nn.Module):
    # data: [t,b,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        self.mean = torch.Tensor(mean).view(1, 1, 2, 1, 1)
        self.std = torch.Tensor(std).view(1, 1, 2, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.cuda()) / self.std.cuda()
        elif mode == 'add':
            return x * self.std.cuda() + self.mean.cuda()
        else:
            raise NotImplementedError


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=0.1):
        super(ResBlock, self).__init__()

        self.res_scale = res_scale
        self.conv1 = weight_norm(nn.Conv2d(n_feats, n_feats*expansion_ratio, kernel_size=3, 
            padding=1, padding_mode='circular'))
        self.conv2 = weight_norm(nn.Conv2d(n_feats*expansion_ratio, n_feats, kernel_size=3, 
            padding=1, padding_mode='circular'))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        s = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = s + self.res_scale * x

        return x

class temporal_sr(nn.Module):
    def __init__(self, t_upscale_factor):
        super(temporal_sr, self).__init__()

        self.t_upscale_factor = t_upscale_factor

    def forward(self, x):  

        t, b, c, h, w = x.shape  
        x = x.permute(1,3,4,2,0) # [b,h,w,c,t]
        x = x.contiguous().view(b*h*w, c, t)

        x = F.interpolate(x, scale_factor=self.t_upscale_factor, mode='linear', align_corners=True)   
        
        x = x.contiguous().view(b, h, w, c, t*self.t_upscale_factor)
        x = x.permute(4,0,3,1,2) # [t,b,c,h,w]

        return x


class PhySR(nn.Module):
    def __init__(self, n_feats, n_layers, upscale_factor, shift_mean_paras, step=1, effective_step=[1]):

        super(PhySR, self).__init__()
        # n_layers: [n_convlstm, n_resblock]

        self.n_convlstm, self.n_resblock = n_layers
        self.t_up_factor, self.s_up_factor = upscale_factor
        self.mean, self.std = shift_mean_paras
        
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []

        ################## temporal super-resolution ###################
        # temporal interpolation
        self.tsr = temporal_sr(self.t_up_factor)

        # change convlstm to conv head
        head = [weight_norm(nn.Conv2d(2, n_feats, kernel_size=3, padding=1, padding_mode='circular'))]

        ################## spatial super-resolution ###################
        body = [ResBlock(n_feats, expansion_ratio=4, res_scale=0.1) for _ in range(self.n_resblock)]
        tail = [weight_norm(nn.Conv2d(n_feats, 2*(self.s_up_factor ** 2), 
            kernel_size=3, padding=1, padding_mode='circular')), nn.PixelShuffle(self.s_up_factor)]  

        skip = [weight_norm(nn.Conv2d(2, 2 * (self.s_up_factor ** 2), kernel_size=3, stride=1,
            padding=1, padding_mode='circular')), nn.PixelShuffle(self.s_up_factor)]    

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        # initialize weights
        self.apply(initialize_weights)

        # shiftmean
        self.shift_mean = ShiftMean(self.mean, self.std)    

    def forward(self, x):
        # input: [t,b,c,h,w]
        outputs = []
        
        # normalize
        x = self.shift_mean(x, mode='sub')
            
        # temporal super-resolution
        x = self.tsr(x) 

        for step in range(self.step):
            # input:[t,b,c,h,w]
            xt = x[step,...]

            # skip connection
            s = self.skip(xt)

            # change convlstm to conv
            xt = self.head(xt)

            # spatial super-resolution
            xt = self.body(xt)
            xt = self.tail(xt)

            # residual connection
            xt += s
            xt = xt.view(1, 4, 2, 256, 256)
            
            if step in self.effective_step:
                outputs.append(xt)    

        outputs = torch.cat(tuple(outputs), dim=0)
        outputs = self.shift_mean(outputs, mode='add')

        return outputs


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class LossGenerator(nn.Module):
    '''Calculate the physical loss and the data loss'''

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
       
        super(LossGenerator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        # forward/backward derivative operator 
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1/2, 0, 1/2]]],
            resol = (dt),
            kernel_size = 3,
            name = 'partial_t').cuda() 

        self.fwd_dt = Conv1dDerivative(
            DerFilter = [[[-3/2, 2, -1/2]]],
            resol = (dt),
            kernel_size = 3,
            name = 'forward_partial_t').cuda() 
 
        self.bwd_dt = Conv1dDerivative(
            DerFilter = [[[1/2, -2, 3/2]]],
            resol = (dt),
            kernel_size = 3,
            name = 'backward_partial_t').cuda() 

    def GetPhyLoss(self, output):
        '''Calculate the physical loss'''
        # output: [t,b,c,h,w]

        ############### spatial derivatives #################
        # laplace u, [t-2,b,c,h,w]
        u = output[:, :, 0:1, :, :]
        len_t,len_b,len_c,len_h,len_w = u.shape 
        # [t,b,c,h,w] -> [t*b,c,h,w]
        u = u.reshape(len_t*len_b, len_c, len_h, len_w)
        laplace_u = self.laplace(u)  
        # change batch to [t,b,c,h,w]
        laplace_u = laplace_u.reshape(len_t,len_b,len_c,len_h-4,len_w-4)

        # laplace v, [t-2,b,c,h,w]
        v = output[:, :, 1:2, :, :]
        len_t,len_b,len_c,len_h,len_w = v.shape 
        v = v.reshape(len_t*len_b, len_c, len_h, len_w)
        laplace_v = self.laplace(v)  
        laplace_v = laplace_v.reshape(len_t,len_b,len_c,len_h-4,len_w-4)

        ############### temporal derivatives #################
        # u_t, [t,b,c,h-4,w-4]
        u = output[:, :, 0:1, 2:-2, 2:-2]
        len_t,len_b,len_c,len_h,len_w = u.shape 
        u = u.permute(3,4,1,2,0) # [h,w,b,c,t]
        u = u.reshape(len_h*len_w*len_b, len_c, len_t) # [h*w*b,c,t]
        u_t = self.dt(u) # [h*w*b,c,t-2]
        u_t0 = self.fwd_dt(u[:,:,0:3])
        u_tn = self.bwd_dt(u[:,:,-3:])
        u_t = torch.cat((u_t0,u_t,u_tn), dim=2) # [h*w*b,c,t]
        u_t = u_t.reshape(len_h, len_w, len_b, len_c, len_t)
        u_t = u_t.permute(4,2,3,0,1)

        # v_t, [t,b,c,h-4,w-4]
        v = output[:, :, 1:2, 2:-2, 2:-2]
        len_t,len_b,len_c,len_h,len_w = v.shape 
        v = v.permute(3,4,1,2,0) # [h,w,b,c,t]
        v = v.reshape(len_h*len_w*len_b, len_c, len_t) # [h*w*b,c,t]
        v_t = self.dt(v)
        v_t0 = self.fwd_dt(v[:,:,0:3])
        v_tn = self.bwd_dt(v[:,:,-3:])
        v_t = torch.cat((v_t0, v_t, v_tn), dim=2)
        v_t = v_t.reshape(len_h, len_w, len_b, len_c, len_t)
        v_t = v_t.permute(4,2,3,0,1)

        ############### corresponding u & v ###################
        u = output[:, :, 0:1, 2:-2, 2:-2]  # [step, b, c, height(Y), width(X)]
        v = output[:, :, 1:2, 2:-2, 2:-2]  # [step, b, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # 2D GS
        Du, Dv = 0.16, 0.08
        f, k = 0.06, 0.062
        f_u = Du*laplace_u - u*v**2 + f*(1-u) - u_t
        f_v = Dv*laplace_v + u*v**2 - (f+k)*v - v_t

        return f_u, f_v


    def GetModelLoss(self, model):
        ''' Get the L2-norm of the model '''

        l2_reg = torch.tensor(0.).cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)

        return l2_reg


def LossGen(output, truth, beta, loss_func):

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    # data loss
    data_loss = L1_loss(output, truth)

    # phy loss, output shape: [t,b,c,h,w]
    output = torch.cat((output[:, :, :, :, -2:], output, output[:, :, :, :, 0:3]), dim=4)
    output = torch.cat((output[:, :, :, -2:, :], output, output[:, :, :, 0:3, :]), dim=3)
    
    f_u, f_v = loss_func.GetPhyLoss(output)
    phy_loss = MSE_loss(f_u, torch.zeros_like(f_u).cuda()) + MSE_loss(
                f_v, torch.zeros_like(f_v).cuda())

    loss = data_loss + beta * phy_loss 

    return loss, data_loss, phy_loss 


def train(model, train_loader, val_loader, n_iters, lr, print_every, dt, dx, 
    beta, save_path, pretrain_flag=False):
    # train_loader: low resolution tensor
    # beta works on physics loss

    best_error = 1e2
    print_loss_total = 0
    train_loss_list, val_loss_list, val_error_list = [], [], []
    pretrain_save_path = save_path + 'pretrain.pt'
    model_save_path = save_path + 'checkpoint.pt'

    if pretrain_flag == True:
        model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, 
            save_dir=pretrain_save_path) 

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.998) 
    loss_function = LossGenerator(dt, dx)

    for epoch in range(n_iters):
        for idx, (lres, hres) in enumerate(train_loader):
            
            optimizer.zero_grad()

            lres, hres = lres.cuda(), hres.cuda() 
            lres, hres = lres.transpose(0,1), hres.transpose(0,1) # (b,t,c,h,w) -> (t,b,c,h,w)
            
            outputs = model(lres)

            # compute loss 
            loss, data_loss, phy_loss = LossGen(outputs, hres, beta, loss_function)
            loss.backward(retain_graph=True)
            print_loss_total += loss.item()

            # gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            optimizer.step()
            scheduler.step()

        if (epoch+1) % print_every == 0:
            # calculate the average training loss
            print_loss_mean = print_loss_total / (print_every*len(train_loader))
            train_loss_list.append(print_loss_mean)
            print_loss_total = 0

            # print the training loss
            print('Train loss (%d/%d %d%%): %.8f'  % (epoch+1, n_iters, 
                (epoch+1)/n_iters*100, print_loss_mean))

            # for print training loss (details)
            print('Epoch %d: data loss(%.8f), phy loss(%.8f)' %(
                epoch+1, data_loss.item(), phy_loss.item()))

            # calculate the validation loss
            val_loss, val_error = validate(model, val_loader, loss_function, beta)

            # for print validation loss
            print('Epoch (%d/%d %d%%): val loss %.8f, val error %.8f'  % (epoch+1, n_iters, 
                (epoch+1)/n_iters*100, val_loss, val_error))
            print('')

            val_loss_list.append(val_loss)
            val_error_list.append(val_error)

            # save model
            if val_error < best_error:
                save_checkpoint(model, optimizer, scheduler, model_save_path)
                best_error = val_error

    return train_loss_list, val_loss_list, val_error_list


def validate(model, val_loader, loss_function, beta):
    ''' evaluate the model performance '''
    val_loss = 0
    val_error = 0
    MSE_function = nn.MSELoss()

    for idx, (lres, hres) in enumerate(val_loader):

        lres, hres = lres.cuda(), hres.cuda() 
        lres, hres = lres.transpose(0,1), hres.transpose(0,1) # (b,t,c,h,w) -> (t,b,c,h,w)
        outputs = model(lres)
 
        # calculate the loss
        loss,_,_ = LossGen(outputs, hres, beta, loss_function)
        val_loss += loss.item()
        
        # calculate the error
        error = torch.sqrt(MSE_function(hres, outputs.detach()) / MSE_function(
            hres, torch.zeros_like(hres).cuda()))
        val_error += error.item()

    val_error = val_error / len(val_loader) 
    val_loss = val_loss / len(val_loader)

    return val_loss, val_error


def test(model, test_loader, save_path, fig_save_path):
    # load the well-trained model
    model_save_path = save_path + 'checkpoint.pt'
    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, 
        save_dir=model_save_path) 

    MSE_function = nn.MSELoss()
    pred_error = 0

    for idx, (lres, hres) in enumerate(test_loader):

        lres, hres = lres.cuda(), hres.cuda() 
        lres, hres = lres.transpose(0,1), hres.transpose(0,1) # (b,t,c,h,w) -> (t,b,c,h,w)
        outputs = model(lres)

        # calculate the error
        error = torch.sqrt(MSE_function(hres, outputs.detach()) / MSE_function(
            hres, torch.zeros_like(hres).cuda()))
        pred_error += error.item()

        torch.save({"pred": outputs.detach().cpu(), "lres": lres.cpu(), 
            "hres": hres.cpu()}, save_path + 'output_'+str(idx)+'.pt')

        # comparison plot
        t = np.arange(hres.shape[0])
        for b in range(hres.shape[1]):
            u_pred = outputs[:, b, 0, :, :].detach().cpu().numpy()
            u_true = hres[:, b, 0, :, :].cpu().numpy() 

            plt.figure()
            plt.plot(t, u_pred[:, 66, 66], label = 'u-wdsr')
            plt.plot(t, u_true[:, 66, 66], label = 'u-Ref.')

            plt.xlabel('t')
            plt.ylabel('u')
            plt.legend()
            plt.savefig(fig_save_path + 'u_comp_[i=%d][b=%d].png' %(idx, b))

    pred_error = pred_error/len(test_loader)

    return pred_error


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler
    

class GSDataset(Dataset):
    def __init__(self, data_dir, data_fname, ICs):
        '''
        Args:
        -----
        data_dir: str,
            folder path to the data    
        data_fname: str
            the name of the dataset file
        ICs: list
            the list of random noise parameters
        '''

        self.data_dir = data_dir
        self.data_fname = data_fname
        self.ICs = ICs
        
        self.samples = []      

        for i in range(len(self.ICs)):     
            # define the data filename
            lres_filename = self.data_fname + str(ICs[i]) + '_2x751x32x32.mat'
            hres_filename = self.data_fname + str(ICs[i]) + '_2x3001x256x256.mat'

            # load the lres and hres tensor, (c,t,h,w) -> (t,c,h,w)
            lres = scio.loadmat(os.path.join(data_dir, lres_filename))
            hres = scio.loadmat(os.path.join(data_dir, hres_filename))            
            lres = lres['uv'][:,50::2,...] # [2,350,32,32]
            hres = hres['uv'][:,200::2,...] # [2,1400,256,256]
            lres = np.transpose(lres, (1,0,2,3))
            hres = np.transpose(hres, (1,0,2,3))

            # segment 1
            lres_tensor = torch.tensor(lres[100:125,...], dtype=torch.float32)
            hres_tensor = torch.tensor(hres[400:500,...], dtype=torch.float32)
            self.samples.append((lres_tensor, hres_tensor))            

            # segment 2
            lres_tensor = torch.tensor(lres[150:175,...], dtype=torch.float32)
            hres_tensor = torch.tensor(hres[600:700,...], dtype=torch.float32)
            self.samples.append((lres_tensor, hres_tensor)) 

            # segment 3
            lres_tensor = torch.tensor(lres[200:225,...], dtype=torch.float32)
            hres_tensor = torch.tensor(hres[800:900,...], dtype=torch.float32)
            self.samples.append((lres_tensor, hres_tensor)) 
            
            # segment 4
            lres_tensor = torch.tensor(lres[250:275,...], dtype=torch.float32)
            hres_tensor = torch.tensor(hres[1000:1100,...], dtype=torch.float32)
            self.samples.append((lres_tensor, hres_tensor)) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

if __name__ == '__main__':
    # define the data file path 
    data_dir = './data/2DGS_256x256/'
    data_fname = '2DGS_IC'

    # define the initial conditions    
    ICs = np.arange(1,11)
    data_loader = GSDataset(data_dir, data_fname, ICs)
    n_datasets = data_loader.__len__()

    # get mean and std
    data = data_loader[0][1]
    total_hres = torch.zeros(40, data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    total_lres = torch.zeros(40, 25, 2, 32, 32) # [b,t,c,h,w]

    for i in range(len(data_loader)):
        total_hres[i,...] = data_loader[i][1]
        total_lres[i,...] = data_loader[i][0]

    mean_hres = torch.mean(total_hres, axis = (0,1,3,4)) 
    std_hres = torch.std(total_hres, axis = (0,1,3,4))

    # split data
    split_ratio = [int(n_datasets*0.7), int(n_datasets*0.2), int(n_datasets*0.1)]
    train_data, val_data, test_data = torch.utils.data.random_split(data_loader, split_ratio)
    
    # change to pytorch data
    # data in train_loader is [b, t, c, h, w] -> [1, 151, 2, 32, 32]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 4, 
        shuffle=True, num_workers=0) 

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = 4, 
        shuffle=False, num_workers=0)    

    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 4, 
        shuffle=False, num_workers=0)

    ######################### build model #############################
    # training parameters
    n_iters = 2000 
    learning_rate = 1e-3
    print_every = 2
    dt = 1.0*5*2
    dx = 1.0  
    steps = 100 
    effective_step = list(range(0, steps))
    
    beta = 0.025 # for physics loss        
    save_path = './model/2DGS[no_convlstm]/'
    fig_save_path = './figures/2DGS[no_convlstm]/'
    print('Super-Resolution for 2D GS equation w/o convlstm...')

    model = PhySR(
        n_feats = 64,
        n_layers = [1, 2], # [n_convlstm, n_resblock]
        upscale_factor = [4, 8], # [t_up, s_up]
        shift_mean_paras = [mean_hres, std_hres],  
        step = steps,
        effective_step = effective_step).cuda()

    start = time.time()
    train_loss_list, val_loss_list, val_error_list = train(model, train_loader, val_loader, 
        n_iters, learning_rate, print_every, dt, dx, beta, save_path)
    end = time.time()
    print('The training time is: ', (end - start))
    print('')

    np.save(save_path + 'train_loss', train_loss_list)
    np.save(save_path + 'val_loss', val_loss_list)
    np.save(save_path + 'val_error', val_error_list)

    ###################### model inference ###########################
    pred_error = test(model, test_loader, save_path, fig_save_path)
    print('The predictive error is: ', pred_error)
    print('Test completed')

    # plot loss
    x_axis = np.arange(0, n_iters, print_every)
    plt.figure()
    plt.plot(x_axis, train_loss_list, label = 'train loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'train loss.png', dpi = 300)

    plt.figure()
    plt.plot(x_axis, val_loss_list, label = 'val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'val loss.png', dpi = 300)

    plt.figure()
    plt.plot(x_axis, val_error_list, label = 'val error')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'val error.png', dpi = 300)

