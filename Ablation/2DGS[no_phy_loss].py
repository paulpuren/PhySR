""" WDSR method for super-resolution with scientific data """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_feats, hidden_feats, input_kernel_size, input_stride, input_padding):
        super(ConvLSTMCell, self).__init__()

        self.hidden_feats = hidden_feats
        self.hidden_kernel_size = 3
        self.num_features = 4
        self.input_padding = input_padding
        self.padding = int((self.hidden_kernel_size - 1) / 2) # for the hidden state

        # input gate
        self.Wxi = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Whi = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')

        # forget gate
        self.Wxf = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Whf = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')

        # candidate gate
        self.Wxc = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Whc = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')

        # output gate
        self.Wxo = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Who = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')       

        # initialization
        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        
        return ch, cc

    def init_hidden_tensor(self, prev_state):

        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


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

        # temporal correction - convlstm
        for i in range(self.n_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                    input_feats=2,
                    hidden_feats=n_feats,
                    input_kernel_size=3,
                    input_stride=1,
                    input_padding=1) 

            setattr(self, name, cell)
            self._all_layers.append(cell)

        ################## spatial super-resolution ###################
        body = [ResBlock(n_feats, expansion_ratio=4, res_scale=0.1) for _ in range(self.n_resblock)]
        tail = [weight_norm(nn.Conv2d(n_feats, 2*(self.s_up_factor ** 2), 
            kernel_size=3, padding=1, padding_mode='circular')), nn.PixelShuffle(self.s_up_factor)]  

        skip = [weight_norm(nn.Conv2d(2, 2 * (self.s_up_factor ** 2), kernel_size=3, stride=1,
            padding=1, padding_mode='circular')), nn.PixelShuffle(self.s_up_factor)]    

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        # initialize weights
        self.apply(initialize_weights)

        # shiftmean
        self.shift_mean = ShiftMean(self.mean, self.std)    

    def forward(self, x, initial_state):
        # input: [t,b,c,h,w]
        internal_state = []
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

            # temporal correction
            for i in range(self.n_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    # output_size needs to be changed here
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state = initial_state[i])  
                    internal_state.append((h,c))
                
                # one-step forward
                (h, c) = internal_state[i]
                xt, new_c = getattr(self, name)(xt, h, c)
                internal_state[i] = (xt, new_c)  

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


def train(model, train_loader, val_loader, init_state, n_iters, lr, print_every, save_path, pretrain_flag=False):
    # train_loader: low resolution tensor

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
    L1_loss = nn.L1Loss()

    for epoch in range(n_iters):
        for idx, (lres, hres) in enumerate(train_loader):
            
            optimizer.zero_grad()

            lres, hres = lres.cuda(), hres.cuda() 
            lres, hres = lres.transpose(0,1), hres.transpose(0,1) # (b,t,c,h,w) -> (t,b,c,h,w)
            
            outputs = model(lres, init_state)

            # compute loss 
            loss = L1_loss(outputs, hres)
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

            # calculate the validation loss
            val_loss, val_error = validate(model, val_loader, init_state)

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


def validate(model, val_loader, init_state):
    ''' evaluate the model performance '''
    val_loss = 0
    val_error = 0
    MSE_function = nn.MSELoss()
    L1_loss = nn.L1Loss()

    for idx, (lres, hres) in enumerate(val_loader):

        lres, hres = lres.cuda(), hres.cuda() 
        lres, hres = lres.transpose(0,1), hres.transpose(0,1) # (b,t,c,h,w) -> (t,b,c,h,w)
        outputs = model(lres, init_state)
 
        # calculate the loss
        loss = L1_loss(outputs, hres)
        val_loss += loss.item()
        
        # calculate the error
        error = torch.sqrt(MSE_function(hres, outputs.detach()) / MSE_function(
            hres, torch.zeros_like(hres).cuda()))
        val_error += error.item()

    val_error = val_error / len(val_loader) 
    val_loss = val_loss / len(val_loader)

    return val_loss, val_error


def test(model, test_loader, init_state, save_path, fig_save_path):
    # load the well-trained model
    model_save_path = save_path + 'checkpoint.pt'
    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, 
        save_dir=model_save_path) 

    MSE_function = nn.MSELoss()
    pred_error = 0

    for idx, (lres, hres) in enumerate(test_loader):

        lres, hres = lres.cuda(), hres.cuda() 
        lres, hres = lres.transpose(0,1), hres.transpose(0,1) # (b,t,c,h,w) -> (t,b,c,h,w)
        outputs = model(lres, init_state)

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
    
    
def get_init_state(batch_size, hidden_channels, output_size, mode='coord'):
    '''initial hidden states for all convlstm layers'''
    # (b, c, h, w)

    num_layers = len(hidden_channels)
    initial_state = []
    if mode == 'coord':
        for i in range(num_layers):
            resolution = output_size[i][0]
            x, y = [np.linspace(-64, 64, resolution+1)] * 2
            x, y = np.meshgrid(x[:-1], y[:-1])  # [32, 32]
            xy = np.concatenate((x[None, :], y[None, :]), 0) # [2, 32, 32]
            xy = np.repeat(xy, int(hidden_channels[i]/2), axis=0) # [c,h,w]
            xy = np.repeat(xy[None, :], batch_size[i], 0) # [b,c,h,w]
            xy = torch.tensor(xy, dtype=torch.float32)
            initial_state.append((xy, xy))

    elif mode == 'zero':
        for i in range(num_layers):
            (h0, c0) = (torch.zeros(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]), torch.zeros(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]))
            initial_state.append((h0,c0))

    elif mode == 'random':
        for i in range(num_layers):
            (h0, c0) = (torch.randn(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]), torch.randn(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]))
            initial_state.append((h0,c0))
    else:
        raise NotImplementedError

    return initial_state


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
         
    save_path = './model/2DGS[no_phy_loss]/'
    fig_save_path = './figures/2DGS[no_phy_loss]/'
    print('Super-Resolution for 2D GS equation w/o phy loss...')

    model = PhySR(
        n_feats = 64,
        n_layers = [1, 2], # [n_convlstm, n_resblock]
        upscale_factor = [4, 8], # [t_up, s_up]
        shift_mean_paras = [mean_hres, std_hres],  
        step = steps,
        effective_step = effective_step).cuda()

    # define the initial states and initial output for model
    init_state = get_init_state(
        batch_size = [4], 
        hidden_channels = [64], 
        output_size = [[32, 32]],
        mode = 'random')

    start = time.time()
    train_loss_list, val_loss_list, val_error_list = train(model, train_loader, val_loader, 
        init_state, n_iters, learning_rate, print_every, save_path)
    end = time.time()
    print('The training time is: ', (end - start))
    print('')

    np.save(save_path + 'train_loss', train_loss_list)
    np.save(save_path + 'val_loss', val_loss_list)
    np.save(save_path + 'val_error', val_error_list)

    ###################### model inference ###########################
    pred_error = test(model, test_loader, init_state, save_path, fig_save_path)
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

