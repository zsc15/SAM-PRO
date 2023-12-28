import os
import numpy as np
import hdf5storage
import argparse
import cv2
import matlab
import matlab.engine
import torch
import scipy.io as scio
import torch.nn as nn
import math
import logging
from PnP_restoration.utils.utils_restoration import single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from scipy import ndimage
from PIL import Image
from PnP_restoration.utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array
from PnP_restoration.utils import utils_sr
from utils.utils import load_model
from utils.config import analyze_parse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Super-resolution', help='image deblurring or super-resolution')###Gaussian deblurring; Uniform deblurring; Super-resolution
parser.add_argument('--scale', type=int, default=3, help='image scale')
parser.add_argument('--algo', type=str, default='SAM_PROv2', help='algorithms')
parser.add_argument('--noise_level', type=float, default=5, help='noise level of image')
args = parser.parse_args()
def initialize_prox(img, degradation_mode, degradation, sf, device):
    if degradation_mode == 'deblurring':
        k = degradation
        k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        FB, FBC, F2B, FBFy = utils_sr.pre_calculate_prox2(img, k_tensor, sf)
        return FB, FBC, F2B, FBFy, k_tensor
    elif degradation_mode == 'SR':
        k = degradation
        k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        FB, FBC, F2B, FBFy = utils_sr.pre_calculate_prox2(img,k_tensor, sf)
        return FB, FBC, F2B, FBFy, k_tensor
    elif degradation_mode == 'inpainting':
        M = array2tensor(degradation).double().to(device)
        My = M*img
        return My
    else:
        print('degradation mode not treated')

def calulate_data_term(k_tensor,degradation_mode, sf,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
#         k_tensor = array2tensor(np.expand_dims(k, 2)).double().to(device)
        if degradation_mode == 'deblurring':
            deg_y = utils_sr.imfilter(y.double(), k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif degradation_mode == 'SR':
            deg_y = utils_sr.imfilter(y.double(), k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            deg_y = deg_y[..., 0::sf, 0::sf]
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
#         elif degradation_mode == 'inpainting':
#             deg_y = M * y.double()
#             f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        else:
            print('degradation not implemented')
        return f
        
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def single2uint(img):
    return np.uint8(img*255.)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def tensor2uint2(img):
    img = img.data.squeeze().float().cpu().numpy()
    img = norm_proj(img)
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def tensor2float(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def tensor2float2(img):
    img = img.data.squeeze().float().cpu().numpy()
    img = norm_proj(img)
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img
    
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 /np.sqrt(mse))

def calculate_grad(img, degradation_mode, FB, FBC,FBFy,sf=1):
    if degradation_mode == 'deblurring':
        grad = utils_sr.grad_solution2(img.double(), FB, FBC, FBFy, 1)
    if degradation_mode == 'SR' :
        grad = utils_sr.grad_solution2(img.double(), FB, FBC, FBFy, sf)
    return grad

def load_model(model_type, sigma,device):
    path = "Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    if model_type == "DnCNN":
        from model.models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda(device.index)
    elif model_type == "SimpleCNN":
        from model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True).cuda(device.index)
    elif model_type == "RealSN_DnCNN":
        from model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()
    elif model_type == "RealSN_SimpleCNN":
        from model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True).cuda(device.index)
    else:
        from model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda(device.index)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def CalMATLAB(IRFolder,GTFolder):
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(eng.fullfile(os.getcwd(),'Metrics')))
    res = eng.evaluate_PSNR(IRFolder,GTFolder)
    res=np.array(res)
    return res

def CalMATLAB2(IRFolder,one_gt_name):
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(eng.fullfile(os.getcwd(),'Metrics')))
    res = eng.evaluate_PSNR_comparison(IRFolder,one_gt_name)
    res=np.array(res)
    return res

def MATLAB_imresize(imgname, Sf=3):
    eng = matlab.engine.start_matlab()
    x = eng.load(imgname)['data']
    res = eng.imresize(x, Sf)
    res=np.array(res)
    return res

def MATLAB_imresize2(imgname, Sf=3):
    eng = matlab.engine.start_matlab()
    x = eng.imread(imgname)
    res = eng.imresize(x, Sf)
    res=np.array(res)
    return res


def MATLAB_degradation(imgname, Sf=3):
    eng = matlab.engine.start_matlab()
    x = eng.imread(imgname)
    x = matlab.double(x)
#     eng.addpath(eng.genpath(eng.fullfile(os.getcwd(),'Metrics')))
    res = eng.imresize(x, Sf)
    res=np.array(res)
    return res


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def norm_proj(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    return x

def imsave(img, img_path):
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
    
def numpy_degradation(x, k, sf=3):
    ''' blur + downsampling
    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double, positive
        sf: down-scale factor
    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    st = 0
    return x[st::sf, st::sf, ...]
##################################算法部分##################################
def SAM_PRO_v2(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    mu_0 = opt['mu_0']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    for i in range(K):
        x = x0
        mu_k = mu_0*(i+1)**(-1.0)
        mu = mu_k if mu_k<1 else 1
        f_est = x
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z= xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        z = (1-beta)*f_est+beta*z
        grad =  calculate_grad(z, degradation_mode, FB, FBC, FBFy, sf=Sf)
        v_est = z - mu*grad/sigma
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        obj_fun[i] = calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()

def RED(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    lambdaa = opt['lambda']
    sigma = opt['sigma']
    input_sigma = opt['input_sigma']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    mu = 0.1####### 0.1 for parrots(sacle =3, super-resolution); 1 for butterfly(sigma=8, uniform PSF)
    for i in range(K):
        x = x0
        grad1 =  calculate_grad(x, degradation_mode, FB, FBC, FBFy, sf=Sf)/sigma
        f_est = x 
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z = xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        grad2 =  (x - z)
        v_est = x - mu*(grad1/sigma+lambdaa*grad2)
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        x_square = torch.norm(x0, p=2)**2
        obj_fun[i] = 1/sigma*calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()/x_square.float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()

def REDPRO(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    lambdaa = opt['lambda']
    sigma = opt['sigma']
    input_sigma = opt['input_sigma']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    mu0 = 2/(1/(input_sigma**2) + lambdaa)
    for i in range(K):
        x = x0
        mu =2*(i+1)**(-0.1)################# 2 for parrots(sacle =3, super-resolution); 4 for butterfly(sigma=8, uniform PSF)
        grad1 =  calculate_grad(x, degradation_mode, FB, FBC, FBFy, sf=Sf)
        f_est = x - mu*grad1/sigma
#         f_est = x
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z = xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        v_est = beta*z+(1-beta)*f_est
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        x_square = torch.norm(x0, p=2)**2
        obj_fun[i] = 1/sigma*calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()/x_square.float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()

def PnP_FBS(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    lambdaa = opt['lambda']
    sigma = opt['sigma']
    input_sigma = opt['input_sigma']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    mu0 = 4
    for i in range(K):
        x = x0
        mu = mu0################# 2 for parrots(sacle =3, super-resolution); 4 for butterfly(sigma=8, uniform PSF)
        grad1 =  calculate_grad(x, degradation_mode, FB, FBC, FBFy, sf=Sf)
        f_est = x - mu*grad1/sigma
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z = xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        v_est = beta*z+(1-beta)*x
        w = v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        x_square = torch.norm(x0, p=2)**2
        obj_fun[i] = 1/sigma*calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()/x_square.float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()


def SAM_PRO_v1(x0,y, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt):
    K = opt['K']
    alpha = opt['alpha']
    beta = opt['beta']
    sigma = opt['sigma']
    mu_0 = opt['mu_0']
    Sf = opt['sf']
    obj_fun = np.zeros(K)
    residual = torch.zeros(K)
    for i in range(K):
        x = x0
        mu_k = mu_0*(i+1)**(-1.0)
        mu = mu_k if mu_k<1 else 1
        f_est = x
        mintmp = torch.min(f_est)
        maxtmp = torch.max(f_est)
        xtilde = (f_est - mintmp) / (maxtmp - mintmp)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift
        r = f(xtilde.float())
        z= xtilde - r
        z= (z - scale_shift) / scale_range
        z = z * (maxtmp - mintmp) + mintmp
        z = (1-beta)*f_est+beta*z
        grad =  calculate_grad(f_est, degradation_mode, FB, FBC, FBFy, sf=Sf)
        v_est = f_est - alpha*grad/sigma
        w = (1-mu)*z+mu*v_est
        residual[i] = torch.norm(w-x0, p=2)
        x0 = w
        obj_fun[i] = calulate_data_term(k_tensor,degradation_mode, Sf,x0,y).float().cpu().numpy()
    return x0, obj_fun, residual.float().cpu().numpy()

def tensor_to_mat(imgname,img_name, x1, setting='DnCNN'):
    x1_out = tensor2float(x1)
    x_est_luma = single2uint(x1_out)
    IR_filename = 'PSNR_mat/'+img_name+setting+str(noise_level)+'_Gaussian_blur_img_mat/'
    if not os.path.exists(IR_filename):
        mkdir(IR_filename)
    scio.savemat(IR_filename+imgname+'_luma.mat', {'data':x_est_luma})

if __name__ == '__main__':
    kernel_path = os.path.join('PnP_restoration/kernels', 'Levin09.mat')
    kernels = hdf5storage.loadmat(kernel_path)['kernels']
    test_imgs = ['bike','butterfly', 'flower', 'girl', 'hat']#'parrots'###bike, butterfly,flower, girl, hat,parrots; 'butterfly', 'flower', 'girl', 'hat'
    img_type = 'RGB'
    task = args.task
    noise_level = args.noise_level
    algorithm = args.algo
    if task == 'Uniform-deblurring': # Uniform blur
        k = (1/81)*np.ones((9,9))
        Sf = 1
    elif task == 'Gaussian-deblurring':  # Gaussian blur
        k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
        Sf = 1
    elif task == 'Super-resolution':  # Gaussian blur
        k = matlab_style_gauss2D(shape=(7,7),sigma=1.6)
        Sf = args.scale
    else: # Motion blur
        k = kernels[0, k_index]
    for name in test_imgs:
        if img_type == 'RGB':
            input_im_uint = imread_uint('test_images/'+name+'.tif',n_channels=3)
        #     input_im_uint = imread_uint('datasets/set12/05.png', n_channels=3)
        else:
            input_im_uint = imread_uint('test_images/'+name+'.tif',n_channels=1)
        #     input_im_uint = imread_uint('datasets/set12/05.png', n_channels=1)
        input_im = np.float32(input_im_uint / 255)
                    # Degrade image
        if Sf>1:#super-resolution
            blur_im = numpy_degradation(input_im, k, sf=Sf)
        else:# debluring
            blur_im = ndimage.filters.convolve(input_im, np.expand_dims(k, axis=2), mode='wrap')
        np.random.seed(seed=0)
        noise = np.random.normal(0, noise_level / 255., blur_im.shape)
        blur_im += noise
        init_im = blur_im
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        init_im1 = rgb2ycbcr(norm_proj(init_im))
        init_im2 = np.expand_dims(init_im1, axis=2)
        img_tensor = array2tensor(init_im2).to(device)
        if img_type == 'RGB' and Sf==1:
            degradation_mode = 'deblurring'
            x0 = img_tensor#初值
        elif Sf>1:
            degradation_mode = 'SR'
            x0 = cv2.resize(init_im1, (init_im1.shape[1] * Sf, init_im1.shape[0] * Sf),interpolation=cv2.INTER_CUBIC)
            x0 = np.expand_dims(x0, axis=2)
            x0 = utils_sr.shift_pixel(x0, Sf)
            x0 = array2tensor(x0).to(device)
        else:
            img_tensor = array2tensor(init_im).to(device)
        FB, FBC, F2B, FBFy, k_tensor = initialize_prox(img_tensor, degradation_mode, k, Sf, device)
        ################################################algorithm setting####################################################################
        if noise_level<=5.0:
            sigma_f=5
        elif 15 >= noise_level>5:
            sigma_f=15
        elif 40 >= noise_level> 15:
            sigma_f=25
        else:
            print('error')
        model_type = 'DnCNN' #'RealSN_DnCNN'|'RealSN_SimpleCNN'
        ####################################################################################################################################
        with torch.no_grad():
            if algorithm == 'SAM_PROv1':###S(x) = x-s\nabla f(x)
                opt_r={'alpha':4.1, 'beta':0.1, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':2000, 'mu_0':500, 'sf': Sf}
                f = load_model(model_type, int(opt_r['sigma_f']), device)
                x1, objfun,r = SAM_PRO_v1(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt_r)
            elif algorithm == 'PnP_FBS':###S(x) = x-s\nabla f(x)
                opt_PnP_FBS={'alpha':2.4, 'beta':0.01, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':2000, 'lambda':0.01,'input_sigma':noise_level**2, 'sf': Sf}
                f = load_model(model_type, int(opt_PnP_FBS['sigma_f']), device)
                x1, objfun, r = PnP_FBS(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt_PnP_FBS)
            elif algorithm == 'SAM_PROv2':###S(x) = Tx-s\nabla f(Tx)
                opt={'alpha':4, 'beta':0.01, 'sigma':noise_level, 'sigma_f':sigma_f, 'K':2000, 'mu_0':500, 'sf': Sf}
                f = load_model(model_type, int(opt['sigma_f']), device)
                x1, objfun,r = SAM_PRO_v2(x0,img_tensor, k_tensor, degradation_mode, FB, FBC, FBFy, f, opt)
            else:
                print('algorithm not implemented ^_^')
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath(eng.fullfile(os.getcwd(),'Metrics')))
        x_gt_luma = rgb2ycbcr(input_im_uint)
        im_ycbr = rgb2ycbcr(norm_proj(init_im), only_y=False)
        im_ycbr2 = rgb2ycbcr(norm_proj(init_im), only_y=False)
        x1_out = tensor2float(x1)
        x_est_luma = single2uint(x1_out)
        H, W, _ = input_im_uint.shape
        h, w = x_est_luma.shape
        x_est_luma = x_est_luma[1:h-1,1:w-1] if W==256 and H==256 else x_est_luma########keep the same size
        PSNR = eng.ComputePSNR(matlab.uint8(x_gt_luma.tolist()), matlab.uint8(x_est_luma.tolist()))###Fllowed by RED paper
        print('{} - PSNR: {:.4f} dB'.format(name, PSNR))