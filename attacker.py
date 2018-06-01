'''
FGSM attack on student model
'''
import MCS2018
import os
import time
import argparse
from torch import optim
#from differential_evolution import differential_evolution
import numpy as np
import pandas as pd
import torch
# for pytorch 3-4 compatibility
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from skimage.measure import compare_ssim
from scipy.optimize import differential_evolution
from skimage.measure import compare_ssim
from student_net_learning.models import *
#import pytorch_ssim

SSIM_THR = 0.95

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

REVERSE_MEAN = [-0.485, -0.456, -0.406]
REVERSE_STD = [1/0.229, 1/0.224, 1/0.225]

parser = argparse.ArgumentParser(description='PyTorch student network training')

parser.add_argument('--root', 
                    required=True,
                    type=str, 
                    help='data root path')
parser.add_argument('--save_root',
                    required=True,
                    type=str,
                    help='path to store results',
                    default='./changed_imgs')
parser.add_argument('--datalist', 
                    required=True,
                    type=str, 
                    help='datalist path')
parser.add_argument('--model_name',
                    type=str, 
                    help='model name', 
                    default='ResNet18')
parser.add_argument('--checkpoint_path',
                    required=True,
                    type=str,
                    help='path to learned student model checkpoints')
parser.add_argument('--cuda',
                    action='store_true', 
                    help='use CUDA')
parser.add_argument('--eps', 
                    type=str, 
                    default='1e-2',
                    help='eps for image noise')
parser.add_argument('--attack_type', 
                    type=str, 
                    default='FGSM',
                    help='attacker type')
parser.add_argument('--attack_mode', 
                    type=str, 
                    default='begin',
                    help='mode: if we attack from begin or previously attacked images')

args = parser.parse_args()

def reverse_normalize(tensor, mean, std):
    '''reverese normalize to convert tensor -> PIL Image'''
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.div_(s).sub_(m)
    return tensor_copy

def get_model(model_name, checkpoint_path):
    '''
    Model architecture choosing
    '''
    if model_name == 'ResNet18':
        net = ResNet18()
    elif model_name == 'ResNet34':
        net = ResNet34()
    elif model_name == 'ResNet50':
        net = ResNet50()
    elif model_name == 'ResNet152':
        net = ResNet152()
    elif model_name == 'DenseNet':
        net = DenseNet121()
    elif model_name == 'VGG11':
        net = VGG('VGG11')
    elif model_name == "xception":
        net = xception(num_classes=512, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    return net

def euclid_dist(x,y, axis=1): 
    return np.sqrt(((x - y) ** 2).sum(axis=axis))

def tensor2img(tensor, on_cuda=True):
    tensor = reverse_normalize(tensor, REVERSE_MEAN, REVERSE_STD)
    # clipping
    tensor[tensor > 1] = 1
    tensor[tensor < 0] = 0
    tensor = tensor.squeeze(0)
    if on_cuda:
        tensor = tensor.cpu()
    return transforms.ToPILImage()(tensor)


class Attacker():
    def __init__(self, model, ssim_thr, transform, img2tensor, args, mode='begin'):
        self.model = model
        self.model.eval()
        self.net = MCS2018.Predictor(0)
        self.ssim_thr = ssim_thr
        self.transform = transform
        self.cropping = transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.Scale(112)
                                      ])
        self.img2tensor = img2tensor
        self.args = args
        self.loss = nn.MSELoss()

    def read_target_descriptors(self, target_img_names):
        target_descriptors = np.ones((len(target_img_names), 512), 
                                     dtype=np.float32)

        for idx, img_name in enumerate(target_img_names):
            img_name = os.path.join("data/imgs", img_name)
            img = Image.open(img_name)
            tensor = self.transform(img).unsqueeze(0)
            if self.args.cuda:
                tensor = tensor.cuda(async=True)

            #res = self.model(Variable(tensor, requires_grad=False))\
            #          .data.cpu().numpy().squeeze(i)
            res = self.net.submit(tensor.cpu().numpy()).squeeze()
            target_descriptors[idx] = res

        return target_descriptors

    def get_target_descriptors_vars(self, target_descriptors):
        """
        convert target_descriptors from numpy to torch.autograd.Variable and return

        args: 
            target_descriptors : numpy array of shape (n, desc_len)
        return: 
            target_outs : list (n, ) of torch.autograd.Variable of shape (desc_len)
        """
        target_outs = []
        for target_descriptor in target_descriptors:
            target_out = torch.from_numpy(target_descriptor)\
                                      .unsqueeze(0)
            if self.args.cuda:
                target_out = target_out.cuda(async=True)
            target_out = Variable(target_out,
                         requires_grad=False)
            target_outs.append(target_out)
        return target_outs

    def get_mean_loss(self, input_var, target_vars, loss_func=None):
        """
        compute losses between input_var and each of target_vars and get mean

        args:
            input_var : torch.autograd.Variable 
            target_vars : list of torch.autograd.Variable of same shapes as input_var
            loss_func : function to compute loss with. If None, self.loss is used
        return:
            loss : mean loss, float
        """
        if loss_func is None:
            loss_func = self.loss
        losses = []
        for target in target_vars:
            losses.append(self.loss(input_var, target).data[0])
        return np.array(losses).mean()

    def get_mean_euclidean_loss(self, input_vec, target_vecs):
        losses = []
        for target in target_vecs:
            losses.append(euclid_dist(input_vec, target))
        return np.array(losses).mean()

    def get_net_desc_var(self, input_tensor):
        """
        get torch.autograd.Variable of self.net output

        args:
            input_tensor : torch tensor to put into self.net
        return:
            net_descs_var : torch.autograd.Variable containing output of self.net for input_tensor
        """
        net_descs = self.net.submit(input_tensor.cpu().numpy()).squeeze()
        net_descs_var = torch.from_numpy(net_descs)
        if self.args.cuda:
            net_descs_var = net_descs_var.cuda(async=True)
        net_descs_var = Variable(net_descs_var, requires_grad=False)
        return net_descs_var

    def attack(self, attack_pairs):
        raise NotImplementedError



class FGSM_Attacker(Attacker):
    '''
    FGSM attacker: https://arxiv.org/pdf/1412.6572.pdf
    model -- white-box model for attack
    eps -- const * Clipped Noise
    ssim_thr -- min value for ssim compare
    transform -- img to tensor transform without CenterCrop and Scale
    '''

    def __init__(self, model, ssim_thr, transform, img2tensor, args, mode='begin',
                    eps=1e-2, max_iter=10000):
        super().__init__(model, ssim_thr, transform, img2tensor, args, mode)
        self.initial_eps = eps
        self.max_iter = max_iter

        self.eps_drop_rate = 2
        self.eps_stop_value = 1e-4

    def attack(self, attack_pairs):
        '''
        Args:
            attack_pairs (dict) - id pair, 'source': 5 imgs,
                                           'target': 5 imgs
        '''
        target_img_names = attack_pairs['target']
        target_descriptors = self.read_target_descriptors(target_img_names)
        final_ssims = []
        
        for img_name in attack_pairs['source']:

            #img is attacked
            if os.path.isfile(os.path.join(self.args.save_root, img_name)):
                continue

            # 1. read image and convert to torch.autograd.Variable
            img = Image.open(os.path.join(self.args.root, img_name))
            tensor = self.transform(img).unsqueeze(0)
            if self.args.cuda:
                tensor = tensor.cuda(async=True)
            input_var = Variable(tensor,
                                requires_grad=True)
    
            # 2. get initial loss for image before attacking
            target_vars = self.get_target_descriptors_vars(target_descriptors)
            #desc_from_orig_image_var = self.get_net_desc_var(tensor)
            #initial_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)
            desc_from_orig_image = self.net.submit(tensor.cpu().numpy()).squeeze()
            net_loss = []
            for target_descriptor in target_descriptors:
                net_loss.append(euclid_dist(desc_from_orig_image, target_descriptor, axis=0))
            initial_loss = np.mean(net_loss)
            print("==============")
            print("INITIAL LOSS:", initial_loss)
            # image to compare ssim with
            original_img = self.cropping(img)
            # 3. attacking
            self.eps = self.initial_eps
            attacked_img = original_img
            final_ssim = None

            for iter_number in range(self.max_iter):
                adv_noise = torch.zeros((3,112,112))
                
                if self.args.cuda:
                    adv_noise = adv_noise.cuda(async=True)

                for target in target_vars:
                    input_var.grad = None
                    # 3.1 calculate loss
                    out = self.model(input_var)
                    calc_loss = self.loss(out, target)
                    calc_loss.backward()
                    # 3.2 calculate noise
                    #noise = input_var.grad / input_var.grad.data.view(-1).std()
                    noise = input_var.grad.data.squeeze()
                    #noise = torch.clamp(noise, min=-2., max=2.).pow(2)

                    #pos_noise_mask = (noise>0).float()
                    #neg_noise_mask = -(noise<0).float()
                    #noise = torch.clamp(torch.abs(noise), min = 0, max=6)
                    #noise = noise * (pos_noise_mask + neg_noise_mask)
                    #noise = torch.clamp(noise, min=-5., max=5.)#.pow(2)
                    #noise = self.eps * 1 * noise.data\
                    #                   .squeeze()
                    # 3.3 add noise
                    adv_noise = adv_noise + noise


                adv_noise = adv_noise / adv_noise.std()
                #adv_noise = self.eps * torch.clamp(adv_noise, min=-4., max=4.)
                pos_noise_mask = (adv_noise>0).float()
                neg_noise_mask = -(adv_noise<0).float()
                adv_noise_ = torch.clamp(adv_noise, min = -5, max=5)
                #adv_noise = adv_noise_ * (pos_noise_mask + neg_noise_mask) + 2*torch.clamp(adv_noise, min=-1, max=1)
                adv_noise = self.eps * adv_noise
                # save in case we made worse
                old_input_var_data = input_var.data
                # change image data
                input_var.data = input_var.data - adv_noise
                changed_img = tensor2img(input_var.data.cpu().squeeze())

                # 3.4 get descriptors from bb for new image data
                #desc_from_changed_image_var = self.get_net_desc_var(input_var.data)
                # get loss for new image
                #new_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)
                new_out = self.net.submit(input_var.data.cpu().numpy()).squeeze()
                new_net_loss = []
                for target_descriptor in target_descriptors:
                    new_net_loss.append(euclid_dist(new_out, target_descriptor, axis=0))
                new_loss = np.mean(new_net_loss)
                # 3.5 SSIM checking
                ssim = compare_ssim(np.array(original_img),
                                    np.array(changed_img),  
                                    multichannel=True)
                # 3.6 check conditions
                if ssim < self.ssim_thr or new_loss > initial_loss:
                    # drop eps
                    print("ololo", self.eps, ssim, new_loss)
                    self.eps = self.eps / self.eps_drop_rate
                    input_var.data = old_input_var_data
                    if self.eps < self.eps_stop_value:
                        break
                        #self.eps = 0.01
                else:
                    # making new image as baseline for next iteration
                    initial_loss = new_loss
                    print("NEW LOSS:", new_loss, "ssim", ssim)
                    attacked_img = changed_img

            # 4. save
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            attacked_img.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))



class OnePixelAttacker(Attacker):
    def __init__(self, model, ssim_thr, transform, img2tensor, args, mode='begin', 
                    popmul_const=200, max_iter=3):
        super().__init__(model, ssim_thr, transform, img2tensor, args, mode)
        self.mode = mode
        self.max_iter = max_iter
        self.popmul_const = popmul_const

    def perturb_image(self, xs, img):
        """
        change one pixel of image
        args:
            xs: tuple (x, y, r, g, b) : x,y -- pixel coords to change
                                        r, g, b -- values to set to img[x,y]
            img: image in which to change pixel data
        """
        xs = xs.astype(int)
        pixels = np.split(xs, len(xs) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
        return img

    def objective_function(self, xs, image, img_before, targets):
        """
        objective function for scipy.optimize.differential_evolution
        args:
            xs: tuple (x, y, r, g, b) : x,y -- pixel coords to change
                                        r, g, b -- values to set to img[x,y]
            image: image in which to change pixel data
        """
        changed_img = self.perturb_image(xs, np.array(image))
        
        tensor = self.img2tensor(changed_img).unsqueeze(0)
        #changed_img_net_desc_var = self.get_net_desc_var(tensor)
        #loss = self.get_mean_loss(changed_img_net_desc_var, targets)
        desc_from_orig_image = self.net.submit(tensor.cpu().numpy()).squeeze()
        net_loss = []
        for target_descriptor in targets:
            net_loss.append(euclid_dist(desc_from_orig_image, target_descriptor, axis=0))
        loss = np.mean(net_loss)
        ssim = compare_ssim(np.array(img_before),
                            np.array(changed_img),
                            multichannel=True) 
        if ssim < 0.95:
            loss = 1e6
        return loss


    def attack(self, attack_pairs):
        target_img_names = attack_pairs['target']
        target_descriptors = self.read_target_descriptors(target_img_names)

        for img_name in attack_pairs['source']:
            #img is attacked
            if os.path.isfile(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png'))):
                continue

            # 1. read image and convert to torch.autograd.Variable
            original_img = Image.open(os.path.join(self.args.root, 
                                    img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
            if self.mode == "continue":
                tensor = self.img2tensor(original_img).unsqueeze(0)
            else:
                original_img = self.cropping(original_img)
                tensor = self.transform(original_img).unsqueeze(0)
            if self.args.cuda:
                tensor = tensor.cuda(async=True)
            input_var = Variable(tensor,
                                requires_grad=True)
            # image to compare ssim to
            img_before = self.cropping(original_img)
            if self.mode == "continue":
                img_before = Image.open(os.path.join('data/imgs', img_name))
                img_before = self.cropping(img_before)
    
            # 2. get initial loss for image before attacking
            target_vars = self.get_target_descriptors_vars(target_descriptors)
            #desc_from_orig_image_var = self.get_net_desc_var(tensor)
            #initial_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)
            desc_from_orig_image = self.net.submit(tensor.cpu().numpy()).squeeze()
            net_loss = []
            for target_descriptor in target_descriptors:
                net_loss.append(euclid_dist(desc_from_orig_image, target_descriptor, axis=0))
            initial_loss = np.mean(net_loss)

            # 3. run differential_evolution
            bounds = [(0,112), (0,112), (0,256), (0,256), (0,256)] * 1
            popmul = max(1, self.popmul_const // len(bounds))
            predict_fn = lambda xs: self.objective_function(xs, original_img, img_before, target_descriptors)
            attack_result = differential_evolution(
                predict_fn, bounds, maxiter=20, popsize=400,
                recombination=1, atol=-1, polish=False, seed=42, disp=True)
            attack_image = self.perturb_image(attack_result.x, np.array(original_img))
            
            # 4. ssim checking
            #ssim = compare_ssim(np.array(img_before),
            #                    np.array(attack_image),  
            #                    multichannel=True)
            # 5. get descriptors from bb for new image data
            #desc_from_changed_image_var = self.get_net_desc_var(self.img2tensor(attack_image).unsqueeze(0))
            # get loss for new image
            new_out = self.net.submit(self.img2tensor(attack_image).unsqueeze(0).cpu().numpy()).squeeze()
            new_net_loss = []
            for target_descriptor in target_descriptors:
                new_net_loss.append(euclid_dist(new_out, target_descriptor, axis=0))
            new_loss = np.mean(new_net_loss)
            #new_loss = self.get_mean_loss(desc_from_changed_image_var, target_vars)
            
            # 6. save
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            if new_loss < initial_loss: #and ssim > 0.95:
                attack_image = Image.fromarray(attack_image)
                attack_image.save(os.path.join(self.args.save_root, 
                    img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
                LOSS_ESTIMATE.append(new_loss)
            else:
                original_img.save(os.path.join(self.args.save_root, 
                    img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
                LOSS_ESTIMATE.append(initial_loss)

            print("Loss estimate:", np.mean(LOSS_ESTIMATE))


import pytorch_ssim
                    
class CW_Attacker(Attacker):
    def __init__(self, model, ssim_thr, transform, img2tensor, args, mode='begin',
                    lr=0.005, max_iter=3):
        super().__init__(model, ssim_thr, transform, img2tensor, args, mode)
        self.mode = mode
        self.initial_lr = lr

        self.upper_bound = 20
        self.lower_bound = 0
        self.initial_scale_const = self.upper_bound / 2.#5e4

        self.max_search_steps = 20
        self.max_iter_steps = 200
        self.loss_2 = pytorch_ssim.SSIM(window_size=7)

    """
    def torch_arctanh(self, x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def tanh_rescale(self, x, x_min=-1., x_max=1.):
        return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min
    """
    def reduce_sum(self, x, keepdim=True):
        # silly PyTorch, when will you get proper reducing sums/means?
        for a in reversed(range(1, x.dim())):
            x = x.sum(a, keepdim=keepdim)
        return x

    def l2_dist(self, x, y, keepdim=True):
        d = ((x - y)**2)
        return self.reduce_sum(d, keepdim=keepdim).sqrt()

    def get_loss(self, input_var, input_adv, target_vars):
        desc_model_changed_image = self.model(input_adv)
        
        loss_1 =  self.l2_dist(desc_model_changed_image, target_vars[0], keepdim=False)
        for i, target in enumerate(target_vars[1:]):
            loss_1 = loss_1 + self.l2_dist(desc_model_changed_image, target, keepdim=False)
        loss_1 = loss_1 / len(target_vars)
        
        loss_2 = self.loss_2(input_adv, input_var)
        
        loss =  self.scale_const * loss_1 - loss_2
        
        return loss, loss_1, loss_2

    def attack(self, attack_pairs):
        '''
        Args:
            attack_pairs (dict) - id pair, 'source': 5 imgs,
                                           'target': 5 imgs
        '''
        target_img_names = attack_pairs['target']
        target_descriptors = self.read_target_descriptors(target_img_names)

        for img_name in attack_pairs['source']:
            print('=====================================================')
            #img is attacked
            #if os.path.isfile(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png'))):
            #    continue

            # 1. read image and convert to torch.autograd.Variable
            original_img = Image.open(os.path.join(self.args.root, img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
            
            tensor = self.transform(original_img).unsqueeze(0)
            if self.args.cuda:
                tensor = tensor.cuda(async=True)
            input_var = Variable(tensor, requires_grad=True)
            # image to compare ssim to
            img_before = self.cropping(original_img)
            if self.mode == "continue":
                img_before = Image.open(os.path.join('data/imgs', img_name))
                img_before = self.cropping(img_before)
    
            # 2. get initial loss for image before attacking
            self.scale_const = self.initial_scale_const
            
            target_vars = self.get_target_descriptors_vars(target_descriptors)

            loss, loss_1, loss_2 = self.get_loss(input_var, input_var, target_vars)

            best_loss = loss_1.data[0]
            initial_loss = loss_1.data[0]
            #print("INIT LOSS:", best_loss, loss_1.data[0], loss_2.data[0])

            # 3. set variable for adding to image
            modifier = torch.zeros(input_var.size()).float()
            modifier = torch.normal(means=modifier, std=0.0001)
            if self.args.cuda:
                modifier = modifier.cuda()
            modifier_var = Variable(modifier, requires_grad=True)

            # 4. run attack

            
            self.best_img = img_before
            for search_step in range(self.max_search_steps):
                # constants
                #print("Search step:", search_step, "Constant:", self.scale_const)
                self.i = 0
                self.j = 0

                ssim_reached = False

                self.lr = self.initial_lr
                optimizer = optim.Adam([modifier_var], lr=self.initial_lr)
                #prev_modifier_var = modifier_var.detach()

                best_step_loss = 0
                for step in range(self.max_iter_steps):
                    # 4.1 change image -- add modifier
                    input_adv = modifier_var + input_var

                    loss, loss_1, loss_2 = self.get_loss(input_var, input_adv, target_vars)
                    
                    #print(dist, loss_main)
                    # 4.3 run optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 4.4 check ssim
                    out_img = tensor2img(input_adv.data.cpu())
                    ssim = compare_ssim(np.array(out_img),
                                        np.array(img_before),  
                                        multichannel=True)
                    #print("ssim", ssim)
                    if ssim <= 0.955:
                        ssim_reached = True
                    else:
                        ssim_reached = False
                        #modifier_var = prev_modifier_var
                        #modifier_var.requires_grad = True
                        #optimizer = optim.Adam([modifier_var], lr=self.lr)
                        #break

                    # 4.5 calculate new loss
                    #desc_net_changed_image = self.get_net_desc_var(input_adv)
                    new_out = self.net.submit(input_adv.data.cpu().numpy()).squeeze()
                    new_net_loss = []
                    for target_descriptor in target_descriptors:
                       new_net_loss.append(euclid_dist(new_out, target_descriptor, axis=0))
                    new_net_loss = np.mean(new_net_loss)


                    #print("NEW LOSS", new_net_loss)
                    #iprint("NEW FAKE LOSS", loss_1.data[0], "ssim:", loss_2.data[0], ssim, 'sum', loss.data[0])
                    if not (loss.data[0] > 0.9999 * best_step_loss) or step == 0:
                        #if new_net_loss < best_loss or step == 0:
                        #best_loss = new_net_loss
                        self.i = 0
                        self.j = 0
                        best_step_loss = loss.data[0]
                    else:
                        self.i += 1
                        self.j += 1
                        if self.i >= 2:
                            self.lr /= 2
                            self.i = 0
                            optimizer = optim.Adam([modifier_var], lr=self.lr)
                        
                        if self.j >= 5: #early stop
                            break
                        #modifier_var = prev_modifier_var
                        #modifier_var.requires_grad = True
                        # set optimizer again as we changed lr
                        #optimizer = optim.Adam([modifier_var], lr=self.lr)
                        #if self.i >= 5:
                        #    self.i = 0
                        #    continue

                    if new_net_loss <= best_loss and ssim > 0.95:
                        self.best_img = out_img
                        #print("NEW LOSS", new_net_loss, "ssim", ssim)
                        best_loss = new_net_loss
                        #print('SAVED')


                    #if loss_.data[0] > prev_loss * .9999:
                    #    break
                    #prev_loss = loss_.data[0]

                # 5 binary search for self.scale_const
                if not ssim_reached:
                    self.lower_bound = self.scale_const
                    self.scale_const = (self.lower_bound + self.upper_bound) / 2
                else:
                    self.upper_bound = self.scale_const
                    self.scale_const = (self.lower_bound + self.upper_bound) / 2

            # 6 save image
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            self.best_img.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))
            print("OLOLO", initial_loss, best_loss)                      

 
def main():
    #print ('TEST: start')
    model = get_model(args.model_name, args.checkpoint_path)
    #print ('TEST: model on cpu')
    #net = MCS2018.Predictor(gpu_id)
    if args.cuda:
        model = model.cuda()
    #print ('TEST: model loaded')  

    transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Scale(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
                ])
    img2tensor = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=MEAN, std=STD)
                 ])

    attackers = {
        'FGSM': FGSM_Attacker,
        'OnePixel' : OnePixelAttacker,
        'CW' : CW_Attacker,
    }

    attacker = attackers[args.attack_type](model,
                        ssim_thr=SSIM_THR,
                        transform=transform,
                        img2tensor=img2tensor,
                        args=args,
                        mode=args.attack_mode)
    #print ('TEST: attacker is created')
    img_pairs = pd.read_csv(args.datalist)
    #print ('TEST: pairs are readed')
    for idx in tqdm(img_pairs.index.values):
        pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                     'target': img_pairs.loc[idx].target_imgs.split('|')}
        
        attacker.attack(pair_dict)

if __name__ == '__main__':
    main()

