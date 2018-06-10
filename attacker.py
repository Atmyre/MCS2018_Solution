"""
implementation of attacks
"""

import MCS2018
import os
import time
import argparse
from torch import optim
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
from StudentModels import load_model, FineTuneModel
import re

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
                    nargs='+',
                    help='model name', 
                    default='ResNet18')
parser.add_argument('--checkpoint_path',
                    nargs='+',
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
parser.add_argument('--start_from',
                    type=int, 
                    help='start from img index',
                    default=0)
parser.add_argument('--iter',
                    type=int, 
                    help='pixel attack iterations',
                    default=1)
args = parser.parse_args()

def reverse_normalize(tensor, mean, std):
    '''
    reverese normalize to convert tensor -> PIL Image
    '''
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.div_(s).sub_(m)
    return tensor_copy

def get_model(model_name, checkpoint_path):
    '''
    Model architecture choosing
    '''
    if model_name == 'ResNet50':
        net = ResNet50()
    elif model_name == 'Xception':
        net = xception(pretrained=False, num_classes=512)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    return net

def get_model2(model_name, checkpoint_path):
    '''
    Model architecture choosing
    '''
    model = load_model(model_name, pretrained=True)
    model = FineTuneModel(model,model_name,512)

    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(checkpoint_path)

    if torch.__version__ == '0.4.0':
        checkpoint['state_dict'] = {re.sub('(conv|norm)\.(\d+)','\g<1>\g<2>', k): v for k, v in checkpoint['state_dict'].items()}

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    model=model.module
   
    return model

def euclid_dist(x,y, axis=0): 
    """
    euclidean distance between x and y
    """
    return np.sqrt(((x - y) ** 2).sum(axis=axis))

def tensor2img(tensor, on_cuda=True):
    """
    convert tensor -> PIL Image
    """
    tensor = reverse_normalize(tensor, REVERSE_MEAN, REVERSE_STD)
    # clipping
    tensor[tensor > 1] = 1
    tensor[tensor < 0] = 0
    tensor = tensor.squeeze(0)
    if on_cuda:
        tensor = tensor.cpu()
    return transforms.ToPILImage()(tensor)


class Attacker():
    """
    base class for all attacks
    """
    def __init__(self, ssim_thr, args):
        self.net = MCS2018.Predictor(0)
        self.ssim_thr = ssim_thr
        self.cropping = transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.Resize(112)
                                      ])
        self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
                ])
        self.img2tensor = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=MEAN, std=STD)
                 ])
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
            target_out = torch.from_numpy(target_descriptor).unsqueeze(0)
            if self.args.cuda:
                target_out = target_out.cuda(async=True)
            target_out = Variable(target_out,
                         requires_grad=False)
            target_outs.append(target_out)
        return target_outs

    def attack(self, attack_pairs):
        '''
        Args:
            attack_pairs (dict) - id pair, 'source': 5 imgs,
                                           'target': 5 imgs
        '''
        raise NotImplementedError



class IFGM_Attacker(Attacker):
    """
    https://arxiv.org/pdf/1412.6572.pdf
    https://arxiv.org/pdf/1710.06081.pdf
    """
    def __init__(self, models, ssim_thr, args, eps=0.016, max_iter=500):
        super().__init__(ssim_thr, args)
        self.models = models
        for m in models:
            m.eval()
        self.eps = eps
        self.max_iter = max_iter
        self.distances = []

    def attack(self, attack_pairs):
        '''
        Args:
            attack_pairs (dict) - id pair, 'source': 5 imgs,
                                           'target': 5 imgs
        '''
        target_img_names = attack_pairs['target']
        target_descriptors_bb = self.read_target_descriptors(target_img_names)
        
        target_descriptors = []
        for idx in range(len(target_descriptors_bb)):
            target_descriptors.append(target_descriptors_bb[idx])

        # Order matters
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 1]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 2]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 4]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[1, 2]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[1, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[1, 4]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[2, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[2, 4]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[3, 4]], axis=0))
        
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 1, 2, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[4, 1, 2, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 4, 2, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 1, 4, 3]], axis=0))
                          
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 1, 2]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 1, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 1, 4]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 2, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[0, 2, 4]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[1, 2, 3]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[1, 2, 4]], axis=0))
        target_descriptors.append(np.mean(target_descriptors_bb[[2, 3, 4]], axis=0))
        
        target_descriptors.insert(0, np.mean(target_descriptors_bb, axis=0))

        
        for img_name in attack_pairs['source']:
            #img is attacked
            if os.path.isfile(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png'))):
                continue

            img = Image.open(os.path.join(self.args.root, img_name))
            original_img = self.cropping(img)
            attacked_img = original_img

            tensor = self.img2tensor(original_img).unsqueeze(0)
            input_var = Variable(tensor.cuda(async=True),
                                 requires_grad=True)

            final_ssim = 0
            best_iter = 0
            best_dist = 0

            early_stop = len(target_descriptors_bb)
            desc_number = 0

            descriptor_bb = self.net.submit(tensor.cpu().numpy()).squeeze()
            for target_descriptor in target_descriptors_bb:
                best_dist += euclid_dist(descriptor_bb, target_descriptor)
                
            for iter_number in tqdm(range(self.max_iter)):
                target_descriptor = target_descriptors[desc_number % len(target_descriptors)]
                input_var.grad = None
                input_var.data.cpu()

                weights = [0.2, 1.0, 1.0, 0.2, 1.2] # Hardcoded weights for last set of student networks.

                out = self.models[0](input_var) * weights[0]
                
                for i, m in enumerate(self.models[1:]):
                    out += m(input_var) * weights[i+1]
                out /= np.sum(weights)

                ti = torch.from_numpy(target_descriptor).unsqueeze(0)
                if self.args.cuda:
                    ti = ti.cuda(async=True)
                target_out = Variable(ti, requires_grad=False)

                calc_loss = self.loss(out, target_out)
                calc_loss.backward()
                adv_noise = input_var.grad.data.squeeze()

                adv_noise.div_(adv_noise.std())
                adv_noise = self.eps * torch.clamp(adv_noise, min=-2., max=2.)
                
                new_img_data = (input_var.data - adv_noise).cpu()
                changed_img = tensor2img(new_img_data.squeeze())

                #SSIM checking
                ssim = compare_ssim(np.array(original_img), 
                                    np.array(changed_img), 
                                    multichannel=True)

                if ssim < self.ssim_thr:
                    break
                else:
                    descriptor_bb = self.net.submit(new_img_data.cpu().numpy()).squeeze()
                    new_dist = 0
                    for target_descriptor in target_descriptors_bb:
                        new_dist += euclid_dist(descriptor_bb, target_descriptor)

                    if new_dist < best_dist:
                        input_var.data = input_var.data - adv_noise
                        best_dist = new_dist
                        attacked_img =  changed_img
                        final_ssim = ssim
                        best_iter = iter_number
                        #desc_number += 1
                        early_stop = len(target_descriptors)
                    else:
                        early_stop -= 1
                        desc_number += 1
                        if early_stop <= 0:
                            early_stop = len(target_descriptors_bb) - 1
                            input_var.data = input_var.data - adv_noise
                            #break
                        
            self.distances.append(best_dist / len(target_descriptors_bb))
            tqdm.write("[%03d / %03d] ssim %f | %f" % (best_iter, iter_number-1, final_ssim, np.mean(self.distances)))
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            attacked_img.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))

LOSS_ESTIMATE = []

class OnePixelAttacker(Attacker):
    """
    genetic algorithm that changes one pixel of image to fool the network
    """
    def __init__(self, ssim_thr, args, mode='begin', popsize=30, max_iter=4, skip_ssim=None):
        super().__init__(ssim_thr, args)
        self.mode = mode
        self.max_iter = max_iter
        self.popsize = popsize
        self.skip_ssim = skip_ssim

    def perturb_image(self, xs, img):
        """
        change one pixel of image
        args:
            xs: tuple (x, y, r, g, b) : x,y -- pixel coords to change
                                        r, g, b -- values to set to img[x,y]
            img: image in which to change pixel data
        """
        xs = xs.astype(int)
        img=img.copy()
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
        changed_img = self.perturb_image(xs, image)
        tensor = self.img2tensor(changed_img).unsqueeze(0)

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
            desc_from_orig_image = self.net.submit(tensor.cpu().numpy()).squeeze()
            net_loss = []
            for target_descriptor in target_descriptors:
                net_loss.append(euclid_dist(desc_from_orig_image, target_descriptor, axis=0))
            initial_loss = np.mean(net_loss)
            print('INITIAL LOSS:', initial_loss)
            changed_image = np.array(original_img)
            
            # 3. run differential_evolution
            bounds = [(0,112), (0,112), (0,256), (0,256), (0,256)]
            should_calc = True
            if self.skip_ssim:
                ssim = compare_ssim(np.array(img_before),
                                    changed_image,  
                                    multichannel=True)
                should_calc = ssim > self.skip_ssim

            if should_calc:
                for i in range(self.args.iter):
                    predict_fn = lambda xs: self.objective_function(xs, changed_image, img_before, target_descriptors)
                    attack_result = differential_evolution(
                        predict_fn, bounds, maxiter=self.max_iter, popsize=self.popsize,
                        recombination=1, atol=-1, polish=False, seed=42, disp=True)
                    attack_image = self.perturb_image(attack_result.x, changed_image)
                    
                    # 4. ssim checking
                    ssim = compare_ssim(np.array(img_before),
                                        attack_image,  
                                        multichannel=True)
                    print(ssim, attack_result.fun)
                    
                    if ssim > 0.95 and attack_result.fun < initial_loss:
                        changed_image = attack_image
                        initial_loss = attack_result.fun

                        if self.skip_ssim and ssim < self.skip_ssim:
                            break
                    else:
                        break

            # 6. save
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            
            attack_image = Image.fromarray(changed_image)
            attack_image.save(os.path.join(self.args.save_root, 
                img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))

            LOSS_ESTIMATE.append(initial_loss)
            print("Loss estimate:", np.mean(LOSS_ESTIMATE))


import pytorch_ssim         
class CW_Attacker(Attacker):
    """
    https://arxiv.org/pdf/1608.04644.pdf
    """
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

    def reduce_sum(self, x, keepdim=True):
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
        target_img_names = attack_pairs['target']
        target_descriptors = self.read_target_descriptors(target_img_names)

        for img_name in attack_pairs['source']:

            #img is attacked
            if os.path.isfile(os.path.join(self.args.save_root, 
                img_name if self.mode=="begin" else img_name.replace('.jpg', '.png'))):
               continue

            # 1. read image and convert to torch.autograd.Variable
            original_img = Image.open(os.path.join(self.args.root, 
                img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
            
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

            # 3. set variable for adding to image
            modifier = torch.zeros(input_var.size()).float()
            modifier = torch.normal(means=modifier, std=0.0001)
            if self.args.cuda:
                modifier = modifier.cuda()
            modifier_var = Variable(modifier, requires_grad=True)

            # 4. run attack           
            self.best_img = img_before
            for search_step in range(self.max_search_steps):
                self.i = 0
                self.j = 0

                ssim_reached = False

                self.lr = self.initial_lr
                optimizer = optim.Adam([modifier_var], lr=self.initial_lr)

                best_step_loss = 0
                for step in range(self.max_iter_steps):
                    # 4.1 change image -- add modifier
                    input_adv = modifier_var + input_var

                    loss, loss_1, loss_2 = self.get_loss(input_var, input_adv, target_vars)
                    
                    # 4.3 run optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 4.4 check ssim
                    out_img = tensor2img(input_adv.data.cpu())
                    ssim = compare_ssim(np.array(out_img),
                                        np.array(img_before),  
                                        multichannel=True)

                    if ssim <= 0.955:
                        ssim_reached = True
                    else:
                        ssim_reached = False

                    # 4.5 calculate new loss
                    new_out = self.net.submit(input_adv.data.cpu().numpy()).squeeze()
                    new_net_loss = []
                    for target_descriptor in target_descriptors:
                       new_net_loss.append(euclid_dist(new_out, target_descriptor, axis=0))
                    new_net_loss = np.mean(new_net_loss)


                    if not (loss.data[0] > 0.9999 * best_step_loss) or step == 0:
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

                    if new_net_loss <= best_loss and ssim > 0.95:
                        self.best_img = out_img
                        best_loss = new_net_loss

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


def main():
    attacker = None
    if args.attack_type == 'IFGM':
        models = []
        for i, model_name in enumerate(args.model_name):
            # team merge artifacts.
            if 'tar' in args.checkpoint_path[i]:
                model = get_model2(model_name, args.checkpoint_path[i])
            else:
                model = get_model(model_name, args.checkpoint_path[i])
            if args.cuda:
                model = model.cuda()
            models.append(model)
    
        attacker = IFGM_Attacker(models, ssim_thr=SSIM_THR, args=args)
    elif args.attack_type == 'OnePixel':
        attacker = OnePixelAttacker(ssim_thr=SSIM_THR, args=args, mode=args.attack_mode)
    elif args.attack_type == 'OnePixel-last-hope':
        attacker = OnePixelAttacker(ssim_thr=SSIM_THR, args=args, mode=args.attack_mode, popsize=30, max_iter=3, skip_ssim=0.951)

    img_pairs = pd.read_csv(args.datalist)
    for idx in tqdm(img_pairs.index.values[args.start_from:]):
        pair_dict = {'source': img_pairs.loc[idx].source_imgs.split('|'),
                     'target': img_pairs.loc[idx].target_imgs.split('|')}
        
        attacker.attack(pair_dict)

if __name__ == '__main__':
    main()
