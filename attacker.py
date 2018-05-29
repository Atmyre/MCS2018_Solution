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
    checkpoint = torch.load(checkpoint_path)
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
            img_name = os.path.join(self.args.root, img_name)
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
                    eps=0.01, max_iter=10000):
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
            desc_from_orig_image_var = self.get_net_desc_var(tensor)
            initial_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)

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
                    noise = input_var.grad / input_var.grad.data.view(-1).std()
                    noise = torch.clamp(noise, min=-2., max=2.)
                    noise = self.eps * noise.data\
                                       .squeeze()
                    # 3.3 add noise
                    adv_noise = adv_noise + noise

                # save in case we made worse
                old_input_var_data = input_var.data
                # change image data
                input_var.data = input_var.data - adv_noise
                changed_img = tensor2img(input_var.data.cpu().squeeze())

                # 3.4 get descriptors from bb for new image data
                desc_from_changed_image_var = self.get_net_desc_var(input_var.data)
                # get loss for new image
                new_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)

                # 3.5 SSIM checking
                ssim = compare_ssim(np.array(original_img),
                                    np.array(changed_img),  
                                    multichannel=True)
                # 3.6 check conditions
                if ssim < self.ssim_thr or new_loss > initial_loss:
                    # drop eps
                    self.eps = self.eps / self.eps_drop_rate
                    input_var.data = old_input_var_data
                    if self.eps < self.eps_stop_value:
                        break
                else:
                    # making new image as baseline for next iteration
                    initial_loss = new_loss
                    attacked_img = changed_img

            # 4. save
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            attacked_img.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))


class OnePixelAttacker(Attacker):
    def __init__(self, model, ssim_thr, transform, img2tensor, args, mode='begin', 
                    popmul_const=100, max_iter=3):
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

    def objective_function(self, xs, image, targets):
        """
        objective function for scipy.optimize.differential_evolution
        args:
            xs: tuple (x, y, r, g, b) : x,y -- pixel coords to change
                                        r, g, b -- values to set to img[x,y]
            image: image in which to change pixel data
        """
        changed_img = self.perturb_image(xs, np.array(image))
        
        tensor = self.img2tensor(img).unsqueeze(0)
        changed_img_net_desc_var = self.get_net_desc_var(tensor)
        loss = self.get_mean_loss(changed_img_net_desc_var, targets)

        return loss


    def attack(self, attack_pairs):
        target_img_names = attack_pairs['target']
        target_descriptors = self.read_target_descriptors(target_img_names)

        for img_name in attack_pairs['source']:
            #img is attacked
            if os.path.isfile(os.path.join(self.args.save_root, img_name if self.mode=="begin" else img_name.replace('.jpg', '.png'))):
                continue

            # 1. read image and convert to torch.autograd.Variable
            original_img = Image.open(os.path.join(self.args.root, 
                                    img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
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
            desc_from_orig_image_var = self.get_net_desc_var(tensor)
            initial_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)

            # 3. run differential_evolution
            bounds = [(0,112), (0,112), (0,256), (0,256), (0,256)] * 1
            popmul = max(1, self.popmul_const // len(bounds))
            predict_fn = lambda xs: self.objective_function(xs, original_img, target_vars)
            attack_result = differential_evolution(
                predict_fn, bounds, maxiter=self.max_iter, popsize=popmul,
                recombination=1, atol=-1, polish=False, disp=False)
            attack_image = self.perturb_image(attack_result.x, np.array(original_img))

            # 4. ssim checking
            ssim = compare_ssim(np.array(img_before),
                                np.array(attack_image),  
                                multichannel=True)

            # 5. get descriptors from bb for new image data
            desc_from_changed_image_var = self.get_net_desc_var(self.img2tensor(attack_image).unsqueeze(0))
            # get loss for new image
            new_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)

            # 6. save
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            if new_loss < initial_loss and ssim > 0.95:
                attack_image = Image.fromarray(attack_image)
                attack_image.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))
            else:
                img.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))


class CW_Attacker(Attacker):
    def __init__(self, model, ssim_thr, transform, img2tensor, args, mode='begin',
                    lr=0.01, max_iter=3):
        super().__init__(model, ssim_thr, transform, img2tensor, args, mode)
        self.mode = mode
        self.initial_lr = lr

        self.upper_bound = 1e10
        self.lower_bound = 0
        self.initial_scale_const = 7e4

        self.max_search_steps = 20
        self.max_iter_steps = 30

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
        d = (x - y)**2
        return self.reduce_sum(d, keepdim=keepdim)


    def attack(self, attack_pairs):
        '''
        Args:
            attack_pairs (dict) - id pair, 'source': 5 imgs,
                                           'target': 5 imgs
        '''
        target_img_names = attack_pairs['target']
        target_descriptors = self.read_target_descriptors(target_img_names)

        for img_name in attack_pairs['source']:
            #img is attacked
            if os.path.isfile(os.path.join(self.args.save_root, img_name if self.mode=="begin" else img_name.replace('.jpg', '.png'))):
                continue

            # 1. read image and convert to torch.autograd.Variable
            original_img = Image.open(os.path.join(self.args.root, 
                                    img_name if self.mode=="begin" else img_name.replace('.jpg', '.png')))
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
            # desc_from_orig_image_var = self.get_net_desc_var(tensor)
            # initial_loss = self.get_mean_loss(desc_from_orig_image_var, target_vars)
            desc_from_orig_image = self.net.submit(tensor.cpu().numpy()).squeeze()
            initial_loss = []
            for target_descriptor in target_descriptors:
                initial_loss.append(euclid_dist(desc_from_orig_image, target_descriptor, axis=0))
            initial_loss = np.mean(initial_loss)
            #print("INIT LOSS:", initial_loss)
            # 3. set variable for adding to image
            modifier = torch.zeros(input_var.size()).float()
            modifier = torch.normal(means=modifier, std=0.0001)
            if self.args.cuda:
                modifier = modifier.cuda()
            modifier_var = Variable(modifier, requires_grad=True)

            # 4. run attack
            best_loss = initial_loss
            self.scale_const = self.initial_scale_const
            self.best_img = original_img
            for search_step in range(self.max_search_steps):
                # constants
                #print("Search step:", search_step, "Constant:", self.scale_const)
                self.i = 0
                best_img = None
                self.changed = False

                self.lr = self.initial_lr
                optimizer = optim.Adam([modifier_var], lr=self.initial_lr)
                prev_modifier_var = modifier_var

                for step in range(self.max_iter_steps):
                    # 4.1 change image -- add modifier
                    input_adv = modifier_var + input_var

                    # 4.2 compute full loss 
                    desc_model_changed_image = self.model(input_adv)
                    
                    # да, я не умею по-нормальному это делать =\
                    loss_main = None
                    for i, target in enumerate(target_vars):
                        if i == 0:
                            loss_main = self.loss(desc_model_changed_image, target) / 5.
                        else:
                            loss_main += self.loss(desc_model_changed_image, target) / 5.
                    # additional loss -- l2 dist between orig and changed image
                    dist = self.l2_dist(input_adv, input_var, keepdim=False) #pytorch_ssim.ssim(input_adv, input_var)
                    # full loss
                    loss = self.scale_const * loss_main + dist 

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
                    if ssim <= 0.95:
                        break

                    # 4.5 calculate new loss
                    #desc_net_changed_image = self.get_net_desc_var(input_adv)
                    new_out = self.net.submit(input_adv.data.cpu().numpy()).squeeze()
                    new_net_loss = []
                    for target_descriptor in target_descriptors:
                        new_net_loss.append(euclid_dist(new_out, target_descriptor, axis=0))
                    new_net_loss = np.mean(new_net_loss)
                    #print("NEW LOSS", new_net_loss, "ssim:", ssim)
                    if new_net_loss >= best_loss:
                        self.i += 1
                        self.lr /= 2
                        modifier_var = prev_modifier_var
                        # set optimizer again as we changed lr
                        optimizer = optim.Adam([modifier_var], lr=self.lr)
                        if self.i >= 5:
                            self.i = 0
                            break
                    else:
                        #print("NEW LOSS", new_net_loss, "ssim:", ssim)
                        prev_modifier_var = modifier_var
                        self.i = 0
                        best_loss = new_net_loss
                        self.best_img = out_img
                        self.changed = True

                    #if loss_.data[0] > prev_loss * .9999:
                    #    break
                    #prev_loss = loss_.data[0]

                # 5 binary search for self.scale_const
                if not self.changed:
                    self.lower_bound = self.scale_const
                    self.scale_const = (self.lower_bound + self.upper_bound) / 2
                else:
                    self.upper_bound = self.scale_const
                    self.scale_const = (self.lower_bound + self.upper_bound) / 2

            # 6 save image
            if not os.path.isdir(self.args.save_root):
                os.makedirs(self.args.save_root)
            attack_image = self.best_img
            attack_image.save(os.path.join(self.args.save_root, img_name.replace('.jpg', '.png')))
            #print("OLOLO", orig_loss, best_loss)                      

 
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

