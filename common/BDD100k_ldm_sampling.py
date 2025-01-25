import math
import torch
from torch import nn
from inspect import isfunction
import numpy as np
import torchvision
from torch.nn import init
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from functools import partial
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class DepthWiseConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride,bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                      stride=stride, groups=dim_in,bias=bias)
        #self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1,bias=bias,stride=1,padding=0)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func_1 = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )
        self.noise_func_2 = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level)*2)
        )

    def forward(self, x, noise_embed):
        
        batch = x.shape[0]
        h=x.shape[2]
        #print(h)
        if h>20:
            if self.use_affine_level:
                gamma, beta = self.noise_func(noise_embed).view(
                    batch, -1, 1, 1).chunk(2, dim=1)
                x = (1 + gamma) * x + beta
            else:
                #print(x.shape)
                #print(self.noise_func(noise_embed).view(batch, -1, 1, 1).shape)
                x = x + self.noise_func_1(noise_embed).view(batch, -1, 1, 1)
        
        if h<=20:
            if self.use_affine_level:
                gamma, beta = self.noise_func(noise_embed).view(
                    batch, -1, 1, 1).chunk(2, dim=1)
                x = (1 + gamma) * x + beta
            else:
            
                x = x + self.noise_func_2(noise_embed).view(batch, -1, 1, 1)
            
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DepthWiseConv(dim, dim*2, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = DepthWiseConv(dim, dim//2, kernel_size=3,stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            DepthWiseConv(dim, dim_out, kernel_size=3, stride=1,padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = DepthWiseConv(
            dim, dim_out, kernel_size=1,stride=1,padding=0) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = DepthWiseConv(in_channel, in_channel * 3, kernel_size=1,stride=1,padding=0, bias=False)
        self.out = DepthWiseConv(in_channel, in_channel, kernel_size=1,stride=1,padding=0)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=150,
        inner_channel=64,
        norm_groups=15,
        channel_mults=(1, 2,3),
        res_blocks=1,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
        ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel1 = 6
        self.input1=Block(pre_channel1,pre_channel1*6,groups=3)
        downs1 = []
        pre_channel1=pre_channel1*6
        for ind in range(3):
            is_last = (ind == num_mults - 1)
            for _ in range(0, res_blocks):
                downs1.append(ResnetBlocWithAttn(
                             pre_channel1,pre_channel1, noise_level_emb_dim=noise_level_channel, 
                             norm_groups=3, dropout=dropout, with_attn=False))
                down_channel1=pre_channel1
                pre_channel1=pre_channel1//2
            if not is_last:
                downs1.append(Downsample(down_channel1))
        self.downs1 = nn.ModuleList(downs1)
        pre_channel1=pre_channel1*2+1
        
        ups1 = []
        for ind in range(3):
            is_last = (ind == num_mults - 1)
            for _ in range(0, res_blocks):
                ups1.append(ResnetBlocWithAttn(
                    pre_channel1, pre_channel1, noise_level_emb_dim=noise_level_channel, norm_groups=3,
                        dropout=dropout, with_attn=False))
                up_channel1=pre_channel1
                pre_channel1 = pre_channel1*2
            if not is_last:
                ups1.append(Upsample(up_channel1))
        self.ups1 = nn.ModuleList(ups1)
        
        pre_channel2=in_channel
        self.input2=Block(pre_channel2,pre_channel2*6,groups=15)
        downs2=[]
        pre_channel2=pre_channel2*6
        for ind in range(3):
            is_last = (ind == num_mults - 1)
            for _ in range(0, res_blocks):
                downs2.append(ResnetBlocWithAttn(
                             pre_channel2, pre_channel2, noise_level_emb_dim=noise_level_channel, 
                             norm_groups=norm_groups, dropout=dropout, with_attn=True))
                down_channel2=pre_channel2
                pre_channel2=pre_channel2//2
            if not is_last:
                downs2.append(Downsample(down_channel2))

        self.downs2 = nn.ModuleList(downs2)
        pre_channel2=pre_channel2*2+1
        
        ups2 = []
        for ind in range(3):
            is_last = (ind == num_mults - 1)
            for _ in range(0, res_blocks):
                #print('100',pre_channel)
                ups2.append(ResnetBlocWithAttn(
                    pre_channel2, pre_channel2, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=True))
                up_channel2=pre_channel2
                pre_channel2 = pre_channel2*2
            if  not is_last:
                ups2.append(Upsample(up_channel2))
        self.ups2 = nn.ModuleList(ups2)
        
        
        self.final_conv = nn.ModuleList([Block(72, 36, groups=3),
                                         Block(36, 18, groups=3),
                                         Block(18, 3, groups=3)])
                                    
    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        list1 = []
        list2 = []
        list3 = []
        b, c, h, w = x.shape
        
        y=x.reshape(b*2,c*25,h//5,w//10)
        y=self.input2(y)
        c1=c*6
        c2=c*3
        for layer in self.downs2:
            if isinstance(layer, ResnetBlocWithAttn):
                y = layer(y, t)
                add1=y.reshape(b,c1,h,w)
                c1=c1//2
                list2.append(add1)
            else:
                y = layer(y)
                add2=y.reshape(b,c2,h,w)
                c2=c2//2
                list2.append(add2)
            list1.append(y)
    
        c3=int(c*3/2)
        c4=c*3
        for layer in self.ups2:
            if isinstance(layer, ResnetBlocWithAttn):
                y = layer(y+list1.pop(),t)
                add3=y.reshape(b,c3,h,w)
                c3=c3*2
                list2.append(add3)
            else:
                y = layer(y+list1.pop())
                add4=y.reshape(b,c4,h,w)
                c4=c4*2
                list2.append(add4)
        
        x=self.input1(x)
        for layer in self.downs1:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x+list2.pop(), t)
            else:
                x = layer(x+list2.pop())    
            list3.append(x)
    
        for layer in self.ups1:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x+list2.pop()+list3.pop(), t)
            else:
                x = layer(x+list2.pop()+list3.pop())
                
        out=torch.cat((x,y.reshape(b,c*6,h,w)),dim=1)
                                        
        for layer in self.final_conv:
                out=layer(out)    
        return out
    
class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac=0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        mean, log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, x_in):
        img = torch.rand_like(x_in, device=x_in.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in)
        return img

    # Compute loss to train the model
    def p_losses(self, x_hr,x_sr):
        x_start = x_hr
        lr_imgs = x_sr
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(x_start.device)
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha**2).sqrt() * noise
       
        pred_noise = self.model(torch.cat([lr_imgs, x_noisy], dim=1), time=sqrt_alpha)
        return self.loss_func(noise, pred_noise)

    def forward(self, x,y, *args, **kwargs):
        return self.p_losses(x,y, *args, **kwargs)


class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                    schedule_opt, save_path1, save_path2,load_path=None, load=False,out_channel=3,
                    lr=1e-5, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path1 = save_path1
        self.save_path2 = save_path2
        self.img_size = img_size
        self.LR_size = LR_size
        model = UNet()
        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)

        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):
        mini=10
        for i in tqdm(range(epoch)):
            train_loss = 0
            self.sr3.train()        
            for (A_images, _), (B_images, _) in zip(self.dataloader, self.testloader): 
      
                imgs_HR = A_images.to(self.device)
                imgs_SR = B_images.to(self.device)
                b, c, h, w = imgs_HR.shape
                self.optimizer.zero_grad()
                loss = self.sr3(imgs_HR,imgs_SR)
                loss = loss.sum() / int(b*c*h*w)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * b
        
            if (i+1) % verbose == 0:
    
                train_loss = train_loss / len(self.testloader)
                print(f'Epoch: {i+1} / loss:{train_loss:.4f}')
            
            if train_loss<mini:
                self.save(self.save_path1,self.save_path2)
                mini=train_loss
        
    def test(self, imgs):
        self.sr3.eval()
        imgs_lr = transforms.Resize(self.img_size)(imgs)
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(imgs_lr)
            else:
                result_SR = self.sr3.super_resolution(imgs_lr)
        
        self.sr3.train()
        return result_SR

    def save(self, save_path1,save_path2):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path1)
        torch.save(network, save_path2)

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")

import os
def inference(testloader,sr3,device):
    output_dir = './output/tmp/afterldm/afterldm'
    os.makedirs(output_dir, exist_ok=True)
    image_index = 100  

    for batch_idx, (images, _) in enumerate(testloader):
        images = images.to(device, non_blocking=True) 
        results = sr3.test(images)  
        
        for i in range(results.shape[0]):  
            filename = f"{image_index}.png"  
            file_path = os.path.join(output_dir, filename)
            torchvision.utils.save_image(results[i], file_path, normalize=True)
            image_index += 1  
