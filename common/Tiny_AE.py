import torch
from torch import nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 3),
    )

def Decoder():
    return nn.Sequential(
        Clamp(), conv(3, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TinyAE(nn.Module):

    def __init__(self, encoder_path, 
                 decoder_path):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
            print('encoder loaded')
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
            print('decoder loaded')


def encoder(vae,device):
    vae.encoder.eval()
    output_dir = './output/tmp/after_ae_encoder/after_ae_encoder'
    os.makedirs(output_dir, exist_ok=True)

    root = './data/BDD100k_s'
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

   
    batch_size = 4
    dataloader = DataLoader(
        torchvision.datasets.ImageFolder(root, transform=transforms_),
        batch_size=batch_size,  
        shuffle=False, 
        num_workers=4,  
        pin_memory=True,
        prefetch_factor=4
    )

    image_index = 100 

    for batch_idx, (images, _) in enumerate(dataloader):
        imgs = images.to(device, non_blocking=True)  
        imgs_en = vae.encoder(imgs)  

        for i in range(imgs_en.shape[0]):  
            filename = f"{image_index}.png"  
            file_path = os.path.join(output_dir, filename)

            torchvision.utils.save_image(imgs_en[i], file_path, normalize=True)
            
            image_index += 1 
    
def decoder(vae,device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae.decoder.eval()  

    output_dir = './output/Restore'
    os.makedirs(output_dir, exist_ok=True)

    root = './output/tmp/afterldm'
   
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

   
    batch_size = 4
    dataloader = DataLoader(
        torchvision.datasets.ImageFolder(root, transform=transforms_),
        batch_size=batch_size,  
        shuffle=False, 
        num_workers=4,  
        pin_memory=True,
        prefetch_factor=4
    )

    image_index = 100  

    for batch_idx, (images, _) in enumerate(dataloader):
        imgs = images.to(device, non_blocking=True)  
        imgs_de = vae.decoder(imgs)  

        for i in range(imgs_de.shape[0]):  
            filename = f"{image_index}.png"  
            file_path = os.path.join(output_dir, filename)

            torchvision.utils.save_image(imgs_de[i], file_path, normalize=True)
            
            image_index += 1  