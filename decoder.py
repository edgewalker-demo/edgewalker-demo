import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
import sys
import random
from common.Tiny_AE import decoder 
from common.Tiny_AE import TinyAE
from common.BDD100k_ldm_sampling import SR3
from common.BDD100k_ldm_sampling import inference
from common.video_image import png_to_mp4_2
from common.video_image import mp4_to_png_2
import torch
import numpy as np
import random
import argparse
torch.set_num_threads(1)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.Generator("cpu").manual_seed(seed)
    torch.Generator("cuda").manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)  

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default='./output/compressed.mp4')
parser.add_argument("--save", type=str, default='./output/restored.mp4')
args = parser.parse_args()
mp4_to_png_2(args.load)
img_size = [90,160]
transforms_ = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
test_dataloader = DataLoader(torchvision.datasets.ImageFolder(root='./output/tmp/LR', transform=transforms_),
                                batch_size=16,shuffle=False, num_workers=4,  pin_memory=True,prefetch_factor=4)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
schedule_opt = {'schedule':'linear', 'n_timestep':4, 'linear_start':1e-4, 'linear_end':0.05}

print("Model loading ...")

sr3 = SR3(device, img_size=img_size, LR_size=img_size, loss_type='l1',
                hr_dataloader='------', lr_dataloader='------', schedule_opt=schedule_opt,
                save_path1='..',save_path2='..',load_path='./model/BDD100K_ldm.pt',load=True,
                lr=1e-6, distributed=False)
AE= TinyAE(encoder_path='./model/BDD100K_encoder.pt', 
           decoder_path='./model/BDD100K_decoder.pt').to(device)

print("Restoring ...")

inference(test_dataloader,sr3,device)
decoder(AE,device)
png_to_mp4_2(args.save,25)

print("Done!")
