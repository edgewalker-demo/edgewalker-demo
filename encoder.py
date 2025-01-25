import torch
import warnings
import random
warnings.filterwarnings("ignore")
from common.Tiny_AE import encoder 
from common.Tiny_AE import TinyAE
from common.color_compress import color_quantization
from common.video_image import mp4_to_png_1
from common.video_image import png_to_mp4_1
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
parser.add_argument("--load", type=str, default="./data/original.mp4")
parser.add_argument("--save", type=str, default="./output/compressed.mp4")
args = parser.parse_args()


input_root=args.load
mp4_to_png_1(input_root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Model loading ...")

AE= TinyAE(encoder_path='./model/BDD100K_encoder.pt', 
           decoder_path='./model/BDD100K_decoder.pt').to(device)

print("Compressing ...")

encoder(AE,device)  
color_quantization()
png_to_mp4_1(args.save,25)

print("Done !")
