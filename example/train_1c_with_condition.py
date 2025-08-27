# Import of libraries
import random, os
#import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
from datetime import date, datetime, timezone, timedelta

import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import *

from torchvision import transforms
from torch.optim import Adam
from data import CustomCrackImageDataset
import torchvision.transforms as transforms

from model_cond import *
from setproctitle import *

setproctitle('Train')



def training_loop(FM, backbone_model, loader, n_eps, optim, device, store_path="ckpt"):
    #mse = nn.MSELoss()
    l1 = nn.L1Loss()
    
    best_loss = float("inf")
    #n_steps = ddpm.n_steps
    os.makedirs(store_path, exist_ok=True)
    
    
    for ep in range(n_eps):
        epoch_loss = 0.0
        for input_images,labels in tqdm(loader):
            # Loading data
            x1 = input_images.to(device)
            if np.random.random() < 0.1:
                y = None
            else:
                y = labels.to(device)
            n = len(x1)
            
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            x0 = torch.randn_like(x1).to(device)
            t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
            vt = backbone_model(xt, t, y1)
            
            loss = l1(vt, ut)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(backbone_model.state_dict(), f"{store_path}/best_ddpm_{str(ep).zfill(5)}_best.pt")
            log_string = f"Loss at epoch {ep}: {epoch_loss:.5f} --> Best model ever "
            print(log_string)


parser = ArgumentParser()
parser.add_argument('--batch_size',   dest="batch_size",      type=int,     default=1024)#1024
parser.add_argument('--n_eps',        dest="n_eps",           type=int,     default=25)
parser.add_argument('--lr',           dest="lr",              type=float,   default=5e-4)
parser.add_argument('--n_steps',      dest="n_steps",         type=int,     default=1000)
parser.add_argument('--th',           dest="th",              type=str,     default="100")
parser.add_argument('--label_num',    dest="label_num",       type=str,     default=233)
parser.add_argument('--wh_w',         dest="wh_w",            type=int,     default=64)
parser.add_argument('--wh_h',         dest="wh_h",            type=int,     default=64)
parser.add_argument('--layer',        dest="layer",           type=int,     default=5)
parser.add_argument('--resblock',     dest="resblock",        type=int,     default=1)
parser.add_argument('--dataset',      dest="dataset",         type=str,     default="JJG")

args = parser.parse_args()

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions


batch_size = args.batch_size    #32
n_eps = args.n_eps              # 100
lr = args.lr                    #5e-4#3e-6#0.001

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
simple_transform = transforms.Compose([transforms.ToTensor()])

defect_th = args.th#"Crack" #Crack_10_Padding
defect_splits = defect_th.split("_")
label_num = args.label_num#int(defect_splits[-3])

KolektorSDD2 = args.dataset
img_dir = KolektorSDD2
trainset = CustomCrackImageDataset(img_dir,"train",label_num, 1, simple_transform, (args.wh_w, args.wh_h))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


n_steps, min_beta, max_beta = args.n_steps, 0.0001, 0.02  # Originally used by the authors
in_ch = 1  
out_ch = 1
ch_mul = (1,)
for it in range(args.layer-1):
    ch_mul = ch_mul + (2,)

sigma = 0.0
backbone_model = UNet(in_channel=in_ch, out_channel=out_ch, channel_mults = ch_mul, res_blocks=args.resblock).to(device)
print("parameter #:",sum(p.numel() for p in backbone_model.parameters() if p.requires_grad))
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(backbone_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
optim = Adam(backbone_model.parameters())


exp_day = str(date.today())

KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)

training_loop(FM,backbone_model, trainloader, n_eps, optim, device,store_path=f"ckpt_{exp_day}_cond_l1_R_{n_steps}_{lr}lr_{in_ch}ch_{batch_size}batchsize_{defect_th}th_{n_eps}ep_{label_num}label_num_{args.wh_w}x{args.wh_h}_{args.layer}_{args.resblock}")