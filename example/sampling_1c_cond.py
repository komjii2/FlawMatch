import random
import os, cv2
import numpy as np
from argparse import ArgumentParser
import time
from tqdm.auto import tqdm

import torch
import torchdiffeq
from model_cond import *
from setproctitle import *
from tqdm import tqdm
from argparse import ArgumentParser

setproctitle('Sampling')
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#class CFGScaleModel(UNet):
#    def __init__(self, model):
#        super().__init__(model)
#    def forward(self, x, time, cond):
#        predicted_noise = self.model.forward(x, time, cond)
#        uncond_predicted_noise = self.model.forward(x, time, None)
#        result = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
#        return result
def show_images(path, labels, images, start_num, cfg_scale):
    min = -1
    max = 1
    """Shows the provided images as sub-pictures in a square"""
    file_name = path.split("/")[-2]+"_"+path.split("/")[-1].replace(".pt","")+"_"+str(cfg_scale)+"cfg"
    os.makedirs(os.path.join("result",file_name),exist_ok = True)
    if type(images) is torch.Tensor:
        images = images.permute(0,2,3,1).detach().cpu().clamp_(min,max).numpy()
        labels = labels.detach().cpu().numpy()
    #for i, (img,label) in enumerate(zip(images, labels)):
    #    cv2.imwrite(os.path.join("result",file_name,str(start_num+i).zfill(5)+"_label"+str(label).zfill(5)+".png"), ((img-min)/(max-min))*255)
    

def generate_new_images(model, n_samples=16, device=None, total_sample=100, c=1, h=64, w=64, cfg_scale=3, label = 33, store_path =None, ys = None):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    with torch.no_grad():
        
        if total_sample % n_samples == 0:
            iter = int(total_sample/n_samples)
        else:
            iter = int(total_sample/n_samples + 1)
        for it in tqdm(range(iter)):
            if it+1 == iter and total_sample % n_samples != 0:
                mini_batch = int(total_sample % n_samples)
            else :
                mini_batch = n_samples
            #x = torch.randn(mini_batch, c, h, w).to(device)
            #time_tensor = (torch.ones(mini_batch, 1) * t).to(device).long()
            y = torch.tensor(ys[it*n_samples:it*n_samples+mini_batch]).to(device)
            traj = torchdiffeq.odeint(
                lambda t, x: model.cond_forward((torch.ones(mini_batch, 1).to(device) * t).long(), x, y, cfg_scale),
                torch.randn(mini_batch, c, h, w, device=device),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
            show_images(store_path, y, traj[-1, :mini_batch], n_samples*it, cfg_scale)
    

parser = ArgumentParser()
parser.add_argument('--store_path',   dest="store_path",      default="")
parser.add_argument('--cfg_scale',   dest="cfg_scale",      default=3.0)
parser.add_argument('--layer',        dest="layer",           type=int,     default=5)
parser.add_argument('--resblock',     dest="resblock",        type=int,     default=1)
parser.add_argument('--wh_w',         dest="wh_w",            type=int,     default=64)
parser.add_argument('--wh_h',         dest="wh_h",            type=int,     default=64)
parser.add_argument('--dataset',      dest="dataset",            type=str)

args = parser.parse_args()
ckpt_path = args.store_path
cfg_scale = float(args.cfg_scale)
in_ch = 1  
out_ch = 1
ch_mul = (1,)
for it in range(args.layer-1):
    ch_mul = ch_mul + (2,)#attn_res = [8, 4]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Loading the trained model
best_model = UNet(in_channel=in_ch, out_channel=out_ch, channel_mults = ch_mul, res_blocks=args.resblock).to(device)
best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
best_model.eval()
print("Model loaded")

sample_num = 2048

test_dir = args.dataset #"dataset/KolektorSDD2/generation/test"
test_files = os.listdir(test_dir)

ckpt_path_splits = ckpt_path.split("/")
dataset_splits = ckpt_path_splits[-2].split("_")
total_label_num = dataset_splits[-5].replace("label","")
print("total_label_num:",total_label_num)
ys = []
for test_file in test_files:
    wh_label_num = int(int(total_label_num)/100)
    lr_label_num = int(int(total_label_num)%100/10)
    tb_label_num = int(int(total_label_num)%10)
    extracted_test_file = test_file.replace(".png","")
    splits = extracted_test_file.split("_")
    test_img = cv2.imread(os.path.join(test_dir,test_file))
    test_img_shape = test_img.shape
    if lr_label_num == 2:
        if splits[-2] == "right":
            x_flag = 1 #"right"
        else:
            x_flag = 0 #"left"
    else:
        if splits[-2] == "right":
            x_flag = 2 #"right"
        elif splits[-2] == "mid":
            x_flag = 1 #"mid"
        else:
            x_flag = 0 #"left"
    if tb_label_num == 2:
        if splits[-1] == "bottom":
            y_flag = 1*lr_label_num #"bottom"
        else:
            y_flag = 0*lr_label_num #"top"
    else:
        if splits[-1] == "bottom":
            y_flag = 2*lr_label_num #"bottom"
        elif splits[-1] == "mid":
            y_flag = 1*lr_label_num #"bottom"
        else:
            y_flag = 0*lr_label_num #"top"
    if wh_label_num == 2:
        if test_img_shape[1]/test_img_shape[0]>1 :
            z_flag = 1*lr_label_num*tb_label_num
        else:
            z_flag = 0*lr_label_num*tb_label_num
    else:
        z_flag = 0
    ys.append(y_flag+x_flag+z_flag)
total_sample = len(test_files)
print("total_sample:", total_sample)
print("Generating new images")
st = time.time()
generate_new_images(
        best_model,
        n_samples=sample_num,
        device=device,
        total_sample=total_sample,
        store_path = ckpt_path,
        ys = ys,
        cfg_scale = cfg_scale,
        w = args.wh_w,
        h = args.wh_h
    )
#print(generated.size())
print("Elapsed time per image :", (time.time()-st)/total_sample)
#show_images(args.store_path, labels, generated)