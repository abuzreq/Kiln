import torch
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import os
import clip
import argparse
import cv2
from pytorch_msssim import ssim
from postproc import pprocess
from kornia.enhance import adjust_contrast, adjust_brightness, adjust_saturation, adjust_gamma
from kornia.enhance.equalization import equalize_clahe

from functools import partial

import numpy as np

import random
import math
import time

from diffusers import DDIMScheduler

from cutouts25 import CutoutConfig, GpuCutoutSampler


'''

xur diffusion

@htoyryla 2025

TILED GENERATION

with md2 ddim sampling with init and target image and clip conditioning 

using diffusers scheduler (not urdiffusion lib)

'''

#from cutouts import cut

parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--text', type=str, default="", help='text prompt')
parser.add_argument('--image', type=str, default="", help='path to init image')
parser.add_argument('--img_prompt', type=str, default="", help='path to image prompt')
parser.add_argument('--tgt_image', type=str, default="", help='path to target image')
parser.add_argument('--lr', type=float, default=5., help='learning rate')
parser.add_argument('--ssimw', type=float, default=1., help='target image weight')
parser.add_argument('--textw', type=float, default=1., help='text weight')
parser.add_argument('--tdecay', type=float, default=1., help='text weight decay')
parser.add_argument('--imgpw', type=float, default=1., help='image prompt weight')

parser.add_argument('--trainsteps', type=int, default=1000, help='diffusion steps')

parser.add_argument('--skip', type=int, default=0, help='skip steps')
parser.add_argument('--dir', type=str, default="out", help='base directory for storing images')
parser.add_argument('--name', type=str, default="", help='basename for storing images')
parser.add_argument('--mul', type=float, default=1., help='noise divisor when using init image')
parser.add_argument('--imb', type=float, default=1., help='init image level')
parser.add_argument('--show', action="store_true", help='show image in a window')
parser.add_argument('--ema', action="store_true", help='use ema model')
parser.add_argument('--imageSize', type=int, default=512, help='image size')
parser.add_argument('--h', type=int, default=0, help='image height')
parser.add_argument('--w', type=int, default=0, help='image width')
parser.add_argument('--modelSize', type=int, default=512, help='native image size of the model')
parser.add_argument('--saveEvery', type=int, default=0, help='image save frequency')
parser.add_argument('--saveAfter', type=int, default=0, help='save images after step')
parser.add_argument('--low', type=float, default=0.4, help='lower limit for cut scale')
parser.add_argument('--high', type=float, default=1.0, help='higher limit for cut scale')
parser.add_argument('--cuts', type=float, default=0.5, help='cutouts scheme, 0 = detail, 1 = structure')
parser.add_argument('--load', type=str, default="", help='path to pt file')
parser.add_argument('--saveiters', action="store_true", help='')
parser.add_argument('--saveorig', action="store_true", help='')
parser.add_argument('--mults', type=int, nargs='*', default=[1, 1, 2, 2, 4, 4, 8, 8], help='')
parser.add_argument('--weak', type=int, default=0, help='weaken init image')
parser.add_argument('--model', type=str, default="unet2", help='model architecture: unet0, unet1, unet2, unetcn0')
parser.add_argument('--spher', action="store_true", help='use spherical loss')

parser.add_argument('--steps', type=int, default=50, help='sampling steps')
parser.add_argument('--eta', type=float, default=0.5, help='ddim eta')

parser.add_argument('--c', type=float, default=0.5, help='adjust im values')
parser.add_argument('--clampim', action="store_true", help='clamp img values')

parser.add_argument('--canvasSize', type=int, default=1024, help='image size')
parser.add_argument('--tilemin', type=int, default=512, help='image size')
parser.add_argument('--tilemax', type=int, default=1024, help='image size')
parser.add_argument('--tiles', type=int, default=64, help='image size')
parser.add_argument('--grid', action="store_true", help='')

parser.add_argument('--postproc', action="store_true", help='use post processing')
parser.add_argument('--contrast', type=float, default=1, help='contrast, 1 for neutral')
parser.add_argument('--saturation', type=float, default=1, help='saturation, 1 for neutral')
parser.add_argument('--gamma', type=float, default=1, help='gamma, 1 for neutral')
parser.add_argument('--unsharp', type=float, default=0, help='unsharp mask')
parser.add_argument('--eqhist', type=float, default=0., help='histogram eq level')
parser.add_argument('--median', type=int, default=0, help='median blur kernel size, 0 for none')
parser.add_argument('--c1', type=float, default=0., help='do not use')
parser.add_argument('--c2', type=float, default=1., help='do not use')
parser.add_argument('--sharpenlast', action="store_true", help='do not use')
parser.add_argument('--sharpkernel', type=int, default=3, help='sharpening kernel')
parser.add_argument('--ovl0', type=float, default=0, help='blend original with blurred image')
parser.add_argument('--bil', type=int, default=0, help='bilateral filter kernel')
parser.add_argument('--bils1', type=int, default=75, help='bilateral filter sigma for color')
parser.add_argument('--bils2', type=int, default=75, help='bilateral filter sigma for space')
parser.add_argument('--noise', type=float, default=0., help='add noise')

parser.add_argument('--latest', action="store_true", help='save latest image for display')
parser.add_argument('--rsort', action="store_true", help='sort input files randomly')

parser.add_argument('--icontrast', type=float, default=1, help='input contrast, 1 for neutral')
parser.add_argument('--isaturation', type=float, default=1, help='input saturation, 1 for neutral')
parser.add_argument('--igamma', type=float, default=1, help='input gamma, 1 for neutral')
parser.add_argument('--ieqhist', type=float, default=0., help='histogram eq level')

parser.add_argument('--blend', type=float, default=0., help='how much of the original to blend in')

parser.add_argument('--onorm', action="store_true", help='normalize output image')

parser.add_argument('--ifull', action="store_true", help='full range at input')

parser.add_argument('--uniq', action="store_true", help='store output if not exists')

opt = parser.parse_args()

mtype = opt.model

if opt.h == 0:
    opt.h = opt.imageSize

if opt.w == 0:
    opt.w = opt.imageSize
    

name = opt.name #"out5/testcd"
steps = opt.steps
bs = 1
ifn = opt.image 



def load_model(fn):
  data = torch.load(fn)

  try:
    print("loaded "+fn+", correct mults: "+",".join(str(x) for x in data['mults']))
  except:
    print("loaded "+fn+", no mults stored")

  m = "ema" if opt.ema else "model"
  dd = data[m].copy()
  
  # if using DDIM remove original scheduler steps
  
  if opt.steps < dd['betas'].shape[0]:
    sched_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2']
    for k in sched_keys:
       del dd[k]

  return dd, data
  
dd, data = load_model(opt.load)

dd_ = {}
for k in dd.keys():
    v = dd[k]
    k_ = k.replace("denoise_fn.","")
    dd_[k_] = v  

#print(data.keys())
        
if 'mtype' in data:
    mtype = data['mtype']
elif 'opt' in data: 
    mtype = data['opt'].model
else:
    mtype = "unet2"
    print("model type not given, assuming unet2")    
    
    
if mtype == "unet0":
  from alt_models.Unet0 import Unet
elif mtype == "unet0k5":
  from alt_models.Unet0k5 import Unet
elif mtype == "unet1":
  from alt_models.Unet1 import Unet
elif mtype == "unet2":
  from alt_models.Unet2 import Unet    
elif mtype == "unetcn0":    
  from alt_models.UnetCN0 import Unet
elif mtype == "tinyunet":
  from alt_models.tinyunet import TinyUNet as Unet 
elif mtype == "tinyunet_with_attention":
  from alt_models.tinyunet_with_attn import TinyUNetWithAttn as Unet   
elif mtype == "tinyunet_with_attention3":
  from alt_models.tinyunet_with_attn3 import TinyUNetWithAttn as Unet   
  
# initialize model and diffusion

opt.mults = data['mults']

# --- NEW: detect prediction type (eps / x0) ---
if 'pred' in data:
    opt.pred = data['pred']          # e.g. "eps" or "x0"
elif 'opt' in data and hasattr(data['opt'], 'pred'):
    opt.pred = data['opt'].pred
else:
    opt.pred = "eps"                 # default for older models
    print("pred type not given, assuming 'eps'")

print("Prediction type:", opt.pred)    

model = Unet(
    dim = 64,
    dim_mults = opt.mults # (1, 2, 4, 8)
).cuda()


model.load_state_dict(dd_, strict=False)


# Setup once:
cfg = CutoutConfig(
    base_low=64,
    base_high=opt.imageSize,          # or current image shorter side
    num_min=4,
    num_max=64,
    strategy="mixture",     # try "loguniform" too
    collapse_range_at_right=True,
    margin_frac=0.02,
    out_size=224,
    clip_normalize=True,
)

cutoutSampler = GpuCutoutSampler(cfg)
cut = cutoutSampler.sample

if opt.textw > 0:
    perceptor, clip_preprocess = clip.load('ViT-B/32', jit=False)
    perceptor = perceptor.eval()
    cnorm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

text = opt.text 

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

transform = transforms.Compose([transforms.Resize((opt.h, opt.w)), transforms.ToTensor()])

if opt.tgt_image != "":   
  if opt.tgt_image == "init":
    imS = imT_.clone()
  else:
    imS = transform(Image.open(opt.tgt_image).convert('RGB')).float().cuda().unsqueeze(0)
    imS = (imS * 2) - 1

if opt.img_prompt != "" and opt.imgpw > 0:   
    imP = transform(Image.open(opt.img_prompt).convert('RGB')).float().cuda().unsqueeze(0)
    nimg = imP.clip(0,1)
    nimg = cut(nimg, slider=opt.cuts)
    imgp_enc = perceptor.encode_image(nimg.detach()).detach()

if opt.text != "" and opt.textw > 0:
    tx = clip.tokenize(text)                        # convert text to a list of tokens 
    txt_enc = perceptor.encode_text(tx.cuda()).detach()   # get sentence embedding for the tokens
    del tx
    
def tilexy():
    th = (random.randint(opt.tilemin, opt.tilemax)//64)*64
    tw = (random.randint(opt.tilemin, opt.tilemax)//64)*64
    ty = random.randint(0, opt.h - th)
    tx = random.randint(0, opt.w - tw)
    #print(ty, tx, th, tw)
    return (ty, tx, th, tw)


    
def tileList():
    tlist = []
    if opt.grid:
        size = opt.tilemin
        nx = opt.w // size
        ny = opt.h // size
        tx = 0
        print(nx, ny)
        for ix in range(0, nx):
            ty = 0
            for iy in range(0, ny):
              tlist.append((ty, tx, size, size))
              ty += size
            tx += size
    else:
        for i in range(0, opt.tiles):
            tlist.append(tilexy())                          
    random.shuffle(tlist)
    return tlist

def getTile(field, pos):
    ty, tx, th, tw = pos
    tile = field[:, :, ty:ty+th, tx:tx+th]
    return tile
    
def putTile(field, pos, content):
    ty, tx, th, tw = pos
    orig = field[:, :, ty:ty+th, tx:tx+th]
    if opt.blend > 0:
      field[:, :, ty:ty+th, tx:tx+th] = (1- opt.blend) * content + opt.blend * orig
    else:
      field[:, :, ty:ty+th, tx:tx+th] = content
    return field    



@torch.enable_grad()
def cond_fn(x, t, x_s):
    global opt    
    x_is_NaN = False
    x.grad = None
    x.requires_grad_()
    n = x.shape[0]         
    
    x_s.requires_grad_()
    x_grad = torch.zeros_like(x_s)
                    
    loss = 0
    losses = []

    nimg = None

    if opt.text != "" and opt.textw > 0:
        nimg = x_s.clip(-1, 1) + 0.5    
        #nimg = cut(nimg, cutn=opt.cutn, low=opt.low, high=opt.high, norm = cnorm)
        nimg = cutoutSampler.sample(nimg, slider = opt.cuts)
        
        # get image encoding from CLIP
 
        img_enc = perceptor.encode_image(nimg) 
  
        # we already have text embedding for the promt in txt_enc
        # so we can evaluate similarity
     
        if opt.spher:
            loss = opt.textw * spherical_dist_loss(txt_enc.detach(), img_enc).mean()
        loss = opt.textw*10*(1-torch.cosine_similarity(txt_enc.detach(), img_enc)).view(-1, bs).T.mean(1)
        losses.append(("Text loss",loss.item())) 
        if opt.tdecay < 1.:
            opt.textw = opt.tdecay * opt.textw
        x_grad += torch.autograd.grad(loss.sum(), x_s, retain_graph = True)[0]

        #del nimg

    if opt.img_prompt != "" and opt.imgpw > 0:
        if nimg == None:
            nimg = x_s.clip(-1, 1) + 0.5     
            #nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
            nimg = cutoutSampler.sample(nimg, slider = opt.cuts)
            img_enc = perceptor.encode_image(nimg)
            del nimg
        loss1 = opt.imgpw*10*(1-torch.cosine_similarity(imgp_enc, img_enc)).view(-1, bs).T.mean(1)  
        losses.append(("Img prompt loss",loss1.item())) 
        loss = loss + loss1     
        
        x_grad += torch.autograd.grad(loss1.sum(), x_s, retain_graph = True)[0]
        
    if opt.tgt_image != "":
          loss_ = opt.ssimw * (1 - ssim((x_s+1)/2, (imS+1)/2)).mean() 
          losses.append(("Ssim loss",loss_.item())) 
          loss = loss + loss_    
          
          x_grad += torch.autograd.grad(loss_.sum(), x_s, retain_graph = True)[0]
    
    if torch.isnan(x_grad).any()==False:
        grad = -torch.autograd.grad(x_s, x, x_grad)[0]
    else:
      x_is_NaN = True
      grad = torch.zeros_like(x)             
          
    del x, x_s, x_grad, loss
          
    return opt.lr*grad.detach()



# important! will not work with diffusers default betas 

def make_betas(timesteps):
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    
    return betas.numpy()
    

scheduler = DDIMScheduler(num_train_timesteps=opt.trainsteps, prediction_type = "epsilon", trained_betas = make_betas(opt.trainsteps), clip_sample=False)
scheduler.set_timesteps(opt.steps, device="cpu")


def get_timesteps(skip = opt.skip):
    offset = scheduler.config.get("steps_offset", 0)
    
    # get the original timestep using init_timestep
    init_timestep = opt.skip + offset
    init_timestep = min(skip, opt.steps)
    
    timesteps = scheduler.timesteps[init_timestep:]
    t_start = max(opt.steps - init_timestep, 0)

    return {'timesteps':timesteps, 't_start': t_start}
       
def getx(ifn=None):
    global timesteps
    init_noise = torch.zeros(bs,3,opt.h,opt.w).normal_(0,1) #.cuda()
    im = None
    if ifn != None:   
        x = transform(Image.open(ifn).convert('RGB')).float().unsqueeze(0)

        if opt.ieqhist > 0:
            x = equalize_clahe(
                x,
                clip_limit=opt.ieqhist,
                grid_size=(8, 8),
            )

        if opt.icontrast != 1:
            x = adjust_contrast(x, opt.icontrast)
        if opt.igamma != 1:
            x = adjust_gamma(x, opt.igamma)
        #if opt.brightness != 0:
        #    imT = adjust_brightness(imT, opt.brightness)
        if opt.isaturation != 1:
            x = adjust_saturation(x, opt.isaturation)
        im = x.clone()

        if opt.ifull:
            x = (x * 2) - 1
        else:
            x -= 0.5

        x = opt.mul*scheduler.add_noise(opt.imb * x, init_noise, timesteps[opt.weak])
        #print(x.min(), x.max(), x.mean(), x.std())
    else:
        x = opt.mul*init_noise * scheduler.init_noise_sigma

    #print(x.shape)
    return x, im

if os.path.isdir(opt.image):
    imgList = os.listdir(opt.image)
    inputlist = []

    '''
    for fname in imgList:
      # skip non-images
      ext = fname.split('.')[-1].lower()
      imgname = fname.split('.')[0].lower()
      if not ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
        continue
        
      fpath = opt.image+os.sep+ fname
      inputlist.append(fpath)
    '''

    for fname in imgList:

        ext = fname.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
            continue

        imgname = os.path.splitext(fname)[0]

        if opt.uniq:
            outpath = os.path.join(opt.dir, imgname + "-final.png")
            if os.path.exists(outpath):
                continue

        fpath = os.path.join(opt.image, fname)
        inputlist.append(fpath)

    if opt.rsort:
        random.shuffle(inputlist)
    else:    
        inputlist.sort() # todo proper numeric sort
    
elif opt.image == "":
    inputlist = [None]

else:     
    inputlist = [opt.image]
    
timesteps = get_timesteps(opt.skip)['timesteps']    

#scheduler.to("cuda")

ctr = 0
for inp in inputlist:
  print(inp)    


  if name != "":
    oname = name+"-"+str(ctr)
  else:
    oname = inp.split("/")[-1].split(".")[0]

  print(oname)  
  xf, im = getx(inp)
  #x = x.cuda()
  #im = im.cuda()
  
  # prepare for tiling
  
  imTf = im.clone()
  #save_image((imTf.clone()+0.5, opt.dir+os.sep+name+"-init.png")
  
  tilelist = tileList()
  print(tilelist)
  
  imTf_ = {}
  #for k in opt.saveIters:
  #    imTf_[str(k)] = imTf.clone()
  if opt.saveorig:
      im_ = imTf.clone().cpu()
      if opt.postproc:
        im_ = pprocess(im_, opt) 
        #im_ += 1
        #im_ /= 2
      save_image(im_, opt.dir+os.sep+oname+"-0.png", normalize=opt.onorm)
      
      if opt.latest:
        time.sleep(12)  # needed to allow the previous image to show as latest
        save_image(im_, "/var/www/html/latest.jpg", normalize=opt.onorm)




  for tn in range(0, len(tilelist)):
    j = 0
    tile = tilelist[tn] #tilexy(opt.h, opt.w)
    x = getTile(xf, tile).cuda()
    
    print(ifn, tn)
    
    k = tn
    if opt.saveorig:
        k += 1

    for i in tqdm(timesteps):
      t = torch.tensor([i] * bs, device='cuda').cuda().detach()
      
      if (opt.text!="" and opt.textw > 0):
         with torch.enable_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x.requires_grad_() 
                # timesteps: CPU for scheduler, GPU for model
                t_sched = t.to('cpu')                      # for scheduler.step and alphas_cumprod
                t_model = t.to(device=x.device)            # for the UNet and cond_fn
        
                pred = model(x, t_model).to(x.dtype)
                 
                #alpha_prod_t = scheduler.alphas_cumprod[t]
                #print(scheduler.alphas_cumprod.device, t_sched.device)
                alpha_prod_t = scheduler.alphas_cumprod[t_sched].to(device=x.device, dtype=x.dtype)
                beta_prod_t = 1 - alpha_prod_t

                # Handle ε vs x0
                if opt.pred == "eps":
                    eps = pred
                    # convert to x0 for convenience (guidance uses x0-space)
                    pred_original_sample = (x - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()

                else:  # opt.pred == "x0"
                    x0_pred = pred
                    pred_original_sample = x0_pred
                    # convert x0 → ε for scheduler
                    eps = (x - alpha_prod_t.sqrt() * x0_pred) / beta_prod_t.sqrt()
               
                #pred_original_sample = (x - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
                #fac = torch.sqrt(beta_prod_t) #.cuda()
                sample = pred_original_sample #* (fac) + x * (1 - fac)
         
                grad = cond_fn(x, t_model, sample).to(device=x.device, dtype=x.dtype)
                eps = eps - torch.sqrt(beta_prod_t) * grad
                            
                #print(eps.device, t.device, x.device)            
                s = scheduler.step(eps.cpu(), t_sched, x.cpu(), eta=opt.eta) #
                x = s['prev_sample'].cuda().detach() 
                x_s = s['pred_original_sample'].detach() 
                del sample, grad, alpha_prod_t, beta_prod_t
               
      else:              
          with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16): 
                       x.requires_grad_() 
                       t = t.to(device=x.device)

                       # raw model output
                       pred = model(x, t).to(x.dtype)
                       
                       alpha_prod_t = scheduler.alphas_cumprod[t.cpu()].to(device=x.device, dtype=x.dtype)
                       beta_prod_t = 1 - alpha_prod_t

                       # Handle ε vs x0
                       if opt.pred == "eps":
                            eps = pred
                            # convert to x0 for convenience (guidance uses x0-space)
                            pred_original_sample = (x - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()

                       else:  # opt.pred == "x0"
                            x0_pred = pred
                            pred_original_sample = x0_pred
                            # convert x0 → ε for scheduler
                            eps = (x - alpha_prod_t.sqrt() * x0_pred) / beta_prod_t.sqrt()
                       

                       #print(eps.device, t.device, x.device)             
                       s = scheduler.step(eps.cpu(), t.cpu(), x.cpu(), eta=opt.eta) #
                       x = s['prev_sample'].cuda().detach() 
                       x_s = s['pred_original_sample'].detach() 
                       del alpha_prod_t, beta_prod_t
                  
      #del noise 
    

    im = (x.clone()+opt.c)
       
    imTf = putTile(imTf, tile, im.cpu().detach()) 
      


    im_ = imTf.clone().cpu()

    print(im_.min(), im_.max())

    im_ = im_.clamp(0,1)

    print(im.min(), im.max(), imTf.min(), imTf.max(), im_.min(), im_.max())
    if opt.postproc:
          im_ = pprocess(im_, opt) 
          im_ -= im_.min()
          im_ /= im_.max()

    if opt.saveiters:
            save_image(im_, opt.dir+os.sep+oname+"-"+str(k)+".png", normalize=opt.onorm)
            if opt.latest:
                save_image(im_, "/var/www/html/latest.jpg", normalize=opt.onorm)
             
  save_image(im_, opt.dir+os.sep+oname+"-final.png", normalize=opt.onorm)               

  if opt.latest:
    save_image(im_, "/var/www/html/latest.jpg", normalize=opt.onorm)

  ctr += 1
