from xurdif import GaussianDiffusion, Trainer
import torch
from torchvision import transforms
import lpips 


'''

xurdiffusion

@htoyryla June 2023, 2025

basic trainer

'''

import argparse

parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--images', type=str, default="", help='path to images')
parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
parser.add_argument('--steps', type=int, default=1000, help='number of diffusion steps')
parser.add_argument('--accum', type=int, default=10, help='number of iterations per gradient update')
parser.add_argument('--trainsteps', type=int, default=100000, help='number of iterations')
parser.add_argument('--dir', type=str, default="train", help='folder for storing sampled images')
parser.add_argument('--name', type=str, default="oma", help='basename for storing sampled images')
parser.add_argument('--amp', action="store_true", help='use automatic mixed precision')
parser.add_argument('--imageSize', type=int, default=512, help='image size')
parser.add_argument('--batchSize', type=int, default=2, help='batch size')
parser.add_argument('--saveEvery', type=int, default=100, help='image and model save frequency')
parser.add_argument('--losstype', type=str, default="l2", help='loss type: l1 or l2')
parser.add_argument('--l1w', type=float, default=1., help='L1 loss weight')
parser.add_argument('--ssimw', type=float, default=10., help='SSIM loss weight')

parser.add_argument('--load', type=str, default="", help='path to pth file')
parser.add_argument('--nostrict', action="store_true", help='')
parser.add_argument('--mults', type=int, nargs='*', default=[1, 1, 2, 2, 4, 4, 8, 8], help='')
parser.add_argument('--nsamples', type=int, default=2, help='how many samples to generate')
# KILN: fixed seed for the snapshot preview so a run's thumbnails form a
# comparable timeline instead of a different random image every time.
# Negative disables seeding and restores the original behaviour.
parser.add_argument('--sampleSeed', type=int, default=-1, help='seed for snapshot samples (-1 = random)')
parser.add_argument('--model', type=str, default="unet2", help='model architecture: unet0, unetok5, unet1,unetcn0')

parser.add_argument('--fit', type=str, default="resize", help='resize | crop')

parser.add_argument('--pred', type=str, default="eps", help='prediction type: eps, x0')
# KILN: the edge-weighted L1 in GaussianDiffusion.p_losses was unconditional.
# Kiln offers it as a training option, so it needs a switch. Phrased as an
# opt-out so a bare run trains exactly as it always did -- note that xurdif2's
# --use_edges is the opposite polarity and defaults OFF, so a future move to
# the v2 trainer must invert this.
parser.add_argument('--noEdges', action="store_true", help='disable the edge-weighted L1 loss')
# KILN: the configurable-attention layout for tinyunet_conf_attention, spelled
# as upstream's xurdiftrainer26b.py spells it ("-1:linear,mid:full"). Pass it as
# one token, --attn=-1:linear,mid:full, because the value begins with '-'.
# Transplanted from xurdiftrainer26b.py, which runs on the unfinished v2 engine.
parser.add_argument("--attn", type=str, default=None, help='attention layout, e.g. "-1:linear,mid:full"')
# KILN: train from a dataset snapshot (a file list plus a framing/augmentation
# recipe) instead of globbing --images. Kiln's datasets read images where they
# are and augment at training time; the dataset class lives in Kiln
# (app/core/engine/train_data.py). Absent, training reads --images as before.
parser.add_argument("--manifest", type=str, default=None, help='Kiln dataset snapshot (json)')


opt = parser.parse_args()

mtype = opt.model

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
elif mtype == "unet2d":
  from alt_models.Unet2d import Unet
elif mtype == "tinyunet":
  from alt_models.tinyunet import TinyUNet as Unet
elif mtype == "tinyunet_with_attention":
  from alt_models.tinyunet_with_attn import TinyUNetWithAttn as Unet
elif mtype == "tinyunet_with_attention3":
  from alt_models.tinyunet_with_attn3 import TinyUNetWithAttn as Unet
elif mtype == "tinyunet_conf_attention":   # KILN: see --attn above
  from alt_models.tinyunet_conf_attn import TinyUNetWithAttn as Unet
else:
  print("Unsupported model: "+mtype)
  exit()


if opt.fit == "resize":
    xf = transforms.Compose([
          transforms.Resize((opt.imageSize, opt.imageSize)),
          transforms.RandomHorizontalFlip(),
          #transforms.CenterCrop(opt.imageSize),
          transforms.ToTensor(),
          transforms.Lambda(lambda t: t - 0.5) # value range -0.5 - 0.5
    ])
elif opt.fit == "crop":
   xf = transforms.Compose([
          #transforms.Resize(opt.imageSize),
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop((opt.imageSize, opt.imageSize)),
          transforms.ToTensor(),
          transforms.Lambda(lambda t: t - 0.5) # value range -0.5 - 0.5
    ])  
else:    
   print("unknown value for fit: ",opt.fit)
   exit()      

# KILN: the conf model takes its layout as a dict; Trainer.save writes both the
# string and the dict into the checkpoint so it can be rebuilt on load.
if "conf" in opt.model:
    from alt_models.tinyunet_conf_attn import parse_attn_config
    opt.attn_config = parse_attn_config(opt.attn)
    model = Unet(
        dim = 64,
        dim_mults = tuple(opt.mults),
        attn_config = opt.attn_config
      ).cuda()
else:
    model = Unet(
        dim = 64,
        dim_mults = tuple(opt.mults)
      ).cuda()

print(model)

model = model.cuda()

#lpips_fn = lpips.LPIPS(net='vgg').to("cuda")  # TODO!!!
#lpips_fn.eval()  # always eval mode

diffusion = GaussianDiffusion(
    model,
    image_size = opt.imageSize,
    timesteps = opt.steps,   # number of steps
    ssimw = opt.ssimw,
    l1w = opt.l1w,
    pred=opt.pred,
    use_edges = not opt.noEdges   # KILN: see --noEdges above
    #loss_type = opt.losstype   # L1 or L2,
    #lpips_fn = lpips_fn
).cuda()


# KILN: see --manifest above. The snapshot's recipe decides flips; the stock
# transform's unconditional RandomHorizontalFlip is not applied on top.
manifest_ds = None
if opt.manifest:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    from app.core.engine.train_data import ManifestDataset
    manifest_ds = ManifestDataset.from_snapshot(opt.manifest, opt.imageSize, fit=opt.fit, engine="xurdif")
    print("dataset:", len(manifest_ds), "images from", opt.manifest)

trainer = Trainer(
    diffusion,
    opt.images,
    image_size = opt.imageSize,
    train_batch_size = opt.batchSize,
    train_lr = opt.lr,
    save_and_sample_every = opt.saveEvery,
    train_num_steps = opt.trainsteps,         # total training steps
    gradient_accumulate_every = opt.accum,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = opt.amp,                       # turn on mixed precision training with apex
    results_folder = opt.dir,
    nsamples = opt.nsamples,
    transform = xf, #,
    opts = opt,
    pred = opt.pred,
    dataset = manifest_ds   # KILN: see --manifest above
)

if opt.load != "":
    # KILN: checkpoints now carry 'opt', an argparse Namespace, which torch 2.6+
    # refuses under its default weights-only load. These are the run's own files.
    data = torch.load(opt.load, weights_only=False)
    #trainer.load(data)
    trainer.step = data['step']
    trainer.model.load_state_dict(data['model'], strict=not opt.nostrict)
    trainer.ema_model.load_state_dict(data['ema'], strict=not opt.nostrict)
    try:
      print("loaded "+opt.load+", correct mults: "+",".join(str(x) for x in data['mults']))
    except:
      print("loaded "+opt.load+", no mults stored")

#if opt.losstype == "lpips":
#  trainer.model._lpips_fn = lpips_fn

     
trainer.train()
