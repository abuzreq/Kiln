# xurdif

a compact diffusion model trained with small, private datasets

## what is xurdif

This repo is essentially an updated and compact version of my urdiffusion (see https://github.com/htoyryla/urdiffusion)

Main changes: 

* Urdiffusion supported a variety of experimental Unet architectures, which has now been replaced with a new lightweight architecture with attention
* support for x0 prediction
* edge aware loss added

With these changes I have found that, using a coherent dataset of a few hundred images, a new model can be trained in as little as a few hours, even if for better quality 12 - 24 hours are recommended. Anyhow, already during the first hour or two it is possible so see what the model is learning.

## philosophy

At the core of urdiffusion lies the idea of a diffusion model as an image generator with a limited range of expression and aesthetics, hooked up with control from init images and text prompts, as well as optional guidance from target images, image prompts and why not even style guidance using Gram matrices.

It is possible to train a limited range model with a small dataset of visually similar images in 6 to 24 hours on a home GPU. With the present implementation, I have found out that it is usually better to train from scratch rather than retrain an existing model with a new set.

When you start guiding such a model with init images and prompts, it is best to start with an open mind, clear of expectations. There is bound to be a tension as to what the guidance from the prompt seems to require and what the model is able to produce. This very tension is the core creative element in urdiffusion. Experiment how the model reacts to the guidance, depending on the image material used in training. The results may be far different from what you expected, but you are still likely to find interesting and useful results and styles. Train new models and experiment with different materials, find out what works for you, collect your own model library.

<img src="https://github.com/htoyryla/urdiffusion/assets/15064373/beca5e0c-e27e-4402-b8d7-fef8acf24c60" width="720px">

## requirements 

* pytorch, torchvision, cudatoolkit
* numpy
* PIL
* pytorch_mssim
* argparse
* tqdm
* einops
* diffusers
* CLIP
* kornia
* opencv (needed by postprocessing)
* gradio

## compatibility with urdiffusion

Xurdif should be downwards compatible with models trained on urdiffusion. To ensure compatibility copy the contents of alt_models folder (in your urdiffusion installation or from urdiffusion repo) into the alt_models folder of your xurdif installation.

## Get some models

Go to https://www.dropbox.com/scl/fo/flh4pczukrrlb3ar1rfuc/AAT22M2b21Tf1yKe3Ji0HS0?rlkey=f1zdhexy36p3hffcun686m77c&dl=0 and copy the files into the models folder

## Run gradio app

To run the gradio app

```
python xurdifapp.py 
```

## To train a model

```
python xurdiftrainer.py --images path_to_your_image_files --steps 1000 --trainstep 280000 --accum 10 --dir path_for_storing_checkpoints --imageSize 512 --batchSize 8 --saveEvery 100 --nsamples 2 --model tinyunet_with_attention3 --losstype l1 --ssimw 0 --lr 4e-4 --mults 1 2 2 2 --pred x0
```

The image path should point directly to the folder containing to the image files. Subfolders are not scanned.

The above command will resize the training images to the imageSize (512px), forcing to square if necessary. Alternatively use --fit crop so that random crops of 512x512px are used.

Look at the terminal output for training progress, usually the loss should decrease quite fast at the beginning. Later on the loss may not decrease anymore but the image quality typically still improves.

In the checkpoint folder you will see both the saved model files as well as sample output from each stored model file. You can the copy selected models into your models folder... just rename them into something meaningful so that you will be able to differentiate between different models in the gradio app.

