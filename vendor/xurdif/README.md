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


## Update 27 July 2026

Added new files instead of committing them as new versions:

xurdiftrainer2b.py: 
* supports edge aware loss for learning noisy or edgy images
* masked training... appears to learn faster or better
* usage example:
```
python xurdiftrainer2b.py --images <path_to_train_images> --steps 1000 --trainstep 280000 --accum 10 --dir <output_folder> --imageSize 512   --batchSize 8 --saveEvery 100 --nsamples 2 --model tinyunet_with_attention3 --lr 5e-5 --losstype l1 --mults 1 2 4 4 8 --pred x0 --use_mask  --fit crop --rot --flip --use_edges [--load <model_path_for_resume>]
```

xurdif2.py:
* basic diffusion library needed for the new trainer

xurdiffer23t2.py:
* command line tool for generating images, typically using init image (or a folder of images)
* usage example
```
python xurdiffer23t2.py --dir <output_path>  --eta 0.5 --steps 50   --lr 1   --image <init_image_path>  --ema --rsort   --h 1024 --w 1536 --mul 0.98 --postproc --eqhist 0.4 --noise 0.02 --unsharp 0.4 --text "text_prompt" --textw 100 --lr 1  --load <model_path>  --skip 33 --img_prompt <image_prompt_path> --imgpw 100 
```
xurdifstile2d.py:
* tiled generation, usually meaningful when using init images
* usage example
```
python xurdifstile2d.py --dir <output_path> --eta 0.5 --steps 200   --lr 0  --load <model_path> --image <init_image_path> --skip 40  --ema --rsort  --model tinyunet_with_attention3   --h 2048 --w 3072 --tilemin 512 --grid --mul 0.9 --icontrast 1.5 --gamma 1.5 --postproc --noise 0.1 --unsharp 0.4
```

## Configurable attention

New model architecture has been added to allow placing attention at different, even multiple levels in the model, as well as selecting the type of each attention layer.

Training:

```
python xurdiftrainer26b.py  --images <path_to_train_images>  --steps 1000 --trainsteps 280000 --accum 10 --dir <output_folder> --imageSize 512 --batchSize 8 --saveEvery 100 --nsamples 2 --model tinyunet_conf_attention --attn=-1:linear,mid:full --lr 4e-4  --losstype l1 --use_edges --use_mask --mults 1 2 2 4 4 --pred x0
```
Examples of attn config:
```
mid:full              # full attention at bottleneck, original architecture
-1:linear,mid:full    # full attention at bottleneck, linear attention at one level up
-1:linear,mid:linear  # linear attention at bottleneck, linear attention at one level up
```
Generation:

```
python xurdiffer26c.py --dir <output_path> --name <base_name>  --eta 0.5 --steps 100     --w 512 --h 512  --load <path_to_checkpoint>  --rounds 100 
```

This will generate 100 samples named base_name-1.png etc. Attn_config and model architecture will be read from the checkpoint.

Generation with a folder of init images and custom post_processing:
```
python xurdiffer26c.py --dir <output_path>  --eta 0.5 --steps 50   --lr 1   --image <init_image_path>  --ema --rsort   --h 512 --w 512 --mul 0.98 --postproc --eqhist 0.4 --noise 0.02 --unsharp 0.4 --text "text_prompt" --textw 100 --lr 1  --load <model_path>  --skip 33 --img_prompt <image_prompt_path> --imgpw 100 
```



