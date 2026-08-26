from kornia.enhance import adjust_contrast, adjust_brightness, adjust_saturation, adjust_gamma
from kornia.filters import unsharp_mask, median_blur
from kornia.enhance.equalization import equalize_clahe
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def laplace_sharpen(image: torch.Tensor, blur_radius: int, alpha: float):
        if blur_radius == 0:
            return (image,)
                 
        device = image.device
        batch_size, channels, height, width = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1).to(device)
        
        #print(kernel.shape)

        sharpened = F.conv2d(image, kernel, padding=center, groups=channels)
        #sharpened -= sharpened.min()
        #sharpened /= sharpened.max() #- torch.clamp(sharpened, 0, 1)
         
        #result = (1 - alpha) * image + alpha * sharpened 
        result = torch.clamp(sharpened + image, 0, 1)
         
        return (result,)

def safe_normalize(im):
    im = im - im.min()
    maxval = im.max()
    return im / maxval if maxval > 0 else im * 0        

def pprocess(imT, opt):

        # normalize to 0..1 (needed for contrast etc adjustments)
        #print(imT)

        #imT -= imT.min()
        #imT /= imT.max()
        imT = safe_normalize(imT)

        if opt.contrast != 1:
            imT = adjust_contrast(imT, opt.contrast)
        if opt.gamma != 1:
            imT = adjust_gamma(imT, opt.gamma)
        #if opt.brightness != 0:
        #    imT = adjust_brightness(imT, opt.brightness)
        if opt.saturation != 1:
            imT = adjust_saturation(imT, opt.saturation)
 
       # usually the histogram is narrow, optionally equalize to get a wider histogram
            
        imT_ = imT.clone() #.cpu()            

        if opt.median > 0:
            #imT = median_blur(imT, (opt.median, opt.median)) 
            # alternative implementation of median blur using opencv
            # to get rid of out of memory crashes using kornia
            # this is a hack: converts from tensor to opencv and back 
           
            img = imT.clone()[0].cpu().permute(1,2,0).numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.medianBlur(img, opt.median)/255.
            imT = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
        elif opt.bil > 0:
            # alternative implementation of median blur using opencv
            # to get rid of out of memory crashes using kornia
            # this is a hack: converts from tensor to opencv and back 
            
            img = imT.clone()[0].cpu().permute(1,2,0).numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.bilateralFilter(img, opt.bil, opt.bils1, opt.bils2)/255.
            imT = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()    
            
            
        if opt.unsharp > 0:
            imT = unsharp_mask(imT, (opt.sharpkernel, opt.sharpkernel), (opt.unsharp, opt.unsharp))    
            
        if hasattr(opt, 'laplace_radius') and (opt.laplace_radius > 0):
              #fake = fake - fake.min()
              imT = laplace_sharpen(imT, opt.laplace_radius, opt.laplace_alpha)[0] 
              #imT -= imT.min()
              #imT /= imT.max()    
              imT = safe_normalize(imT)


        #print(imT, imT_)
        if opt.ovl0 != 0:
            imT = opt.ovl0 * imT_.to(imT.get_device()) + (1 - opt.ovl0) * imT


        #imT -= imT.min()
        #imT /= imT.max()   
        imT = safe_normalize(imT)


        if opt.eqhist > 0:
            #print(imT, opt.eqhist)
            imT = equalize_clahe(imT, clip_limit = opt.eqhist, grid_size = (8,8))
            
        if opt.sharpenlast and opt.unsharp > 0:
            imT = unsharp_mask(imT, (opt.sharpkernel, opt.sharpkernel), (opt.unsharp, opt.unsharp))
            

        if opt.noise > 0:
            noise = np.random.normal(0, opt.noise, (opt.h, opt.w))
            noise = np.repeat(noise[np.newaxis, np.newaxis,:,:], 3, axis=1)
            noise = torch.from_numpy(noise).to(imT.device)
            imT = (imT + noise).clamp(0,1)    

        imT = 2 * imT
        imT = imT - 1

        return imT   

