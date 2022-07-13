import os
import cv2
import torch
import shutil
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    """
    AF(n, ndf) = a vgg19 feature extractor for loss computation

    Representation Invaraint:
        - True

    Represnetation Exposure:
        - Safe
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

def zero_upsampling(img: torch.Tensor, factors: int) -> torch.Tensor:
    """
    Upsamples an image by assigning each input pixel to its corresponding pixel at high resolution 
    and leaving all the missing pixels around it as zeros. The location of each input pixel falls equally 
    in between factor pixels in the high resolution, where factor is on of the upsampling factors

    For a 3D image and factors = (k1, k2)
        original index (n, i, j, ...) -> new index (n, i', j', ...) where (n, i', j', ...) = (n, i*k1, j*k2, ...)
    For a 4D image and factors = (k1, k2)
        original index (n, m, i, j, ...) -> new index (n, m, i', j', ...) where (n, m, i', j', ...) = (n, m, i*k1, j*k2, ...)

    Currently supports only 3D and 4D images

    Example
        upsampling by a factors = (2, 2) and 3D image
                         [[[1, 0, 2, 0, 3, 0],        
        [[[1, 2, 3]],      [0, 0, 0, 0, 0, 0]],
         [[4, 5, 6]],  -> [[4, 0, 5, 0, 6, 0],    
         [[7, 8, 9]]]      [0, 0, 0, 0, 0, 0]],    
                          [[7, 0, 8, 0, 9, 0],
                           [0, 0, 0, 0, 0, 0]]]

    Inputs
        :img: <Tensor> representing the image of size C x H x W or D x C x H x W
        :factors: <tuple<int>> of length 2 containing the factors, (n, m), of how much to upscale each dimension of the image by 

    Outputs
        :returns: <Tensor> of the upsampled image with the same dimensions as the input img, i.e 1 x C x H x W or D x C x H x W
    """
    x = len(factors)
    img_size = list(img.shape)
    dim = len(img_size)
    upscaled_img_size = img_size[0: dim - x] + [img_size[dim - x + i] * factors[i] for i in range(x)]
    upscaled_img = np.zeros(upscaled_img_size)
    print(f"upscaling img from {img_size} -> {upscaled_img_size}")

    n, m = factors
    if dim == 3: upscaled_img[:, ::n, ::m] = img
    if dim == 4: upscaled_img[:, :, ::n, ::m] = img


    return torch.Tensor(upscaled_img).unsqueeze(0)

def create_video(frames, home_dir, scale_factor = 1):
    """
    Creates and saves a video from input frames scaled by scale_factor (default is 1)

    Inputs
        :frames: <list> frames to turn into a video
        :home_dir: <str> path to where the video should be stored
        :scale_factor: <int> to scale the video's resolution by, 1 by default

    Outputs
        :returns: path to the cretaed video 
    """
    print("Creating Video ...")
    video_name = input("Video name: ")
    frames_path = f"{home_dir}/outputs/videos/frames"
    video_path = f"{home_dir}/outputs/videos/{video_name}.mp4"

    os.makedirs(frames_path)
    for i in range(len(frames)):
        frame = frames[i]
        h, w, _ = frame.shape
        if scale_factor > 1:
            low_res_frame = cv2.resize(frame,  (w // scale_factor, h // scale_factor))
            low_res_frame = cv2.resize(low_res_frame,  (w, h))
            frame_to_write = low_res_frame
        else: frame_to_write = frame
        
        cv2.imwrite(f"{frames_path}/frame_{i}.png", frame_to_write)
    os.system(f"ffmpeg -r 30 -i {frames_path}/frame_%d.png -vcodec mpeg4 -y {video_path}")
    shutil.rmtree(frames_path)

    return video_path
    
def random_init(m):
    """
    Randomly initiaties the weights such that the weights have a mean of 0 and a standard deviation of 0.2 

    Inputs
        :m: <nn.Layer> to be randomly initalized
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def display(image, permute = None, scale = 1, dtype = np.uint8):
    """
    Displays an image transposed as needed

    Input
        :image: <np.ndarray> representing the image to be displayed
        :permute: <list> | None of how dimensions should be permuted, default is None
        :scale: <int> factor to scale pixels by, default is 1
        :dtype: <np.???> type of the pixel values, default is np.uint8
    """
    if permute is not None:
        image = (image.transpose(permute) * scale).astype(dtype)
    print(f"Image shape: {image.shape}")
    plt.subplots(1, 1)
    plt[0].imshow(image)
    plt.show()
