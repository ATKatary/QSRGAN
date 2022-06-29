import torch
import torch.nn as nn
from implementation.srcnngan_utils import zero_upsampling

### Classes ###
class SRCNN(nn.Module):
    """
    AF(net, k, c) = a 3 layer CNN with ReLU activations and upscale factor of 2 by default on c channel images

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self, k = 2, c = 3, ndf = 64) -> None:
        ### Representation ###
        super(SRCNN, self).__init__()
        self.upscale_factor = (k, k)

        self.net = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=9, padding=2, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate'),
            nn.Conv2d(32, c, kernel_size=5, padding=2, padding_mode='replicate')
        )

    def forward(self, input, mode = 'bilinear'):
        """ Forward porpagation of input through the net with given upscaling mode (default bilinear) """
        if mode == "zero": f = zero_upsampling
        else: f = nn.Upsample(scale_factor=self.upscale_factor, mode=mode)
        
        return self.net(f(input))
    
    def initiate(self, device, pretrained_weights):
        """ Initializes the model using pretrained weights """
        print("Initializing ...")
        model = nn.DataParallel(self).to(device)
        model.load_state_dict(pretrained_weights)
        model.eval()
        return model