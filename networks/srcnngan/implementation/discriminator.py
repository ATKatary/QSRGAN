import torch
import torch.nn as nn
from implementation.srcnn_utils import zero_upsampling

### Classes ###
class Discriminator(nn.Module):
    def __init__(self, c = 3, ndf = 64):
        """
        AF(n, ndf) = a 4 layer 

        Representation Invaraint:
            - True

        Represnetation Exposure:
            - Safe
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(c, ndf, kernel_size=5, padding=2, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(ndf, ndf * 2, kernel_size=1, padding=2, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(ndf * 2, 1, kernel_size=9, padding=2, padding_mode='zeros'),
            nn.Sigmoid()
        )

    def forward(self, input):
        """ Forward porpagation of input through the net """
        return self.main(input)