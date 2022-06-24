import torch
import torch.nn as nn
from implementation.srcnn_utils import zero_upsampling

### Classes ###
class Discriminator(nn.Module):
    def __init__(self, c = 3, ndf = 64):
        """
        AF(n, ndf) = a 3 layer discriminator for a super resolution network

        Representation Invaraint:
            - True

        Represnetation Exposure:
            - Safe
        """
        super(Discriminator, self).__init__()
        
        layers = [
            nn.Conv2d(c, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        ] + self.block(ndf) + self.block(ndf * 2) + self.block(ndf * 4)
        layers.append(nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1))
        
        self.main = nn.Sequential(*layers)
    
    def block(self, in_c):
        """
        """
        return [
            nn.Conv2d(in_c, in_c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_c * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

    def forward(self, input):
        """ Forward porpagation of input through the net """
        return self.main(input)