import torch
import torch.nn as nn
from implementation.srcnn_utils import zero_upsampling

### Classes ###
class SRCNN(nn.Module):
    """
    AF(k, c, ndf, n_residual) = a 21 layer CNN with ReLU like activations and upscale factor of 2 by default on c channel images

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self, k = 2, c = 3, ndf = 64, n_residual = 16) -> None:
        ### Representation ###
        super(SRCNN, self).__init__()
        self.upscale_factor = (k, k)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, ndf, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        
        self.residuals = nn.Sequential(*[ResidualBlock(ndf) for _ in range(n_residual)])

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf, 0.8),
            nn.ReLU(True),
        )

        self.upsample = nn.Sequential(*(self.upsample_bloc(k)*2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf, c, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def upsample_bloc(self, k):
        """
        An upsampling block with a scale factor of k
        """
        return [
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=k),
            nn.PReLU()
        ]

    def forward(self, input, mode = 'bilinear'):
        """ Forward porpagation of input through the net with given upscaling mode (default bilinear) """
        conv1_output = self.conv1(input)
        output = self.residuals(conv1_output)

        conv2_output = self.conv2(output)
        output = torch.add(conv1_output, conv2_output)

        output = self.upsample(output)
        return self.conv3(output)

    
    def initiate(self, device, pretrained_weights):
        """ Initializes the model using pretrained weights """
        print("Initializing ...")
        self.load_state_dict(pretrained_weights)
        self.eval()
        return self
            
class ResidualBlock(nn.Module):
    """
    AF() = 

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),

            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, input):
        """ Forward porpagation of input through the residual block """
        return input + self.block(input)
