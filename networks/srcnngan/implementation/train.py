from pickletools import optimize
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .srcnn import SRCNN
from .srcnn_utils import *
import torch.optim as optim
from .loss_functions import psnr
from .data_utils import SRCNNDataset
from torchvision.utils import save_image
from .discriminator import Discriminator

### Functions ###
def train_and_validate(device, val_inputs, val_labels, train_inputs, train_labels, batch_size, epochs, lr, home_dir, beta1 = 0.5):
    """
    Tests out the network against an inout image

    Inputs
        :device: the computation device CPU or GPU
        :val_inputs: <SRCNNDataset> the inputs for validation
        :val_labels: <SRCNNDataset> the labels for validation
        :train_inputs: <SRCNNDataset> the inputs for training
        :train_labels: <SRCNNDataset> the labels for training
        :batch_size: <int> size of batches to load data in
        :epochs: <int> number of times to train the network
        :lr: <float> rate at which we update the network weights
        :home_dir: <str> the home directory containing subdirectories to read from and write to
    
    Outputs
        :returns: the traind SRCNN model
    """
    gen = SRCNN().to(device)
    gen.apply(random_init)
    gen_optimizer = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

    disc = Discriminator().to(device)
    disc.apply(random_init)
    disc_optimizer = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
        
    train_data = SRCNNDataset(train_inputs, train_labels)
    train_loader = train_data.load(batch_size)

    real_label = 1
    fake_label = 0

    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        criterion = nn.BCELoss()
        
        for _, data in tqdm(enumerate(train_loader), total=len(train_data) / batch_size):
            lr, gt = data
            lr = lr.to(device)
            gt = gt.to(device)

            disc.zero_grad()

            output = disc(gt)
            label = torch.full(output.shape, real_label, dtype=torch.float, device=device)
            disc_real_loss = criterion(output, label)

            disc_real_loss.backward()

            sr = gen(lr)
            label.fill_(fake_label)
            output = disc(sr.detach())
            disc_fake_loss = criterion(output, label)
            disc_fake_loss.backward()
            disc_loss = disc_real_loss + disc_fake_loss
            disc_optimizer.step()

            gen.zero_grad()
            label.fill_(real_label)
            output = disc(sr)
            gen_loss = criterion(output, gt)
            gen_loss.backward()
            gen_optimizer.step()

        print(f"D Loss: {disc_loss:.3f}\tG Loss: {gen_loss:.3f}")

    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes\nSaving model ...")

    model_name = input("Model name:")
    torch.save(gen.state_dict(), f"{home_dir}/pretrained/{model_name}.pth")
    return gen
