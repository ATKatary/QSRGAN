import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .qsrgan import QSRGAN
import torch.optim as optim
from .qsrgan_utils import *
from .data_utils import Dataset
from .loss_functions import psnr
from torch.autograd import Variable
from torchvision.utils import make_grid
from .discriminator import Discriminator

### Functions ###
def train_and_validate(device, val_inputs, val_labels, train_inputs, train_labels, batch_size, epochs, lr, home_dir, input_shape, n_gen = 4):
    """
    Tests out the network against an inout image

    Inputs
        :device: the computation device CPU or GPU
        :val_inputs: <Dataset> the inputs for validation
        :val_labels: <Dataset> the labels for validation
        :train_inputs: <Dataset> the inputs for training
        :train_labels: <Dataset> the labels for training
        :batch_size: <int> size of batches to load data in
        :epochs: <int> number of times to train the network
        :lr: [<float>, <float>] rate at which we update the [gen, disc] weights
        :home_dir: <str> the home directory containing subdirectories to read from and write to
        :input_shape: <list> of ints representing the input shape of the images to the discriminator, i.e c x h x w where h, w = size of labels
    
    Outputs
        :returns: the traind SRCNN or SRGAN model
    """
    model_name = input("Model name: ")
    k = None
    while k is None:
        try: k = int(input("Enter integer upscaling factor: "))
        except: k = None

    gen = QSRGAN(n_gen, k).to(device)
    disc = Discriminator(input_shape).to(device)

    gen_lr, disc_lr = lr
    gen_optimizer = optim.SGD(disc.parameters(), lr=gen_lr)
    disc_optimizer = optim.SGD(disc.parameters(), lr=disc_lr)

    train_data = Dataset(train_inputs, train_labels)
    train_loader = train_data.load(batch_size)

    val_data = Dataset(val_inputs, val_labels)
    val_loader = train_data.load(batch_size)

    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
       
        gen_loss, disc_loss, psnr = _train((gen, disc), train_loader, len(train_data), device, lr, feature_extractor, (gen_optimizer, disc_optimizer))
        print(f"Training: Gen Loss: {gen_loss:.3f}\tPSNR: {psnr:.3f}\tDisc Loss: {disc_loss:.3f}")
        
        gen_loss, psnr = _validate(gen, val_loader, epoch, len(val_data), device, home_dir, feature_extractor)
        print(f"Validation: Gen Loss: {gen_loss:.3f}\tPSNR: {psnr:.3f}")
        
        if epoch % 10 == 0:
            end = time.time()
            print(f"Finished epoch {epoch} in: {((end - start) / 60):.3f} minutes\nSaving model ...")
            torch.save(gen.state_dict(), f"{home_dir}/pretrained/{model_name}{epoch}.pth")

    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes\nSaving model ...")

    torch.save(gen.state_dict(), f"{home_dir}/pretrained/{model_name}final.pth")
    return gen


### Helper Functions ###
def _store(path, image):
    """
    Stores the passed in image tensor at the path

    Inputs
    :path: <str> path of location to save the image tensor in
    :image: <torch.Tensor> representing the image with dimensions (c, w, h)
    """
    image = make_grid(image.detach().cpu(), padding=2, normalize=True).numpy()
    image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(path, image)

def _train(models, dataloader, n, device, lr, feature_extractor, optimizer = None, criterion = (nn.MSELoss(), nn.L1Loss())):
    """
    Trains the SRCNN or SRGAN

    Inputs
        :model: <SRCNN> | <SRGAN> to train 
        :dataloader: <DataLoader> loading the training data 
        :n: <int> length of the training data
        :lr: <float> learning rate
        :optimizer: the optimization function for backward propogation, by defualt it is Adam
        :criterion: the loss function, by default MSE
    
    Outputs
        :returns: the final loss and psnr loss of the model
    """
    gen, disc = models

    if optimizer is None: 
        gen_optimizer = optim.Adam(gen.parameters(), lr = lr)
        disc_optimizer = optim.Adam(disc.parameters(), lr = lr)
    else: gen_optimizer, disc_optimizer = optimizer

    gan_criterion, content_criterion = criterion
    gan_criterion.to(device)
    content_criterion.to(device)

    gen.train()
    disc.train()

    real_label = 1
    fake_label = 0
    running_psnr = 0.0
    running_gen_loss = 0.0
    running_disc_loss = 0.0
    batch_size = dataloader.batch_size
    for _, data in tqdm(enumerate(dataloader), total = int(n / batch_size)):
        # low_res = data[0].to(device)
        # label = data[1].to(device)
        low_res = Variable(data[0] / 255).to(device)
        
        label = Variable(data[1] / 255).to(device)

        # Training the generator
        gen_optimizer.zero_grad()

        super_res = gen(low_res)
        disc_output = disc(super_res)

        # real = torch.full((low_res.size(0), *disc.output_shape), real_label, dtype=torch.float, device=device)
        # fake = torch.full((label.size()), fake_label, dtype=torch.float, device=device)
        real = Variable(torch.Tensor(np.ones((disc_output.shape))), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(np.zeros((disc_output.shape))), requires_grad=False).to(device)
        
        gan_loss = gan_criterion(disc_output, real)
        gen_features = feature_extractor(super_res)
        real_features = feature_extractor(label)
        content_loss = content_criterion(gen_features, real_features.detach())
        
        gen_loss = content_loss + (10 ** -3) * gan_loss
        gen_loss.backward()
        gen_optimizer.step()

        # Training the discriminator
        disc_optimizer.zero_grad()

        disc_real_output = disc(label)
        disc_real_loss = gan_criterion(disc_real_output, real)
        disc_fake_output = disc(super_res.detach())
        disc_fake_loss = gan_criterion(disc_fake_output, fake)
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        disc_loss.backward()
        disc_optimizer.step()

        running_gen_loss += gen_loss.item()
        running_disc_loss += disc_loss.item()
        running_psnr += psnr(label, super_res)

    final_gen_loss = running_gen_loss / len(dataloader)
    final_disc_loss = running_disc_loss / len(dataloader)
    final_psnr = running_psnr / len(dataloader)
    return final_gen_loss, final_disc_loss, final_psnr

def _validate(model, dataloader, epoch, n, device, home_dir, feature_extractor, criterion = nn.MSELoss()):
    """
    Tests out the network against an inout image

    Inputs
        :model: <SRCNN> | <SRGAN> to train
        :dataloader: <DataLoader> loading the training data  
        :epoch: <int> epoch this network is currently being trained for
        :n: <int> length of the training data
        :device: the computation device CPU or GPU
        :home_dir: <str> the home directory containing subdirectories to read from and write to
        :criterion: the loss function, by default MSE
    
    Outputs
        :returns: a tuple (loss, psnr) of the final loss and final psnr
    """
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    criterion.to(device)
    batch_size = dataloader.batch_size

    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total = int(n / batch_size)):
            image_data = (data[0] / 255).to(device)
            label = (data[1] / 255).to(device)
            
            output = model(image_data)
            loss = criterion(output, label)

            running_loss += loss.item()
            running_psnr += psnr(label, output)

        output = output.cpu()
        
        if epoch % 1 == 0:
            _store(f"{home_dir}/outputs/training/labels/train{epoch}.png", label)
            _store(f"{home_dir}/outputs/training/super_res/train{epoch}.png", output)
            _store(f"{home_dir}/outputs/training/low_res/train{epoch}.png", image_data)

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / len(dataloader)
    return final_loss, final_psnr