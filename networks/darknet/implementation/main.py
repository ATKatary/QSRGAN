import imp
from .segmentor import Segmentor
from .helpers import *

def segment(weights_path, cfg_path, classes_path, img_path):
    """
    Tests the segementation netowkr and displays the results

    Inputs
        :weights_path: <str> path to the pretrained weights
        :cfg_path: <str> path to the testing confugrations
        :classes_path: <str> path to the file containing the names of classes of interest
        :img_path: <str> path to the image to segment
    """
    network = Segmentor(weights_path, cfg_path, classes_path)
    network.segment(img_path)
    network.crop()
    
    
    if network.fig_image is not None: display(network.fig_image)
    for img, _, label, confidence in network.roi.items():
        print(f"Identified {label} with {confidence * 100}% confidence")
        display(img)

    