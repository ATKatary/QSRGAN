import os
import cv2
import shutil
import matplotlib.pyplot as plt

def display(img):
    """
    Displays the passes image

    Inputs
        :img: to be displayed
    """
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig('detected.jpg')
    plt.show()

def check_overwrite(dir, overwrite = False): 
    """
    Checks whether to overwrite a directory if it exists, else creates

    Inputs
        :dir: <str> path of directory to check
        :overwrite> <bool> indicating whether to overwrite directory if it exists, default is False
    """
    if os.path.exists(dir):
        if not overwrite: 
            overwrite = input(f"Directory {dir} exists! Do you want to overwrite? (y/n): ")
            overwrite = True if overwrite == "y" else False

        if overwrite: 
            shutil.rmtree(dir)
            os.mkdir(dir)
    
    else: os.mkdir(dir)