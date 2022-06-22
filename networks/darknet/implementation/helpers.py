import cv2
import matplotlib.pyplot as plt

def display(img):
    """
    Displays the passes image

    Inputs
        :img: to be displayed
    """
    plt.figure(figsize=(24, 24))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig('detected.jpg')
    plt.show()