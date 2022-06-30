import cv2
import requests
from .segmentor import Segmentor
from .helpers import *

def segment(weights_path, cfg_path, classes_path, img_path, home_dir, store_dir):
    """
    Tests the segementation netowkr and displays the results

    Inputs
        :weights_path: <str> path to the pretrained weights
        :cfg_path: <str> path to the testing confugrations
        :classes_path: <str> path to the file containing the names of classes of interest
        :img_path: <str> path to the image to segment
        :home_dir: <str> path to the directory containing the implementation
        :store_dir: <str> path to where to store segemented images
    """
    network = Segmentor(weights_path, cfg_path, classes_path)
    if img_path is None:
        img_url = input("Image url: ")
        img_path = download(img_url, home_dir, fn="test")

    network.segment(img_path)
    
    if network.fig_img is not None and len(network.roi) > 0: display(network.fig_img)

    i = 0
    results = []
    for img, _, label, confidence in network.roi:
        print(f"Identified {label} with {float(confidence) * 100}% confidence") 
        result_path = f"{store_dir}/result{i}.png"
        if img.shape[0] * img.shape[1] == 0: continue
        cv2.imwrite(result_path, img)
        results.append(result_path)

        i += 1
        display(img)
        
    return results

def download(url, home_dir, stream = False, fn = None):
    """
    Downloads the content of a url to the specified home_dir

    Inputs
        :url: <str> to the location contianing the content
        :home_dir: <str> the home directory containing subdirectories to write to
        :fn: the name to give the file when saved
    
    Outputs
        :returns: the path to the saved file containing the content
    """
    if fn is None:
        fn = url.split('/')[-1]

    r = requests.get(url, stream=stream)
    if r.status_code == 200:
        with open(f"{home_dir}/inputs/images/{fn}.png", 'wb') as output_file:
            if stream:
                for chunk in r.iter_content(chunk_size=1024**2): 
                    if chunk: output_file.write(chunk)
            else:
                output_file.write(r.content)
                print("{} downloaded: {:.2f} KB".format(fn, len(r.content) / 1024.0))
            return f"{home_dir}/inputs/images/{fn}.png"
    else:
        raise ValueError(f"url not found: {url}")
 