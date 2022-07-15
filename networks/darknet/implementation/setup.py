import re
import glob
import argparse

def get_data(data_dir):
    """
    Reads the training, validation, and testing data and stores the path to all images in a text file

    Inputs
        :data_dir: <str> path to the directory containing the data
    """
    data = {data_type: glob.glob(f"{data_dir}/{data_type}/*.jpg") for data_type in ['train', 'test', 'valid']}
    for data_type, data in data.items():
        with open(f"data/{data_type}.txt", 'w') as f:
            f.write('\n'.join(data))

def adjust_train_cfg(darknet_path, classes):
    """
    Adjusts the training confugration file for yolov3

    Inputs
        :darknet_path: <str> path to the directory containing the darkent cloned implementation
        :classes: <list> of classes of interest
    """
    n = len(classes)
    batches = 2000 * n
    filters = 3 * (n + 5)

    with open(f"{darknet_path}/cfg/yolov3-train.cfg", "r+") as cfg_file:
        cfg_content = cfg_file.read()
        cfg_content = re.sub("batch=1", "batch=64", cfg_content)
        cfg_content = re.sub("classes=80", f"classes={n}", cfg_content)
        cfg_content = re.sub("subdivisions=1", "subdivisions=64", cfg_content)
        cfg_content = re.sub("filters=255", f"filters={filters}", cfg_content)
        cfg_content = re.sub("max_batches = 500200", f"max_batches={batches}", cfg_content)
        
        cfg_file.seek(0)
        cfg_file.truncate(0)
        cfg_file.write(cfg_content)
        cfg_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--classes', type=list, default=['driver'])
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--darknet_path', type=str, default='/content/darkent')

    args = parser.parse_args()
    get_data(args.data_dir)
    adjust_train_cfg(args.darknet_path, args.classes)