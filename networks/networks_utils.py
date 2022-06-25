import os 
import shutil 
import pathlib

### Global Constants ###
current_dir = pathlib.Path(__file__).parent.resolve()

def move(src_dir, dst_dir, path_pattern):
    """
    Moves files from one directory to another

    Inputs
        :src_dir: <str> path to the directory containing the files to be moved
        :dst_dir: <str> path to the directory where files will be moved
        :path_pattern: function(dst_dir, i) -> new file name
    """
    i = 0
    for filename in os.listdir(src_dir):
        if filename.endswith(".jpg"):
            src_path = os.path.join(src_dir, filename)
            dst_path = path_pattern(dst_dir, i)
            shutil.move(src_path, dst_path)
            i += 1
