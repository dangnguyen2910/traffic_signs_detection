import subprocess 
import zipfile
import os
import shutil

zip_file = 'cardetection.zip'


if not os.path.exists("data"):
    subprocess.run("curl -L -o cardetection.zip\
      https://www.kaggle.com/api/v1/datasets/download/pkdarabi/cardetection", shell=True, check=True)

    with zipfile.ZipFile(zip_file, 'r') as f:
        f.extractall()

    os.remove(zip_file)
    os.rename('car', 'data')
    shutil.move('video.mp4', 'data/video.mp4')



