import os
from PIL import Image, ImageStat
import Pickler
from Pickler import Pickles as PS
import numpy as np

# directory path containing the images
directory_path = "Textures"

files = []
averages = []

# iterate over the images in the directory
for filename in os.listdir(directory_path):
    # construct the full path of the image
    file_path = os.path.join(directory_path, filename)
    # check if the file is an image
    if os.path.isfile(file_path) and filename.lower().endswith('.png'):
        # open the image file
        with Image.open(file_path).convert('RGBA') as image:
            # get the average color of the image
            average_color = ImageStat.Stat(image).mean[0:3]
            # convert the color to an integer
            int_color = np.array(int(x) for x in average_color)
            # store the average color
            averages.append(int_color)
            files.append(file_path)

Pickler.Save(PS.ImageList, files)
Pickler.Save(PS.AverageList, averages)