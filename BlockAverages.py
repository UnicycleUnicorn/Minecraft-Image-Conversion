import os
from PIL import Image, ImageStat
import Pickler
from Pickler import Pickles as PS
import numpy as np
import Util
from scipy.spatial import KDTree

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
            int_color = tuple([int(x) for x in average_color])
            # store the average color
            averages.append(int_color)
            files.append(file_path)

Pickler.Save(PS.ImageList, files)
Pickler.Save(PS.AverageList, averages)


preComputedCache = np.zeros(shape = (256, 256, 256), dtype = np.int16)
for r in range(256):
    print(f"R: {r}")
    for g in range(256):
        for b in range(256):
            pixel = (r, g, b)
            closestIndex = 0
            closestDistance = Util.distance(pixel, averages[0])
            for i in range(1, len(averages)):
                distance = Util.distance(pixel, averages[i])
                if (distance < closestDistance):
                    closestDistance = distance
                    closestIndex = i
            preComputedCache[r, g, b] = closestIndex

Pickler.Save(PS.AverageTuple, Util.totuple(preComputedCache))


kdTree = KDTree(averages, copy_data = True)
Pickler.Save(PS.AverageTree, kdTree)