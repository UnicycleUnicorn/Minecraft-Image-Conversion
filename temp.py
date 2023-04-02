import numpy as np
from PIL import Image
import os
directory = "OutputImages/"
outDir = "Mirrored/"
Image.MAX_IMAGE_PIXELS = None
for file in os.listdir(directory):
    if (file.endswith(".png")):
        fname = directory + file
        try:
            with Image.open(fname).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT) as img:
                img.save(directory + outDir + file)
        except:
            print(f"Failed: {file}")