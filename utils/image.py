import numpy as np
from PIL import Image

# convert to grayscale → resize to 64×64 → scale to [0,1] → flatten
def preprocess_image(image, target_size=(64, 64)):
    grayscale_image = image.convert("L") # why the hell is there RGB images
    resized_image = grayscale_image.resize(target_size)
    resized_image = np.array(resized_image, dtype=np.float32) / 255.0
    return resized_image.flatten()

