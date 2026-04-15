import os
import numpy as np
from PIL import Image
import tqdm

def load_image(image_path):
    return Image.open(image_path)

def load_data(data_dir):
    data = []
    labels = []
    for label in ["NORMAL", "PNEUMONIA"]:
        label_dir = os.path.join(data_dir, label)
        for image_name in tqdm.tqdm(os.listdir(label_dir), desc=f"Loading {label} images from {data_dir}"):
            image_path = os.path.join(label_dir, image_name)
            data.append(load_image(image_path))
            labels.append(0 if label == "NORMAL" else 1)
    return data, labels