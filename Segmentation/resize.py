

# IMPORTS
import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image




# Set paths
image_input_dir = "./archive/brisc2025/segmentation_task/train/images"          # folder containing original .jpg images
mask_input_dir = "./archive/brisc2025/segmentation_task/train/masks"            # folder containing original .png masks

image_output_dir = "./archive/brisc2025/segmentation_input2/train/images" # folder to save resized images
mask_output_dir = "./archive/brisc2025/segmentation_input2/train/masks"   # folder to save resized masks

# Desired output size
new_size = (448,448)# (width, height)

# Create output directories if they don't exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)


def resize_image(input_path, output_path, size, is_mask=False):
    img = Image.open(input_path)

    # For masks → nearest neighbor (keeps integer class values)
    if is_mask:
        img = img.resize(size, Image.NEAREST)
    else:
        # For images → bilinear interpolation
        img = img.resize(size, Image.BILINEAR)

    img.save(output_path)


# Resize all JPG images
for filename in os.listdir(image_input_dir):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(image_input_dir, filename)
        output_path = os.path.join(image_output_dir, filename)
        resize_image(input_path, output_path, new_size, is_mask=False)
        print("Resized image:", filename)

# Resize all PNG masks
for filename in os.listdir(mask_input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(mask_input_dir, filename)
        output_path = os.path.join(mask_output_dir, filename)
        resize_image(input_path, output_path, new_size, is_mask=True)
        print("Resized mask:", filename)

print("Done! Images and masks resized and saved.")

# Set paths
image_input_dir = "./archive/brisc2025/segmentation_task/test/images"          # folder containing original .jpg images
mask_input_dir = "./archive/brisc2025/segmentation_task/test/masks"            # folder containing original .png masks

image_output_dir = "./archive/brisc2025/segmentation_input2/test/images" # folder to save resized images
mask_output_dir = "./archive/brisc2025/segmentation_input2/test/masks"   # folder to save resized masks

 

# Create output directories if they don't exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)


# Resize all JPG images
for filename in os.listdir(image_input_dir):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(image_input_dir, filename)
        output_path = os.path.join(image_output_dir, filename)
        resize_image(input_path, output_path, new_size, is_mask=False)
        print("Resized image:", filename)

# Resize all PNG masks
for filename in os.listdir(mask_input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(mask_input_dir, filename)
        output_path = os.path.join(mask_output_dir, filename)
        resize_image(input_path, output_path, new_size, is_mask=True)
        print("Resized mask:", filename)

print("Done! Images and masks resized and saved.")