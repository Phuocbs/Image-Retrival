from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import (
    read_image_from_path,
    folder_to_images
)

from offline_feature_extractor import FeatureExtractor

root_img_path = "img/"
root_fearure_path = "feature/"
dic_categories = ['scenery', 'furniture', 'animal', 'plant']
size = (224, 224)


fe = FeatureExtractor()

imgs_feature = []
paths_feature = []

# GPU: Wall time: 5min 31s
for folder in os.listdir(root_img_path):
    if folder.split("_")[0] in dic_categories:
        path = root_img_path + folder
        print(path)
        images_np, images_path = folder_to_images(path)
        paths_feature.extend(np.array(images_path))
        imgs_feature.extend(fe.extract(images_np))
        
np.savez_compressed(root_fearure_path+"all_feartures", array1=np.array(paths_feature), array2=np.array(imgs_feature))