import os
import random
import numpy as np
from PIL import Image
from utils.preprocess import *

def load_dataset(dataset_dir):
    dataset = []
    classes = {}
    target_size = (32,32)
    
    label = 0
    for class_name in os.listdir(dataset_dir):
        
        # if class_name is a directory (excluding .DS_Store)
        if not os.path.isdir(os.path.join(dataset_dir, class_name)):
            continue
        
        classes[label] = class_name
        class_img_path = os.path.join(dataset_dir, class_name)
        
        # preprocess all images in class_img_path and add them to dataset
        for img in os.listdir(class_img_path):
            img_path = os.path.join(class_img_path, img)
            # print(os.path.join(class_name, img))
            # might fail to open image
            try:
                image = Image.open(img_path)
            except:
                continue
                
            preprocessed_image = preprocess(image, target_size)
            
            assert preprocessed_image.shape == (target_size[0]*target_size[1]*3, 1), "image shape = {}".format(preprocessed_image.shape)
            
            dataset.append((preprocessed_image, label))
        
        label += 1
    
    random.shuffle(dataset)
    
    return dataset, classes

