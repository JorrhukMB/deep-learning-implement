import numpy as np
from PIL import Image

def expand2square(img):
    width, height = img.size
    if width == height:
        return img
    
    if width > height:
        ret = Image.new('RGB', (width, width), (0, 0, 0))
        ret.paste(img, (0, (width - height // 2)))
    else:
        ret = Image.new('RGB', (height, height), (0, 0, 0))
        ret.paste(img, ((height - width // 2), 0))
        
    return ret

def preprocess(img, target_size):
    ret = img.convert('RGB')
    ret = expand2square(ret)
    ret = ret.resize(target_size)
    
    ret = np.array(ret)
    ret = ret.reshape((target_size[0]*target_size[1]*3, 1))
    
    return ret

