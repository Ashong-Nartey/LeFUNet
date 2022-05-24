import os
import sys
import numpy as np
from PIL import Image
from zipfile import ZipFile
from natsort import natsorted

def convert_one_channel(img):
    #Convert 3 channelled images to grayscale 
    if len(img.shape)>2:
        img=img[:,:,0]
        return img
    else:
        return img
      
def pre_splitted_masks(resize_shape=(512,512),path):
    
    dirs=natsorted(os.listdir(path))
    sizes=np.zeros([len(dirs),2])
    masks=img=Image.open(path+dirs[0])
    sizes[0,:]=masks.size
    masks=(masks.resize((resize_shape),Image.ANTIALIAS))
    masks=convert_one_channel(np.asarray(masks))
    for i in range (1,len(dirs)):
        img=Image.open(path+dirs[i])
        sizes[i,:]=img.size
        img=img.resize((resize_shape),Image.ANTIALIAS)
        img=convert_one_channel(np.asarray(img))
        masks=np.concatenate((masks,img)) 
    masks=np.reshape(masks,(len(dirs),resize_shape[0],resize_shape[1],1))
    return masks
