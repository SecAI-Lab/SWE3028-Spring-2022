import pickle
import urllib.request
import random

import os
import numpy as np

# save path : path for saving the 'quickdraw datasets' as a .npy file
# category path : path where 'categories.txt' file exists

def download(save_path,category_path):
    total_classes=[]

    with open(category_path,'r') as f:
        total_classes=f.readlines()

    total_classes=list(map(lambda s:s.rstip(),total_classes))

    base='https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    example_image=None

    for i,c in enumerate(total_classes):
        _c=c.replace(' ','%20')
        path=f'{base}{_c}.npy'
        download=urllib.request.urlretrieve(path,f'/{_c}.npy')

        images=np.load(download[0],encoding='latin1',allow_pickle=True)
        images=images.reshape(-1,28*28)

        np.save(save_path+'/{class_list[i]}',images)



