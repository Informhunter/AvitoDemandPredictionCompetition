import os
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from zipfile import ZipFile
from tqdm import tqdm
from multiprocessing.connection import Client


def PIL2array(img):
    return np.array(img.getdata(),
                    np.float32).reshape(img.size[1], img.size[0], 3)


def main():

    z = ZipFile('../data/test_jpg.zip')
    
    

    address = ('localhost', 6000)

    conn = Client(address, authkey=b'secret password')
    
    print('Connection established')
    

    sizes = []
    batch_size = 1000
    batch_no = 1
    k = 0
    batch = []

    for name in tqdm(z.namelist()):
        if not name.endswith('.jpg'):
            print("Wut.jpg")
            continue
        h = os.path.basename(name)[:-4]
        try:
            img = Image.open(z.open(name))
        except Exception as e:
            print("Wut.open", e)
            continue

        sizes.append(img.size)
        img = img.resize((224, 224))
        img = PIL2array(img)
        img /= 127.5
        img -= 1.
        batch.append(img)
        k += 1
        if k == batch_size:
            conn.send((batch, batch_no))
            batch_no += 1
            k = 0
            batch = []

    if batch:
        conn.send((batch, batch_no))
            
if __name__ == "__main__":
    main()