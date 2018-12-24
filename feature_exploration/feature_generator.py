import os
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from zipfile import ZipFile
from multiprocessing.connection import Listener

import keras

vgg = keras.applications.VGG16(include_top=False, pooling='avg')


def image_batch_to_feature_batch(batch, batch_no):
    batch = np.array(batch)
    features = vgg.predict(batch)
    with open('./img_features_vgg16/test/{}.pkl'.format(batch_no), 'wb') as f:
        pickle.dump(features, f)



def main():
    address = ('localhost', 6000)
    print("Starting to listen")
    while True:
        with Listener(address, authkey=b'secret password') as listener:
            with listener.accept() as conn:
                print("Accepted connection")
                while True:
                    try:
                        batch, batch_no = conn.recv()
                    except:
                        break
                    image_batch_to_feature_batch(batch, batch_no)
                    print("Finished with batch ", batch_no)
if __name__ == "__main__":
    main()