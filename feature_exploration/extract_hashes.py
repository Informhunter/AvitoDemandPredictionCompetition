import os
from PIL import Image
import pickle
from zipfile import ZipFile
from tqdm import tqdm


def main():

    z = ZipFile('../data/test_jpg.zip')
    

    hashlist = []

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

        hashlist.append(h)

    with open('./test_img_features_vgg16/test_hashes.pkl', 'wb') as f:
        pickle.dump(hashlist, f)

            
if __name__ == "__main__":
    main()