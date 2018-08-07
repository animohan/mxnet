import os
import gzip
import shutil
import struct
import urllib
import numpy as np

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def downloadmnist(path):
    safe_mkdir(path)
    url = 'http://yann.lecun.com/exdb/mnist'
    filenames = ["train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']

    expected_bytes = [9912422, 28881, 1648877, 4542]

    for filename, byte in zip(filenames, expected_bytes):
        download_url = os.path.join(url, filename)
        local_dest = os.path.join(path, filename)
        download_file(download_url, local_dest, byte, True)



def download_file(download_url, local_dest, expected_bytes, unzip_and_remove = True):
    if (os.path.exists(local_dest) or os.path.exists(local_dest[:-3])):
        print('%s already exists'%local_dest)
    else:
        print("Downloading %s" %download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)
        if expected_bytes:
            if file_stat.st_size == expected_bytes:
                print('Successfully downloaded %s' %local_dest)
                if unzip_and_remove :
                    with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3],'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_dest)
            else:
                print("Error: File size mismatch")

def parse_data(path, dataset, flatten):
    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _,num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1

    return imgs, new_labels

def readmnist(path):
    imgs, labels = parse_data(path,'train', flatten = True)
    test_imgs,test_labels= parse_data(path, "t10k", flatten = True)

    indices = np.random.permutation(labels.shape[0])
    train_imgs = imgs[indices]
    train_labels = labels[indices]

    return train_imgs, train_labels,test_imgs, test_labels