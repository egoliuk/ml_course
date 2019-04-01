import numpy as np, os

from scipy import misc
from sklearn.utils import shuffle

ORL_dir = 'dataset/orl_faces'
image_width = 92
image_height = 112

def load_dataset():
    images, labels = load_imgs()
    images, labels = shuffle(images, labels, random_state=0)

    images = np.asarray(images, dtype='float32').reshape([-1, image_width, image_height, 1])

    labels = np.asarray(labels)
    labels = labels.reshape((labels.shape[0], 1))

    return split_dataset(images, labels)

def load_imgs(dataset_dir = ORL_dir, img_shape = (image_width, image_height)):
    (images, labels, ID) = ([], [], 0)

    for(_, dirs, _) in os.walk(dataset_dir):
        dirs = sorted(dirs)
        for sub_dir in dirs:
            sub_path = os.path.join(dataset_dir, sub_dir)
            for file_name in os.listdir(sub_path):
                path = sub_path + "/" + file_name
                img = misc.imread(path, mode='L')

                #print img.shape
                (height, width) = img.shape

                if(width != img_shape[0] or height != img_shape[1]):
                    img = img.resize((img_shape[0], img_shape[1]))

                images.append(img)

                labels.append(int(ID))

            ID += 1

    return images, labels

def split_dataset(dataset, labels):
    training_X = dataset[:int(dataset.shape[0] * 0.8),:]
    training_Y = labels[:int(labels.shape[0] * 0.8),:]
    test_X = dataset[int(dataset.shape[0] * 0.8):,:]
    test_Y = labels[int(labels.shape[0] * 0.8):,:]
    return training_X, training_Y, test_X, test_Y

def normalize(images):
    images = images / 255
    return images