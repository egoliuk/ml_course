import numpy as np, os

from scipy import misc, ndimage
from sklearn.utils import shuffle as sklearn_shuffle

ORL_dir = 'dataset/orl_faces'
num_classes = 40
num_samples_per_class = 10
percent_train_samples = .8
percent_test_samples = .2
image_width = 92
image_height = 112

def load_dataset(shuffle=True):
    images, labels = load_imgs()

    images = np.asarray(images, dtype='float32').reshape([-1, image_width, image_height, 1])

    labels = np.asarray(labels)
    labels = labels.reshape((labels.shape[0], 1))

    if shuffle:
        training_X, training_Y, test_X, test_Y = split_dataset(images, labels)
        training_X, training_Y = sklearn_shuffle(training_X, training_Y, random_state=0)
        test_X, test_Y = sklearn_shuffle(test_X, test_Y, random_state=0)
        return training_X, training_Y, test_X, test_Y
    else:
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

                (height, width) = img.shape

                if(width != img_shape[0] or height != img_shape[1]):
                    img = img.resize((img_shape[0], img_shape[1]))

                images.append(img)

                labels.append(int(ID))

            ID += 1

    return images, labels

def split_dataset(dataset, labels):
    num_train_samples_per_class = int(num_samples_per_class * percent_train_samples)
    num_test_samples_per_class = num_samples_per_class - num_train_samples_per_class
    dataset_length = labels.shape[0]

    train_selection = np.resize(np.concatenate((np.repeat(True, num_train_samples_per_class), 
                                                np.repeat(False, num_test_samples_per_class)), axis=0),
                                (dataset_length, 1))

    training_X = dataset[train_selection.flatten()]
    training_Y = labels[train_selection].reshape((320, 1))
    test_X = dataset[np.invert(train_selection.flatten())]
    test_Y = labels[np.invert(train_selection)].reshape((80, 1))
    return training_X, training_Y, test_X, test_Y

def normalize(images):
    images = images / 255
    return images

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty*2:cropy,startx*2:cropx]

def rotate_image(image, angle):
    """
    Rotate a NumPy image about it centre by the given angle (in degrees).
    The returned image will be cropped and scaled to the same size, any empty borders after rotation will be removed.
    """
    imgWidth=image.shape[1];
    imgHeight=image.shape[0];

    rotated_img = ndimage.interpolation.rotate(image, angle, cval=0.01, mode='constant', reshape=True)
    cropped_img = crop_center(rotated_img, imgWidth, imgHeight)
    zoom_coeff = imgWidth/cropped_img.shape[1]
    zoomed_img = ndimage.zoom(cropped_img, zoom_coeff)
    vertical_offset = int((zoomed_img.shape[0] - imgHeight) / 2)
    out_img = zoomed_img[vertical_offset:vertical_offset+imgHeight, :]
    
    return out_img