import numpy as np, os
import math
# import cv2

from scipy import misc, ndimage
from sklearn.utils import shuffle

from PIL import Image
# 
# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.gray()

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

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty*2:cropy,startx*2:cropx]

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """
#     im = Image.fromarray(np.uint8(image))
    imgWidth=image.shape[1];
    imgHeight=image.shape[0];
#     w = math.sin(angle * math.pi/180) * imgHeight + math.cos(angle * math.pi/180) * imgWidth;
#     h = math.sin(angle * math.pi/180) * imgWidth + math.cos(angle * math.pi/180) * imgHeight;
#     new = np.pad(image,((int((w - image.shape[0])/2),),(int((h-image.shape[1])/2),)),'constant')
#     plus10_img = ndimage.interpolation.rotate(new, angle, cval=0.01, reshape=False)
    
    rotated_img = ndimage.interpolation.rotate(image, angle, cval=0.01, mode='constant', reshape=True)
#     print("After rotation: ", rotated_img.shape)
    cropped_img = crop_center(rotated_img, imgWidth, imgHeight)
#     print("After crop: ", cropped_img.shape)
    zoom_coeff = imgWidth/cropped_img.shape[1]
#     print("Zoom coeff: ", zoom_coeff)
    zoomed_img = ndimage.zoom(cropped_img, zoom_coeff)
    vertical_offset = int((zoomed_img.shape[0] - imgHeight) / 2)
#     print("After zoom: ", zoomed_img.shape)
#     print("Vertical offset: ", vertical_offset)
    out_img = zoomed_img[vertical_offset:vertical_offset+imgHeight, :]
    
    return out_img