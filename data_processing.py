import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian

from tqdm import tqdm 

meta_path = "/home/pneuma/Desktop/ML/Deep Learning/Animal Classification /Animal_Classification_DataSet"
train_dir = "/home/pneuma/Desktop/ML/Deep Learning/Animal Classification /Animal_Classification_DataSet/train"
test_dir = "/home/pneuma/Desktop/ML/Deep Learning/Animal Classification /Animal_Classification_DataSet/test"
val_dir = "/home/pneuma/Desktop/ML/Deep Learning/Animal Classification /Animal_Classification_DataSet/val"
extension = ['.jpg', '.png']
images_class = ["cat", "dog", "elephant", "tiger"]


#Load image data from directory
def get_filenames(dirname):
    file_paths = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return filepath

#Load train image from directory
def get_train_filename(dirname):
    file_path = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_path.append(filepath)
    return file_path

#Load test image from directory
def get_test_filename(dirname):
    file_path = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_path.append(filepath)
    return file_path

#Load val image from directory
def get_val_filename(dirname):
    file_path = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_path.append(filepath)
    return file_path


#filenames for all images, train images, test images, and val images
all_image_files = get_filenames(meta_path)
train_image_files = get_train_filename(train_dir)
test_image_files = get_train_filename(test_dir)
val_image_files = get_train_filename(val_dir)

#One hot encoding
def encode_class(class_name, image_class):
    idx = image_class.index(class_name)
    vec = np.zeros(len(image_class))
    vec[idx] = 1
    return vec

#function to Read image
def read_image(image_file):
    img = []
    label_image = []
    for file in image_file:
        image = cv.imread(file)
        #resize image
        image = cv.resize(image, (224, 224), interpolation=cv.INTER_CUBIC)
        img.append(image)
        
        #Data Label preparation
        basepath, _ = os.path.split(file)
        _, label_name = os.path.split(basepath)
        
        #using one hot encoding
        target = encode_class(label_name, images_class)
        label_image.append(target)
        
    return img, label_image

#get train image and label
img, label_image = read_image(train_image_files)
X_train = np.array(img)
y_train = np.array(label_image)

#get test image and label
img, label_image = read_image(test_image_files)
X_test = np.array(img)
y_test = np.array(label_image)

#get val image and label
img, label_image = read_image(val_image_files)
X_val = np.array(img)
y_val = np.array(label_image)

#Augmenting train data
final_train_data = []
final_target_train = []
for i in tqdm(range(X_train.shape[0])):
    final_train_data.append(X_train[i])
    final_train_data.append(gaussian(X_train[i], sigma=1))
    final_train_data.append(rotate(X_train[i], angle=45))
    final_train_data.append(random_noise(X_train[i], var=0.2**2))
    final_train_data.append(np.fliplr(X_train[i]))
    final_train_data.append(np.flipud(X_train[i]))
    
    for j in range(6):
        final_target_train.append(y_train[j])
        
final_data_train = np.array(final_train_data)
final_target_train = np.array(final_target_train)

#saving data
np.savez("/home/pneuma/Documents/Data/Animal_Classification/image_class_train.npz",
         image_train=final_data_train, label_train=final_target_train)
np.savez("/home/pneuma/Documents/Data/Animal_Classification/image_class_test.npz",
         image_test=X_test, label_test=y_test)
np.savez("/home/pneuma/Documents/Data/Animal_Classification/image_class_val.npz",
         image_val=X_val, label_val=y_val)
