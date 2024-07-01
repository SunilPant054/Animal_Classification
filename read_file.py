import numpy as np

data_path_train = "/home/pneuma/Documents/Data/Animal_Classification/image_class_train.npz"
data_path_test = "/home/pneuma/Documents/Data/Animal_Classification/image_class_test.npz"
data_path_val = "/home/pneuma/Documents/Data/Animal_Classification/image_class_val.npz"

#Load train data
train_data = np.load(data_path_train)
train_images, train_targets = train_data["image_train"], train_data["label_train"]


#Load test data
test_data = np.load(data_path_test)
test_images, test_targets = test_data["image_test"], test_data["label_test"]


#Load val data
val_data = np.load(data_path_val)
val_images, val_targets = val_data["image_val"], val_data["label_val"]
print(val_images.shape, val_targets.shape)