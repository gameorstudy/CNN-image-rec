import os
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# root direcotry name
root_dir_name = 'data/'

# create arrays to store different types of images, labels
train_images = []
test_images = []
train_labels = []
test_labels = []

#
w = 75
h = 75

# iterate images files and push into above arrays
for roots, dirs, files in tqdm(os.walk(root_dir_name, topdown=True)):
    for _file in files[:1]:
        file_path_name = os.path.join(roots, _file)
        if 'train' in file_path_name:
            label = file_path_name.split('\\')[-2][5:-5]
        else:
            label = file_path_name.split('\\')[-2][5:-4]
        
        # cv2.imread returns an array
        f = cv2.imread(file_path_name)
        # cv2.resize change the dimensions of the image
        img = cv2.resize(f, (w, h))
        
        if 'train' in file_path_name:
            train_images.append(img.reshape(1, w, h, 3) / 255.0) # reshape / 255.0 to do nomalization
            train_labels.append(label)
        else:
            test_images.append(img.reshape(1, w, h, 3) / 255.0)
            test_labels.append(label)

# np.vstack is used to stack the sequence vertically to make a single array
train_images = np.vstack(train_images)
train_labels = np.array(train_labels)
test_images = np.vstack(test_images)
test_labels = np.array(test_labels)

class_name = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']

# LabelEncoder can be used to normalize labels
le = LabelEncoder()
# fit label encoder
le.fit(class_name)
# transform labels to normalized encoding
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)



print('done')