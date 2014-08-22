from sklearn.cross_validation import train_test_split
from __future__ import division
import cv2
import numpy as np
import glob
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
import scipy.cluster.vq as vq
import pdb
import cPickle as pickle
import numpy
from sklearn import svm
import random
import os.path
from scipy import spatial
from test_remove import remove_bad_matches
from sklearn.preprocessing import LabelEncoder
def get_surf_descriptors(image_name):
     surf = cv2.SURF(400)
     img = cv2.imread(image_name)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     [kp, desc] =  surf.detectAndCompute(gray, None)
     return kp, desc

def build_dataset(list_of_image_names):
     image_to_descriptors = {}
     for index,image_name in enumerate(list_of_image_names):
          image_to_descriptors[image_name] = get_surf_descriptors(image_name)[1]
     return image_to_descriptors

def create_labels_matrix(image_names):
    labels_matrix = []
    for i in image_names:
        if i.split("/")[7] == "no-logo":
            labels_matrix.append("no-logo")
        else:
            labels_matrix.append("starbucks")
    return labels_matrix

def create_histogram(labels):
     hist, edges = np.histogram(labels, bins=range(num_clusters + 1), normed=True)
     return hist

"""
"""


label_dict = {0: 'starbucks', 1: 'no-logo'}
folder_name = "/Users/User1/Downloads/FlickrLogos-v2/classes/jpg/starbucks/"
other_folder_name = "/Users/User1/Downloads/FlickrLogos-v2/classes/jpg/no-logo/"
image_names = glob.glob(os.path.join(folder_name, "*.jpg")) + glob.glob(os.path.join(other_folder_name, "*.jpg"))
num_clusters = 1000
all_image_histograms = {}
# ===== Read in logo + mask images and compute sift descriptors
image_to_descriptors = build_dataset(image_names)
all_descriptors = np.vstack(image_to_descriptors.values())

# ===== Build vocab list using kmeans
kmeans = MiniBatchKMeans(n_clusters = num_clusters)
kmeans.fit(all_descriptors)

# ===== Compute histogram for all logo images
for index,image_name in enumerate(image_names):
     if index % 1000 ==0: print index
     labels = kmeans.predict(image_to_descriptors[image_name])
     all_image_histograms[image_name] = create_histogram(labels)

# ===== Partition data
enc = LabelEncoder()
label_encoder = enc.fit(create_labels_matrix(all_image_histograms.keys()))
y = label_encoder.transform(create_labels_matrix(all_image_histograms.keys()))
x_train, x_test, y_train, y_test = train_test_split(all_image_histograms.values(), y,  test_size=.3, random_state=42)

from sklearn.svm import LinearSVC
clf = LinearSVC(C = 1000, loss = "l2")
clf = clf.fit(x_train, y_train)
a = clf.predict(x_test)
print "Accuracy is: %f" % (sum(a == y_test) / float(len(a)))
