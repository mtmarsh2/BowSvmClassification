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

# ===== Global Variables
build_combined_masks_flag = 0
num_clusters = 1000
sift = cv2.SIFT()
all_image_histograms = {}
image_to_descriptors = {}
non_logo_image_descriptors = {}
folder_name = "/Users/User1/Downloads/FlickrLogos-v2/classes/jpg/starbucks/"
other_folder_name = "/Users/User1/Downloads/FlickrLogos-v2/classes/jpg/no-logo/"
list_of_logo_image_names = glob.glob(os.path.join(folder_name, "*.jpg"))[10:]
list_of_non_logo_image_names = glob.glob(os.path.join(other_folder_name, "*.jpg"))[:50]
assert(len(list_of_logo_image_names) != 0)
assert(len(list_of_non_logo_image_names) != 0)

#1. Read in all images
#2. Generate vocabulary list
#3. Generate histogram bow representation for all images
#4. Run nearest neighbor on target image to find top k close images
#5. See which parameter k is best for classification 

def create_histogram(labels):
     hist, edges = np.histogram(labels, bins=num_clusters, normed=True)
     return hist

def combine_with_mask(image_path):
    image_name = os.path.basename(image_path)
    path_to_mask = "/Users/User1/Downloads/FlickrLogos-v2/classes/masks/starbucks/%s.mask.0.png" % image_name
    return cv2.bitwise_and(cv2.imread(image_path), cv2.imread(path_to_mask))
def classify_images(svm, list_of_image_names):
     labels = []
     for index, image_name in enumerate(list_of_image_names):
          target_hist = create_histogram((kmeans.predict(get_sift_descriptors(image_name))))
          labels.append(clf.predict(target_hist))
     return labels


def test_classification(predicted_labels, trained_labels):
     total_correct = 0
     for elem1, elem2 in zip(predicted_labels, trained_labels):
          if elem1 == elem2:
               total_correct += 1
     return total_correct / len(predicted_labels)
def get_sift_descriptors(image_name):
     img = cv2.imread(image_name)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     [kp, desc] =  sift.detectAndCompute(gray, None)
     #return only subset
     #num_elems = min(500, len(desc))
     #return random.sample(desc, num_elems)
     return kp, desc

def build_dataset(list_of_image_names):
     image_to_descriptors = {}
     for index,image_name in enumerate(list_of_image_names):
          image_to_descriptors[image_name] = get_sift_descriptors(image_name)[1]
     return image_to_descriptors

def create_histogram(labels):
     hist, edges = np.histogram(labels, bins=num_clusters)
     return hist

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def classify_images(svm, list_of_image_names):
     labels = []
     for index, image_name in enumerate(list_of_image_names):
          target_hist = create_histogram((kmeans.predict(get_sift_descriptors(image_name)[1])))
          labels.append(clf.predict(target_hist))
     return labels

def create_labels_matrix(image_names):
    labels_matrix = []
    for i in image_names:
        if i.split("/")[7] == "no-logo":
            labels_matrix.append("no-logo")
        else:
            labels_matrix.append("starbucks")
    return labels_matrix

def ransac_test(target_image_name, list_of_neighbors):
    norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    close_neighbors = []
    target_kp, target_desc = get_sift_descriptors(target_image_name)
    for image_name in list_of_neighbors:
        neighbor_kp, neighbor_desc = get_sift_descriptors(image_name)
        raw_matches = matcher.knnMatch(target_desc, trainDescriptors = neighbor_desc, k = 2) #2
        p1, p2, kp_pairs = filter_matches(target_kp, neighbor_kp, raw_matches)
        p1,p2 = remove_bad_matches(p1, p2)
        p2,p1 = remove_bad_matches(p2, p1)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            if np.sum(status) >= 20: close_neighbors.append(image_name)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)
    return close_neighbors

if(build_combined_masks_flag):
    # ===== Produce logo + mask images
    for image_name in list_of_logo_image_names:
        assert(os.path.isfile(image_name))
        logo_and_mask_image = combine_with_mask(image_name)
        assert(len(logo_and_mask_image) != 0)
        cv2.imwrite("/Users/User1/Downloads/FlickrLogos-v2/classes/combined/%s" % image_name, logo_and_mask_image)

# ===== Read in logo + mask images and compute sift descriptors
image_to_descriptors = build_dataset(list_of_logo_image_names)
list_of_sift_descriptors_logos = np.vstack(image_to_descriptors.values())

# ===== Read in non logo images and compute sift descriptors on those also
non_logo_image_descriptors = build_dataset(list_of_non_logo_image_names)
list_of_sift_descriptors_nonlogos = np.vstack(non_logo_image_descriptors.values())

all_descriptors = np.vstack( (list_of_sift_descriptors_logos, list_of_sift_descriptors_nonlogos))

# ===== Build vocab list using kmeans
if(os.path.isfile('/Users/User1/classify/codeward')):
    kmeans = pickle.load(open('/Users/User1/classify/codeward', 'rb'))
else:
    kmeans = MiniBatchKMeans(n_clusters = num_clusters)
    kmeans.fit(all_descriptors)

# ===== Compute histogram for all logo images
for index,image_name in enumerate(list_of_logo_image_names):
     if index % 1000 ==0: print index
     #image_to_descriptors[image_name] = get_sift_descriptors(image_name)
     labels = kmeans.predict(image_to_descriptors[image_name])
     all_image_histograms[image_name] = create_histogram(labels)

# ===== Compute histogram for all non logo images
for index,image_name in enumerate(list_of_non_logo_image_names):
     if index % 1000 ==0: print index
     #image_to_descriptors[image_name] = get_sift_descriptors(image_name)
     labels = kmeans.predict(non_logo_image_descriptors[image_name])
     all_image_histograms[image_name] = create_histogram(labels)

correct_labels = create_labels_matrix(all_image_histograms.keys())
clf = svm.SVC()
clf.fit(all_image_histograms.values(), correct_labels)
mytree = spatial.KDTree(all_image_histograms.values())

from sklearn.svm import LinearSVC
new_svc = svm.LinearSVC()
# ===== Take new starbucks logo image and compute binary 'and' with histogram of all other images
new_image = glob.glob(os.path.join(folder_name, "*.jpg"))[6]
logo_and_mask_image = combine_with_mask(new_image)
assert(len(logo_and_mask_image) != 0)
cv2.imwrite("/Users/User1/Downloads/FlickrLogos-v2/classes/combined/%s" % os.path.basename(new_image), logo_and_mask_image)
target_image_descriptor = build_dataset([new_image])
labels = kmeans.predict(target_image_descriptor[new_image])
target_histogram = create_histogram(labels)
assert( len(target_histogram) == num_clusters)
# ===== Filter out wrong images with ransac
[a,b] = mytree.query(target_histogram, k=20)
# ===== Create list of names of neighbors
neighbor_list = []
for i in b:
    neighbor_list.append(all_image_histograms.keys()[i])

# ==== Take each neighbor and only return ones which pass ransac test
close_images = ransac_test(new_image, neighbor_list)
print close_images
