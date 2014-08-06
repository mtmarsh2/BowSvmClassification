import cv2
import numpy as np
import glob
import os
from sklearn.cluster import KMeans, MiniBatchKMeans, svm
import scipy.cluster.vq as vq
import pdb
import cPickle as pickle
import numpy
from sklearn import svm
"""
1. Read in images from folder
2. Find sift descriptors on all
3. Create codebook with all descriptors
4. for each image, create histogram of features
5. Save to file
6. Use svm to classify target image
"""

folder_name = "/Users/mm71593/classify/train"
sift = cv2.SIFT()
num_clusters = 300
image_to_descriptors = {}
list_of_image_names = glob.glob(os.path.join(folder_name, "*.jpg"))
all_image_histograms = {}
kmeans = MiniBatchKMeans(n_clusters = 1000, batch_size = 1000, max_iter = 250)

def count_type_in_list(mylist, mytype):
	return len([elem for elem in mylist if os.path.basename(elem).split(".")[0] == mytype ])

def get_sift_descriptors(image_name):
	img = cv2.imread(image_name)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	[kp, desc] =  sift.detectAndCompute(gray, None)
	return desc

def compute_codebook(list_of_sift_descriptors):
	return vq.kmeans2(list_of_sift_descriptors, num_clusters)
	

def create_histogram(labels):
	hist, edges = np.histogram(labels, bins=range(num_clusters), normed=True))
	return hist

def create_labels_matrix(image_names):
	return [ os.path.basename(image_name).split(".")[0] for image_name in image_names]

#takes in list of image names, returns has with image_name -> matrix of sift descriptors
def build_dataset(list_of_image_names):
	image_to_descriptors = {}
	for index,image_name in enumerate(list_of_image_names):
		if index % 20 == 0:
			image_to_descriptors[image_name] = get_sift_descriptors(image_name)
	return image_to_descriptors

#dumps descriptors to file
def dump_sift_descriptors_to_file(filename, image_to_descriptors_hash):
	with open(filename, "wb") as myfile:
		pickle.dump(image_to_descriptors_hash, myfile)

#loads descriptors from file
def load_sift_descriptors_from_file(filename):
	with open(filename, "rb") as myfile:
		return pickle.load(myfile)

if __name__ == "__main__":
	image_to_descriptors = build_dataset(list_of_image_names)
	#make a list of all sift descriptors to be fed into kmeans
	list_of_sift_descriptors = np.vstack(image_to_descriptors.values())

	#create bow histograms for images
	for index,image_name in enumerate(list_of_image_names):
		if index % 100 == 0:
			image_to_descriptors[image_name] = get_sift_descriptors(image_name)
			labels = kmeans.predict(image_to_descriptors[image_name])
			all_image_histograms[image_name] = create_histogram(labels)

correct_labels = create_labels_matrix(all_image_histograms.keys())

###SVM Training
clf = svm.SVC()
clf.fit(all_image_histograms.values(), correct_labels)

#start the testing part 
total_correct = 0

for i in list_of_image_names:
	target_hist = create_histogram((kmeans.predict(get_sift_descriptors(image_name))))
	if clf.predict(target_hist)[0] in create_labels_matrix(i):
		total_correct = total_correct + 1
import pdb
for j,image_name in enumerate(list_of_image_names):
	target_hist = create_histogram((kmeans.predict(get_sift_descriptors(image_name))))
	clf.predict(target_hist)[0] + image_name

"""
for image_name in list_of_image_names:
	pass

classify_image()
"""
