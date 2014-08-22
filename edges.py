import cv2
import numpy as np

def nothing(x):
    pass

#image window
cv2.namedWindow('image')

#loading images
img = cv2.imread('leo-messi-pic.jpg',0)     # load your image with proper path

# create trackbars for color change
cv2.createTrackbar('th1','image',0,255,nothing)
cv2.createTrackbar('th2','image',0,255,nothing)

while(1):
    # get current positions of four trackbars
    th1 = cv2.getTrackbarPos('th1','image')
    th2 = cv2.getTrackbarPos('th2','image')
    #apply canny 
    edges = cv2.Canny(img,th1,th2)
    #show the image
    cv2.imshow('image',edges)
    #press ESC to stop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()