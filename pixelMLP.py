#import timeit
#from __future__ import print_function
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import argparse
from sklearn.neural_network import MLPClassifier


def makeMask(img, orig, mlp):
    simg = img / 255.
    print(simg)
    ooi=mlp.predict(simg)
    ooi *= 255
    print("result is ", ooi)
    mask = ooi.reshape(orig.shape[0], orig.shape[1])
    print("mask is: ")
    print(mask)
    print("mask shape is: ")
    print(mask.shape)
    # mask = np.apply_along_axis(mlp.predict, 1, img)
    # rows, clns = img.shape[:2]
    # #xpos, ypos = 900, 100 # 100, 900
    # #xpos1, ypos1 = 242, 444 # 90, 1250 # 1250, 90
    # mask = np.zeros(img.shape[0], dtype= "uint8")
    # # print("value of image.shape[:2] and image.shape: ", img.shape[:2], img.shape)
    # for row in range(rows):
    #     for col in range(clns):
    #         #print("r,c", row, col)
    #         r, g, b = img[row, col] / 255.
    #         rgb = np.asarray([r, g, b])
    #         mask[row,col] = int(255 * mlp.predict(rgb.reshape(1, -1)))
    # print("Mask is ready")
    # #
    # erode mask
    mask = mask.astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    masked = cv2.bitwise_and(orig, orig, mask = mask)
    cv2.imshow("masked", masked)
    cv2.waitKey(0)
    #
    # r,g,b = img[xpos,ypos] / 255.
    # r1,g1,b1 = img[xpos1,ypos1] / 255.
    # rgb = np.asarray([r,g,b])
    # rgb1 = np.asarray([r1,g1,b1])
    # print("rgb at (", xpos, ypos, ")",  rgb, [r,g,b])
    # print("ooi at (", xpos, ypos, ")", mlp.predict(rgb.reshape(1, -1)))
    # print("rgb at (", xpos1, ypos1, ")",  rgb1, [r1,g1,b1])
    # print("ooi at (", xpos1, ypos1, ")", mlp.predict(rgb1.reshape(1, -1)))
    # print(mask.shape)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, help="Path to the data file")
args = vars(ap.parse_args())
dataset = np.loadtxt(args["data"], delimiter=",")

print(dataset.shape)

#N = 750 # random sample
N = dataset.shape[0] # take all the data
p_train_test = 0.75
traintest = int(np.floor(N*p_train_test))

np.random.shuffle(dataset)
X = dataset[:N,1:4] / 255.
y = dataset[:N,0]
print("sum of y ", y.sum())

X_train, X_test = X[:traintest], X[traintest:]
Y_train, Y_test = y[:traintest], y[traintest:]
print("Shape of X training and test data set: ", X_train.shape, X_test.shape)
print("Shape of Y training and test data set: ", Y_train.shape, Y_test.shape)
print("first 5 rows of X_train: ", X_train[:5].dtype)
#
# set up the neural net
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-3, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train, Y_train)
print("Training set score: %f" % mlp.score(X_train, Y_train))
print("Test set score: %f" % mlp.score(X_test, Y_test))

#
# we now have the information to create a mask for the original image
#
fnimage = args["data"][:-4]
print("Image file name is: ", fnimage)
img = cv2.imread(fnimage)
#imgslice = img[10:15,10:35]
print("Shape of img: ", img.shape)
#reshapedimgslice = imgslice.reshape((imgslice.shape[0]*imgslice.shape[1],3))
reshapedimg = img.reshape((img.shape[0]*img.shape[1],3))
print(reshapedimg)
print(reshapedimg.shape)
#print("slice shape", imgslice.shape)
#mask = makeMask(img, mlp)
makeMask(np.asarray(reshapedimg), img, mlp)
print("We are finished")
