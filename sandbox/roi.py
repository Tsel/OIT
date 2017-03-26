import numpy as np
import cv2
import argparse


def select_roi(image):
    rects = []
    r = cv2.selectROI("Selector", image, rects, fromCenter=False)
    #
    # crop image
    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #
    # display cropped image
    cv2.imshow("cropped image", imCrop)
    cv2.waitKey(0)
    print(rects)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    print("Image is: ", args["image"])

    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(args["image"])

    select_roi(image)


if __name__ == "__main__":
    main()