"""
oitutils provides basic functionalities.

created 22.2.17
"""

import argparse
import cv2
import imutils
from skimage.transform import pyramid_gaussian


def getclargs():
    """
    Returns the command line arguments
    :return: command line arguments as dict str: str
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    required=False,
                    help="path to video file")
    ap.add_argument("-i", "--image",
                    required=False,
                    help="path to image file")
    ap.add_argument("-W", "--width",
                    required=False,
                    help="width of image display in pixel")
    ap.add_argument("-H", "--height",
                    required=False,
                    help="height of image display in pixel")

    return vars(ap.parse_args())


def scaling(capture, args):
    """
    This function is not used any longer, because functionality is implemente in imutils
    Scale video capture according to the command line arguments
    :param capture: video capture to calculate video width and height
    :param args: commonad line arguments (for details see function getclargs)
    :return:
    """
    w = capture.get(3)
    h = capture.get(4)

    # if new size ist set by width and height
    if args['width'] is not None and args['height'] is not None:
        return (int(args['width']), int(args['height']))
    elif args['width'] is not None and args['height'] is None:
        nw = int(args['width'])
        nh = int(float(nw)/float(w) * h)
        return (nw, nh)
    elif args['width'] is None and args['height'] is not None:
        nh = int(args['height'])
        nw = int(float(nh)/float(h) * w)
        return (nw, nh)
    else:
        return (int(args['width']), int(args['height']))


def show(args):
    """
    Shows the first frame of the video capture
    :param args: these are the args from the command line (for details see function getclargs)
    """
    capture = cv2.VideoCapture(args['video'])
    if args['width'] is not None:
        w = int(args['width'])
    else:
        w = None
    if args['height'] is not None:
        h = int(args['height'])
    else:
        h = None
    img = capture.read()[1]
    img = imutils.resize(img, w, h)
    print(img.shape)
    cv2.imshow("First frame of video capture", img)
    cv2.waitKey(0)


def image_pyramid(args):
    capture = cv2.VideoCapture(args['video'])
    img = capture.read()[1]
    for (i, resized) in enumerate(pyramid_gaussian(img, downscale=2.1415)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break

        # show the resized image
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)


# def downscale(args, factor=1.0):
#     capture = cv2.VideoCapture(args['video'])
#     img = capture.read()[1]
#     return pyramid_gaussian(img, downscale=factor)


def frameshow(args):
    capture = cv2.VideoCapture(args['video'])
    nf = capture.get(7) # this gets the number of frame
    count = 0
    while count < nf:
        count += 1
        img = capture.read()[1]
        cv2.imshow("Frame {}".format(count), img)
        cv2.waitKey(0)


def play_video(args):
    cap = cv2.VideoCapture(args['video'])
    if args['width'] is not None:
        w = int(args['width'])
    else:
        w = None
    if args['height'] is not None:
        h = int(args['height'])
    else:
        h = None

    count = 0
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        frame = imutils.resize(frame, w, h)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame {}'.format(count), frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count > 50:
            break
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


def readImage():
    args = getclargs()
    image = cv2.imread(args["image"])
    return image


def main():
    img = readImage()
    args = getclargs()
    print(args)
    play_video(args)

    #show(args)
    #image_pyramid(args)
    #frameshow(args)

if __name__ == "__main__":
    main()
