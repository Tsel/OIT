"""
Script processing enables image processing
"""

from __future__ import print_function
import cv2
import numpy as np
import argparse

def show(img, header="Image"):
    cv2.imshow(header, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def define_rect(image):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = [] # Starting and ending points
    win_name = "image" # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                required=True,
                help = "path to the image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])

(b, g, r) = img[0, 0]
print("Pixel at (0,0) - Red {}, Green {}, Blue {}". format(r,g,b))
cv2.imshow("Original", img)
cv2.waitKey()
cv2.destroyAllWindows()

print("shape of image: {}".format(img.shape))
print("size of image:  {}".format(img.size))

#
# mouse select roi
# Points of the target window
#points = define_rect(img)
#print("Starting point is ", points[0])
#print("Ending   point is ", points[1])

#
# move 5X5 sliding window over image
cx = cy = 0
x = 0
y = 0
dx = dy = 20
red = (0,0,255)
for x in np.arange(0, img.shape[1]-dx, dx):
    for y in np.arange(0,img.shape[0]-dy, dy):
        cv2.rectangle(img, (x, y), (x+dx, y+dy), red,1)
        slicedimage = img[y:y+dy, x:x+dx]
        # print("Shape of sliced image {} at Position x {} , y {}".format(slicedimage.shape, cx, cy))
        # (b, g, r) = np.mean(img[x:x+dx, y:y+dy])
        # print("Pixel at (cx,cv) - Red {}, Green {}, Blue {}".format(uint(r), uint(g), unit(b)))
        cy += 1
    cx += 1

cv2.imshow("Sliding square", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for x in np.arange(0, img.shape[1] - dx, dx):
#     print(x, x + dx)
#     cv2.rectangle(img, (x, y), (x+dx, y + dy), red, 1)
#     cv2.imshow("Sliding rectangle", img)
#     cv2.waitKey(1)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.rectangle(img, (x, y), (x+dx, y + dy), red, 1)
# cv2.imshow("Sliding rectangle", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
