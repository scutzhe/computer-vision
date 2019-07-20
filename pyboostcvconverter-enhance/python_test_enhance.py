import numpy
import pbcvt # your module, also the name of your compiled dynamic library file w/o the extension

a = numpy.array([[1., 2., 3.]])
b = numpy.array([[1.],
                 [2.],
                 [3.]])
print(pbcvt.dot(a, b)) # should print [[14.]]
print(pbcvt.dot2(a, b)) # should also print [[14.]]


import cv2

img = cv2.imread("./origin.jpg")
img_enhance = pbcvt.enhance_msRestinex(img)
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.namedWindow("img_enhance")
cv2.imshow("img_enhance", img_enhance)
cv2.waitKey(0)
cv2.destroyAllWindows()