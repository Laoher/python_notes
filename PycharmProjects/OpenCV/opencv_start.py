import cv2
import numpy as np
# read
i=cv2.imread('OpenCV_logo.png')
'''
parameters: 
cv2.IMREAD_UNCHANGED: do not change the img type when reading
'''

# show
# cv2.imshow("demo",i)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save
cv2.imwrite('OpenCV_save.png',i)

# get pixel value
print(i[100,100])
print(i.item(88,142,2)) # for RGB image, the last number '2' is necessary

# change pixel value
i[100,100] = [12,12,12]
# cv2.imshow("demo",i)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  # I can see a blue spot that was changed by me
i.itemset((88,142,1),5)  # must follow this syntax

# change multiple pixels
# i[100:150,100:150] = [255,255,255]  # no need to show the image every time

# shape of the image, return no of pixels, Channel
print("size of image i is: ",i.size)
# type of the image, return no of pixels, Channel
print("type of image i is: ", i.dtype)

# image ROI: region of interest
# b=np.ones((400,800,3))  # I am not sure the effect of the row
b=i[185:585, 370:1170]
i[:400,:800]=b
cv2.imshow("modify",i)
cv2.waitKey(-1)
cv2.destroyAllWindows()