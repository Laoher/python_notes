import os
import random
import numpy as np
np.random.seed(5)
import cv2

X=[]
y=[]

true_path='dataset/'
false_path='dataset/faulty'
# true_files = os.walk(true_path)
for (dirpath, dirnames, filenames) in os.walk(true_path):
    for filename in filenames:
        if filename.endswith('chips (229).jpg'):
            img = cv2.imread(os.path.join(dirpath, filename))
            for i in range(5):
                y = np.random.randint(1,5)
                print(y)
                x = np.random.randint(1,5)
                print(x)
                cropImg = img[y:(y + 50), x:(x + 50)]
                # print(cropImg)
            # print(filename)

#
# def img_shift(img,filename,dirpath):
#     for i in range(1,2,5):
#         cv2.imwrite(os.path.join(dirpath, filename)[:-4]+str(i) +'up.jpg', cv2.warpAffine(img, np.array([[1, 0, 0],[0, 1, i]], dtype=np.float32),(50,50)))
#         cv2.imwrite(os.path.join(dirpath, filename)[:-4]+str(i) + 'down.jpg', cv2.warpAffine(img, np.array([[1, 0, 0], [0, 1, -i]], dtype=np.float32),(50,50)))
#         cv2.imwrite(os.path.join(dirpath, filename)[:-4]+str(i) + 'left.jpg', cv2.warpAffine(img, np.array([[1, 0, -i], [0, 1, 0]], dtype=np.float32),(50,50)))
#         cv2.imwrite(os.path.join(dirpath, filename)[:-4]+str(i) +'right.jpg', cv2.warpAffine(img, np.array([[1, 0, i],[0, 1, 0]], dtype=np.float32),(50,50)))

# # resize
# for (dirpath, dirnames, filenames) in os.walk(data_path):
#     for filename in filenames:
#         if filename.endswith('.jpg'):
#             img = cv2.imread(os.path.join(dirpath, filename))
#             cv2.imwrite(os.path.join(dirpath, filename), cv2.resize(img,(50,50)))
## data augmentation

    # shift
    # for (dirpath, dirnames, filenames) in os.walk(data_path):
    #     for filename in filenames:
    #         if filename.endswith('.jpg'):
    #             img = cv2.imread(os.path.join(dirpath, filename))
    #             img_shift(img,filename,dirpath)
