from utils import random_crop
import numpy as np
import skimage.io as io
from skimage import data_dir


str1 = '/home/xdml/PycharmProjects/dltower/homework/DATA/train/'+'*.JPEG'
coll1 = io.ImageCollection(str1)

print(len(coll1))

doc1 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/label_train.txt','w')
doc2 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/label_test.txt','w')

for i in range(len(coll1)):
    if i < 4000:
        l = coll1[i]
        crop, label, center = random_crop(l)
        str2 = '/home/xdml/PycharmProjects/dltower/homework/DATA/train_h/' + str(i) +'.JPEG'
        io.imsave(str2,crop)
        doc1.write(str(label))

    if i >= 4000:
        l = coll1[i]
        crop, label, center = random_crop(l)
        str3 = '/home/xdml/PycharmProjects/dltower/homework/DATA/test_h/' + str(i) +'.JPEG'
        io.imsave(str3,crop)
        doc2.write(str(label))

doc1.close()
doc2.close()



