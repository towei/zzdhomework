import numpy as np
import skimage.io as io
from keras.models import model_from_json


model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

# str1 = '/home/xdml/PycharmProjects/dltower/homework/DATA/test_crop_easy/'+'*.JPEG'
# coll1 = io.ImageCollection(str1)
str1 = '/home/xdml/PycharmProjects/dltower/homework/DATA/test_crop_hard/'+'*.JPEG'
coll1 = io.ImageCollection(str1)

# doc1 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/result1.txt','w')
doc1 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/result2.txt','w')


pre = np.zeros((len(coll1),25,25,3))
res = np.zeros((len(coll1),1))
a = np.zeros((len(coll1),4))

def sitemax(x):
    for j in range(0,4):
        if x[j] == np.max(x):
            return j

for i in range(len(coll1)):
    l = coll1[i]
    m = np.asarray(l)
    pre[i,:,:,:] = m


a = model.predict(pre, batch_size=2, verbose=0)
print(a)

for k in range (0,len(coll1)):
    w = a[k,:]
    print(w)
    s = sitemax(w)
    doc1.write(str(s))

doc1.close()



