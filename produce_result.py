import os


# img_name = os.listdir('/home/xdml/PycharmProjects/dltower/homework/DATA/test_crop_easy/')
# doc1 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/result1.txt')
# doc2 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/17021211059_easy.txt','w')

img_name = os.listdir('/home/xdml/PycharmProjects/dltower/homework/DATA/test_crop_hard/')
doc1 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/result2.txt')
doc2 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/17021211059_hard.txt','w')

img_name.sort()

s1 = doc1.read()

for i in range(0,len(img_name)):
    wr = str(img_name[i]+','+s1[i]+'\n')
    doc2.write(str(wr))

doc2.close()

