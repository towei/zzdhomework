import cv2
import numpy as np

def random_crop(img):
    '''
    at least 5 pixels
    :param img: 64*64*3 image array
    :return: a tuple (25*25 image array, label, center)
    '''
    label = np.random.randint(0, 4)
    center = np.random.randint(12, 27, size=2)
    m = np.array([center,
                  [center[0],63-center[1]],
                  [63-center[0],63-center[1]],
                  [63-center[0],center[1]]])
    center = m[label]
    crop = img[center[0]-12:center[0]+13, center[1]-12:center[1]+13]

    return (crop, label, center)

if __name__=='__main__':
    img = cv2.imread("test.JPEG")
    crop, label, center = random_crop(img)
    print(crop.shape)
    print(label)
    print(center)
    print(crop)
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
