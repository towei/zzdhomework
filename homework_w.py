import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import skimage.io as io
import os

batch_size = 50
num_classes = 4
epochs = 30
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_homework_model.h5'


str1 = '/home/xdml/PycharmProjects/dltower/homework/DATA/train_h/'+'*.JPEG'
coll1 = io.ImageCollection(str1)
str2 = '/home/xdml/PycharmProjects/dltower/homework/DATA/test_h/'+'*.JPEG'
coll2 = io.ImageCollection(str2)

x_train = np.zeros((len(coll1),25,25,3))
x_test = np.zeros((len(coll2),25,25,3))
for i in range(len(coll1)):
    l = coll1[i]
    m = np.asarray(l)
    x_train[i,:,:,:] = m

for j in range(len(coll2)):
    lb = coll2[j]
    mb = np.asarray(lb)
    x_test[j,:,:,:] = mb

print(x_train.shape)
print(x_test.shape)

d1 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/label_train.txt')
d2 = open('/home/xdml/PycharmProjects/dltower/homework/DATA/label_test.txt')
s1 = d1.read()
s2 = d2.read()
y_train = np.zeros((len(coll1),1))
y_test = np.zeros((len(coll2),1))
for i in range (0,len(coll1)):
    y_train[i,0] = s1[i]
for j in range (0,len(coll2)):
    y_test[j,0] = s2[j]

print(y_train.shape)
print(y_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
          validation_data=(x_test, y_test),shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

json_string = model.to_json()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')

print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])