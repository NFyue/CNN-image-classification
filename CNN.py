import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Flatten, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from keras.preprocessing import image
from keras import backend
from keras.callbacks import TensorBoard
from keras import regularizers

# load data

#change of image size
imgsize = 140
val_cc = []
val_loss = []
acc = []
loss = []
# imgsize = 0
# imagesize = [28,32,64,100]
# for each in range(0,4):
# 	imgsize = imagesize[each]

train=[]
test=[]
y1=[]
y2=[]
q = -1

name = ['aircraft_carrier','banana_boat','oil_tanker','passenger_ship','yacht']
for picname in range(0,5):
	path="/Users/yueshu/Desktop/testing_images/" + name[picname]
	# print path
	q +=1
	for a,b,c in os.walk(path):
		
		for k in c:
			# print 1
			path1=path+'/'+k
			# print path1
			if "jpg" in k:
				image = cv2.imread(path1)
				re = cv2.resize(image, (imgsize, imgsize))
				re=re.reshape(imgsize,imgsize,3,)
				test.append(re)
				y2.append(q)

z = -1
for picname in range(0,5):
	path="/Users/yueshu/Desktop/training_images2/" + name[picname]
	z += 1
	for a,b,c in os.walk(path):
		# print c
		for k in c:
			path1=path+'/'+k
			# print path1
			if "jpg" in k:
				image1 = cv2.imread(path1)
				re1 = cv2.resize(image1, (imgsize, imgsize))
				re1=re1.reshape(imgsize,imgsize,3,)
				train.append(re1)
				y1.append(z)

num_pixels = imgsize*3
train = np.array(train)
test = np.array(test)
y1=np.array(y1)
y2=np.array(y2)

(X_train, y_train), (X_test, y_test)=(train,y1),(test,y2)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# design model

# #use existed models to fix the weights and get better results:parameters:imgsize;poolingsize;dense
# base_model = InceptionV3(weights='imagenet', input_shape=(imgsize,imgsize,3), include_top=False)
# # print(model.summary())
# model = Sequential()
# model.add(base_model)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes,activation='softmax'))



# trained by the training sets themselves:paramaters:imgsize;num and size of filters;pooling size;dense 




#change dropout
# drop = [0.1,0.2,0.3,0.4,0.5]
# dropsize = 0
# for each in range(0,5):
# 	dropsize = drop[each]

#change neural
# neulist = [32, 64,96, 128, 160,192, 224, 256, 298,330, 362, 394, 426, 458, 490, 512, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 850, 1000]
# neu = 0
# length = len(neulist)
# for each in range(0,length):
# 	neu = neulist[each]

# change epochs
# epolist = [30,40,50,60,70,80]
# epo = 0
# for each in range(0,6):
# 	epo = epolist[each]

#change batch size
# batchlist = [32, 64, 96, 128, 160, 192, 224]
# bat = 0
# for each in range(0,7):
# 	bat = batchlist[each]

# change num of filter
# fil = [4,8,16,32]
# fil_num = 0
# for each in range(0,4) :
# 	fil_num = fil[each]

# change size of filter
# filt = [1,2,3,4,5,6]
# filt_num = 0
# for each in range(0,6):
# 	filt_num = filt[each]

#change pooling size
# pool = [1,2,3,4,5,6,7]
# poolsize = 0
# for each in range(0,7):
# 	poolsize = pool[each]

#change dropout
# drop = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# dropsize = 0
# for each in range(0,8):
# 	dropsize = drop[each]

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(imgsize,imgsize,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(32,(3,3),activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(5,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(64, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
# Fit the model

hist=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=2, callbacks=[tensorboard])

val_cc.append(max(hist.history['val_acc'])*100)
val_loss.append(min(hist.history['val_loss']))
acc.append(max(hist.history['acc'])*100)
loss.append(min(hist.history['loss']))


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print ('training accuracy = ')
print (acc)

print ('testing accuracy = ')
print (val_cc)


print("Baseline Error: %.2f%%" % (100-scores[1]*100))