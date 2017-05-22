from scipy.io import loadmat
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
import numpy as np

data= loadmat('data4students.mat')
X = data['datasetInputs']
Y = data['datasetTargets']
x_train=X[0,0]
x_val= X[0,1]
x_test=X[0,2]
y_train=Y[0,0]
y_val= Y[0,1]
y_test=Y[0,2]

x_train=x_train/255.0
x_val= x_val/255.0
x_test=x_test/255.0


x_train = x_train.reshape(x_train.shape[0],1,30,30)
x_val = x_val.reshape(x_val.shape[0],1,30,30)
x_test = x_test.reshape(x_test.shape[0],1,30,30)

model=Sequential()
model.add(Conv2D(128,(5,5),activation='relu',data_format='channels_first',input_shape=(1,30,30)))

model.add(Conv2D(32,(3,3),activation='tanh'))

model.add(Conv2D(16,(3,3),activation='tanh'))

model.add(MaxPool2D((2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='SGD')
history = model.fit(x_train,y_train,epochs=100,batch_size=50)

model.evaluate(x_test,y_test, batch_size=32, verbose=0)
