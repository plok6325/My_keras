import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Activation,MaxPool2D,Reshape,BatchNormalization,LeakyReLU,UpSampling2D,Flatten
from keras.activations import relu
from keras.optimizers import SGD
from keras.datasets import mnist
import math as math
from tqdm import tqdm_notebook as tqdm
from keras import backend as K
from PIL import Image
import time
K.set_image_dim_ordering('th')

f= np.load('omniglot.npz')
images= (f['images'])



def discriminator_model():
    model = Sequential()
    model.add(Conv2D(
                    64, (3, 3),
                    padding='same',
                    input_shape=(1, 104, 104)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_model():
    model = Sequential()
    model.add(Dense(512,input_shape=(100,)))
    model.add(Activation('tanh'))
    model.add(Dense(32*13*13))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((32, 13, 13), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.summary()
    return model

def Generator_plus_discriminator(generator,discriminator ):
    model=Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model



def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image

f= np.load('omniglot.npz')
images= f['images']
images=images.astype('float')
images= (images-255/2)/(255/2)
images= np.reshape(images,(images.shape[0],1,images.shape[1],images.shape[1]))


def train(BATCH_SIZE):
    X_train = images
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        Generator_plus_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    #d_optim = Adam(lr=1e-4)
    #g_optim = Adam(lr=1e-3)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in (range(100)):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in tqdm(range(int(X_train.shape[0]/BATCH_SIZE))):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            if index % 10 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True

            if index % 10 == 9:

                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


train(BATCH_SIZE=16)
