# -*- coding: utf-8 -*-


from tensorflow import keras

import time
import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, Conv2D, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Reshape, Flatten, SpatialDropout2D 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from InstanceNormalization import InstanceNormalization
import dataset
from utils import save_model, show_image, read_image, load_model

# Shape of images
image_shape = (160, 160, 3)

batch_size = 64
lr_decay_ratio = 0.8
epochs=30





def downsampling_conv_block(x, channels, kernel_size = 4, weight_decay=1e-4, dropout = Dropout, dropout_rate = 0.2):
    x = ZeroPadding2D()(x)
    x = Conv2D(channels, kernel_size, strides=(2, 2), kernel_regularizer=l2(weight_decay), use_bias=False)(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = LeakyReLU(alpha=0.01)(x) 
    x = dropout(dropout_rate)(x)
    return x



def create_model():
    input = Input(shape=image_shape, name='image')
    x = input
    channels = 32
    repeat = 5
    for i in range(repeat - 1):
        channels *= 2
        x = downsampling_conv_block(x, channels, dropout = SpatialDropout2D)
        print(K.int_shape(x))
      
    channels *= 2
    x = downsampling_conv_block(x, channels, dropout_rate = 0.5)
    print(K.int_shape(x))
        
    c_kernel = int(image_shape[0] / (2 ** repeat))
    x = Conv2D(1, c_kernel, use_bias=False, activation='sigmoid')(x)
    print(K.int_shape(x))
    x = Reshape((1,))(x)
    print(K.int_shape(x))
    return Model(input, x)


    

x_train, train_size = dataset.load_celeba('CelebA', batch_size, part='train', consumer = 'classifier')
x_val, val_size = dataset.load_celeba('CelebA', batch_size, part='val', consumer = 'classifier')


def train(learning_rate = 0.0002):
    
    classifier = create_model()
    
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, epsilon=1e-08)
    classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    filepath = 'gender-classifier{epoch:04d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, period=1)
    
    def schedule(epoch, lr):
        if epoch > 0:
            lr *= lr_decay_ratio
        return lr
        
    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
        
    classifier.fit(x_train, epochs=epochs, steps_per_epoch=train_size//batch_size,
                validation_data=(x_val), validation_steps=val_size//batch_size,
                callbacks=[lr_scheduler, checkpoint], verbose=1)

    return classifier




def test_classifier(classifier):
    for part in ('train', 'val', 'test'):
        input_images = dataset.fetch_smallbatch_from_celeba('CelebA', part=part)
        labels = classifier.predict(input_images)
        for image, label in zip(input_images, labels):
            show_image(image)
            print(label)
        


def test(classifier, image_file_name):
    image = read_image(image_file_name)
    label = classifier.predict(np.expand_dims(image, axis = 0))
    
    show_image(image)
    print(label)


classifier = train(0.0005)
save_model(classifier, 'gender-classifier' + str(time.time()))



test_classifier(classifier)

test(classifier, 'test_attr_trans_from_CelebA/201349.jpg')
test(classifier, 'test_attr_trans_from_CelebA/202016.jpg')
test(classifier, 'test_attr_trans_from_CelebA/202052.jpg')
test(classifier, 'test_attr_trans_from_CelebA/202163.jpg')
test(classifier, 'test_attr_trans_from_CelebA/202033.jpg')


test(classifier, 'test_attr_trans_from_CelebA/kate2.jpg')
test(classifier, 'test_attr_trans_from_CelebA/hero.jpg')
test(classifier, 'test_attr_trans_from_CelebA/lqm.jpg')
test(classifier, 'test_attr_trans_from_CelebA/beckham.jpg')
test(classifier, 'test_attr_trans_from_CelebA/dbl.jpg')
test(classifier, 'test_attr_trans_from_CelebA/bad1.jpg')
test(classifier, 'test_attr_trans_from_CelebA/mbp.jpg')
test(classifier, 'test_attr_trans_from_CelebA/nc.jpg')
test(classifier, 'test_attr_trans_from_CelebA/wm1.jpg')



test(classifier, 'test_attr_trans_from_CelebA/jack_r.jpg')
test(classifier, 'test_attr_trans_from_CelebA/rose_r.jpg')
test(classifier, 'test_attr_trans_from_CelebA/trump.jpg')
test(classifier, 'test_attr_trans_from_CelebA/fbb.jpg')
test(classifier, 'test_attr_trans_from_CelebA/lc.jpg')
test(classifier, 'test_attr_trans_from_CelebA/jt.jpg')
test(classifier, 'test_attr_trans_from_CelebA/mnls.jpg')
