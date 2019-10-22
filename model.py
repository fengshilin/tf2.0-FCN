import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.initializers import Constant
# from tensorflow.nn import conv2d_transpose

from config import image_shape


def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor % 2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) /
                       factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


class MyModel(tf.keras.Model):
    def __init__(self, n_class):
        super().__init__()
        self.vgg16_model = self.load_vgg()

        self.conv_test = Conv2D(filters=n_class, kernel_size=(1, 1))
        self.deconv_test = Conv2DTranspose(filters=n_class,
                                           kernel_size=(64, 64),
                                           strides=(32, 32),
                                           padding='same',
                                           activation='sigmoid',
                                           kernel_initializer=Constant(bilinear_upsample_weights(32, n_class)))

    def call(self, input):
      x = self.vgg16_model(input)
      x = self.conv_test(x)
      x = self.deconv_test(x)
      return x

    def load_vgg(self):
        # 加载vgg16模型，其中注意input_tensor，include_top
        vgg16_model = tf.keras.applications.vgg16.VGG16(
            weights='imagenet', include_top=False, input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))
        for layer in vgg16_model.layers[:15]:
          layer.trainable = False
        return vgg16_model
