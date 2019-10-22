import tensorflow as tf
import scipy
import cv2
import os
import numpy as np

from model import MyModel
from config import num_epochs, learning_rate, batch_size, weight_path, image_shape, test_dir
from dataload import test_generator
from deeplab import DeepLabV3Plus

def write_img(pred_images, filename):

    pred = pred_images[0]
    COLORMAP = [[0, 0, 255], [0, 255, 0]]
    cm = np.array(COLORMAP).astype(np.uint8)

    pred = np.argmax(np.array(pred), axis=2)

    pred_val = cm[pred]
    cv2.imwrite(os.path.join("data",filename.split("/")[-1]), pred_val)
    print(os.path.join("data",filename.split("/")[-1])+"finished")


test_dataset = tf.data.Dataset.from_generator(
    test_generator, tf.float32, tf.TensorShape([None, None, None]))
test_dataset = test_dataset.batch(5)

model = DeepLabV3Plus(image_shape[0], image_shape[1], nclasses=2)
# model = MyModel(2)
model.load_weights(weight_path+'fcn_20191021')

test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]

for filename in test_filenames:
  image = scipy.misc.imresize(
      scipy.misc.imread(filename), image_shape)
  image = image[np.newaxis, :, :, :].astype("float32")
  out = model.predict(image)
  write_img(out, filename)
