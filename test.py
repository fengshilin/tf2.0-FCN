import tensorflow as tf
import scipy
import cv2
import os
import numpy as np
from PIL import Image

from model import MyModel
from config import num_epochs, learning_rate, batch_size, weight_path, image_shape, test_dir, result_path
from dataload import test_generator
from deeplab import DeepLabV3Plus


COLORMAP = [[0, 0, 255], [0, 255, 0]]
cm = np.array(COLORMAP).astype(np.uint8)

def addweight(pred, test_img):
    # 标签添加透明通道，叠加在原图上
    pred = Image.fromarray(pred.astype('uint8')).convert('RGBA')

    test_img = test_img[0]
    out = np.zeros(test_img.shape, test_img.dtype)
    cv2.normalize(test_img, out, 0,
                  255, cv2.NORM_MINMAX)
    image = Image.fromarray(out.astype('uint8')).convert('RGBA')
    
    image = Image.blend(image,pred,0.3)
    return image


def write_pred(image, pred, x_names):
    
    pred = pred[0]  # pred维度为[h, w, n_class]
    x_name = x_names[0]
    pred = np.argmax(pred, axis=2)  # 获取通道的最大值的指数，比如模型输出某点的像素值为[0.1,0.5]，则该点的argmax为1.
    pred = cm[pred]  # 将预测结果的像素值改为cm定义的值，这是语义分割常用方法。这一步是为了将上一步的1转换为cm的第二个值，即[0,255,0]

    weighted_pred = addweight(pred, image) 
    
    weighted_pred.save(os.path.join(result_path,filename.split("/")[-1]))
    print(filename.split("/")[-1]+" finished")


# def write_img(pred_images, filename):

#     pred = pred_images[0]
#     COLORMAP = [[0, 0, 255], [0, 255, 0]]
#     cm = np.array(COLORMAP).astype(np.uint8)

#     pred = np.argmax(np.array(pred), axis=2)

#     pred_val = cm[pred]
#     cv2.imwrite(os.path.join("data",filename.split("/")[-1]), pred_val)
#     print(os.path.join("data",filename.split("/")[-1])+"finished")


test_dataset = tf.data.Dataset.from_generator(
    test_generator, tf.float32, tf.TensorShape([None, None, None]))
test_dataset = test_dataset.batch(5)

model = DeepLabV3Plus(image_shape[0], image_shape[1], nclasses=2)
# model = MyModel(2)
model.load_weights(weight_path+'fcn_20191021.ckpt')

test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]

for filename in test_filenames:
  image = scipy.misc.imresize(
      scipy.misc.imread(filename), image_shape)  # image的维度为[h, w, channel], 下一步将其转换为[batch, h, w, channel]作为模型的输入, 这里batch=1。
  image = image[np.newaxis, :, :, :].astype("float32")
  out = model.predict(image)  # out的维度为[batch, h, w, n_class]
  write_pred(image, out, filename)
