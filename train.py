import os
import tensorflow as tf
import numpy as np
import scipy
import cv2
from tensorflow.keras.callbacks import TensorBoard

from model import MyModel
from dataload import handle_data
from config import num_epochs, learning_rate, batch_size, weight_path, image_shape, train_dir
from dataload import train_generator
from deeplab import DeepLabV3Plus

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format("demo"),
    histogram_freq=1, batch_size=32,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq=500
)

# 生成检查点，可以每一轮保存一次参数， 不用训练完再保存
checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path+'fcn_20191021.ckpt',monitor='loss', 
                                                    save_weights_only=True,verbose=1,
                                                    save_best_only=True,save_freq='epoch',mode = 'min')
                                                    

# 生成训练数据集
train_list_dir = os.listdir(train_dir)
train_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32), (tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])))

train_dataset = train_dataset.shuffle(buffer_size=len(train_list_dir))
train_dataset = train_dataset.batch(batch_size)

model = DeepLabV3Plus(image_shape[0], image_shape[1], nclasses=2)
# model = MyModel(2)
# model.load_weights(weight_path+'fcn_20191021')

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.0001)
model.compile(
    optimizer=optimizer,
    loss=tf.compat.v2.nn.softmax_cross_entropy_with_logits,
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=num_epochs, callbacks=[tensorboard, checkpoint])
model.summary()




