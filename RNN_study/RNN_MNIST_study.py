# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow_core.python.keras.datasets import mnist
from tensorflow.keras import layers

from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.keras.utils import np_utils
from tensorflow_core.python.keras.utils.vis_utils import plot_model

# 数据长度，一行有28个像素
input_size = 28
# 序列长度，一共有28行
time_steps = 28
# 隐藏层cell个数
cell_size = 50

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (60000,28,28)
x_train = x_train / 255.0
y_train = y_train / 255.0
# 转换为one_hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = keras.Sequential()
model.add(layers.SimpleRNN(units=cell_size, input_shape=(time_steps, input_size)))
model.add(layers.Dense(units=10, activation='softmax'))

# 模型编译
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=10)
plot_model(model, to_file='77.png', show_shapes=True, show_layer_names='True', rankdir='TB')

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
