"""
RNN 对 MNIST 进行分类

Train on 54000 samples, validate on 6000 samples
Epoch 1/3
54000/54000 [==============================] - 4s 83us/sample - loss: 0.8596 - accuracy: 0.7112 - val_loss: 0.4418 - val_accuracy: 0.8622
Epoch 2/3
54000/54000 [==============================] - 4s 73us/sample - loss: 0.4262 - accuracy: 0.8710 - val_loss: 0.3250 - val_accuracy: 0.9008
Epoch 3/3
54000/54000 [==============================] - 4s 74us/sample - loss: 0.3319 - accuracy: 0.9006 - val_loss: 0.2248 - val_accuracy: 0.9370
10000/10000 [==============================] - 0s 41us/sample - loss: 0.2639 - accuracy: 0.9255
"""
from tensorflow import keras
from tensorflow_core.python.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from tensorflow_core.python.keras.utils import np_utils

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
# (60000, 28, 28)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 标签转换为 one_hot 格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = keras.Sequential()
model.add(layers.SimpleRNN(units=50, input_shape=(28, 28))) # SimpleRNN是无隐藏层的循环神经网络
model.add(layers.Dense(units=10, activation='softmax'))
model.summary()

# 模型编译
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
