"""
LSTM + MNIST
MNIST 非序列数据，不适合 LSTM

Train on 54000 samples, validate on 6000 samples
Epoch 1/3
54000/54000 [==============================] - 9s 159us/sample - loss: 0.6904 - accuracy: 0.7757 - val_loss: 0.2421 - val_accuracy: 0.9278
Epoch 2/3
54000/54000 [==============================] - 7s 138us/sample - loss: 0.2173 - accuracy: 0.9367 - val_loss: 0.1456 - val_accuracy: 0.9553
Epoch 3/3
54000/54000 [==============================] - 7s 137us/sample - loss: 0.1500 - accuracy: 0.9564 - val_loss: 0.1134 - val_accuracy: 0.9678
10000/10000 [==============================] - 1s 73us/sample - loss: 0.1293 - accuracy: 0.9613
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
model.add(layers.LSTM(units=50, input_shape=(28, 28)))
model.add(layers.Dense(units=10, activation='softmax'))
model.summary()

# 模型编译
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)