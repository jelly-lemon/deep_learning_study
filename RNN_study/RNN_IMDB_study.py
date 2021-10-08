"""
RNN + IMDB 二分类

Train on 22500 samples, validate on 2500 samples
Epoch 1/3
22500/22500 [==============================] - 82s 4ms/sample - loss: 0.5765 - accuracy: 0.6796 - val_loss: 0.4442 - val_accuracy: 0.7976
Epoch 2/3
22500/22500 [==============================] - 82s 4ms/sample - loss: 0.3085 - accuracy: 0.8763 - val_loss: 0.4071 - val_accuracy: 0.8348
Epoch 3/3
22500/22500 [==============================] - 82s 4ms/sample - loss: 0.1261 - accuracy: 0.9570 - val_loss: 0.5366 - val_accuracy: 0.8292
25000/25000 [==============================] - 4s 179us/sample - loss: 0.5655 - accuracy: 0.8162
"""
from tensorflow import keras
from tensorflow_core.python.keras.datasets import imdb
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tensorflow_core.python.keras.optimizer_v2.adam import Adam


# 载入数据
# num_words:
# skip_top: 忽略前 N 个出现频率最高的单词（句子里的这个单词被去掉了，那句子变成什么样了？）
# maxlen：具体的最大长度（超出则丢掉）
# x_test：一个句子就是一个数组，数字代表了单词的索引号
# y_test：0 或 1
(x_train, y_train), (x_test, y_test) = imdb.load_data()


# 求词语最大索引号+1 作为词汇表长度，因为 0 是保留的
vocabulary_size = -1
for seq in x_train:
    max_index = max(seq)
    if max_index > vocabulary_size:
        vocabulary_size = max_index

for seq in x_test:
    max_index = max(seq)
    if max_index > vocabulary_size:
        vocabulary_size = max_index

vocabulary_size += 1

# 截长补短，让每句文本长度为 100 个单词
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)


# 创建模型
model = keras.Sequential()
# input_dim: 词汇表的大小，==最大索引值+1
# output_dim: 输出维度（input_length x output_dim）
model.add(Embedding(output_dim=300, input_dim=vocabulary_size, input_length=100))
model.add(SimpleRNN(units=16))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# 模型编译
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
