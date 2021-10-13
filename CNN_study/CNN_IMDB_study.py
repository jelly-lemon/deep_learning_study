from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.datasets import imdb
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence


# 载入数据
# num_words:
# skip_top: 忽略前 N 个出现频率最高的单词（句子里的这个单词被去掉了，那句子变成什么样了？）
# maxlen：具体的最大长度（超出则丢掉）
# x_test：一个句子就是一个数组，数字代表了单词的索引号
# y_test：0 或 1
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print("x_train.shape:", x_train.shape)
print("x_train.dtype:", x_train.dtype)
print("y_train.shape:", y_train.shape)
print("y_train.dtype:", y_train.dtype)
#y_train = y_train.astype("float32")
#y_test = y_test.astype("float32")


# 截长补短，让每句文本长度为 100 个单词
x_train = sequence.pad_sequences(x_train, maxlen=250)
x_test = sequence.pad_sequences(x_test, maxlen=250)


# 创建模型
inputs = Input(shape=x_train[0].shape)
embed_1 = Embedding(input_dim=num_words, output_dim=10)(inputs)
conv1D_1 = Conv1D(8, 3, activation='relu')(embed_1)
pool1D_1 = MaxPool1D(2)(conv1D_1)
flatten_1 = Flatten()(pool1D_1)
dense_1 = Dense(16, activation="relu")(flatten_1)
drop_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(8, activation="relu")(dense_1)
drop_2 = Dropout(0.2)(dense_2)
outputs = Dense(1, activation="sigmoid")(drop_2)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["acc"])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
