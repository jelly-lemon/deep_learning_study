import numpy
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence

from attention import Attention


class RecordBestTestAccuracy(Callback):
    """
    一次 epoch 结束时回调函数
    """
    def __init__(self):
        super().__init__()
        self.val_accuracies = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.val_accuracies.append(logs['val_accuracy'])
        self.val_losses.append(logs['val_loss'])

def train_and_evaluate_model_on_imdb(add_attention=True):
    """
    在 IMDB 数据集上训练并评估模型

    :param add_attention: 是否加入注意力机制
    """
    numpy.random.seed(7)

    # 加载数据，只使用前 n 个单词
    # load the dataset but only keep the top n words, zero the rest
    # num_words: 保留词频最高的前5000个单词
    # 一个句子包含了许多单词，把词频很小的单词给去掉
    top_words = 5000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
    print(x_train.shape, x_train[0])
    print(y_train.shape, y_train[0])

    # pad_sequences：将序列转化为经过填充以后的一个长度相同的新序列
    # truncate and pad input sequences
    max_review_length = 500 # 最大长度，不够就填充
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

    # 创建模型
    embedding_vector_length = 32
    model = Sequential([
        Embedding(top_words, embedding_vector_length, input_length=max_review_length),
        Dropout(0.5),
        # attention vs no attention. same number of parameters so fair comparison.
        # *([]) 将一个列表对象解析为单独的多个对象，相当于 LSTM, Attention
        # 是否添加注意力层
        *([LSTM(100, return_sequences=True), Attention()] if add_attention
          else [LSTM(100), Dense(350, activation='relu')]),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ]
    )

    # 配置训练参数
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # 开始训练
    rbta = RecordBestTestAccuracy()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[rbta])

    # 输出最优结果
    print(f"Max Test Accuracy: {100 * np.max(rbta.val_accuracies):.2f} %")
    print(f"Mean Test Accuracy: {100 * np.mean(rbta.val_accuracies):.2f} %")


def main():
    # 10 epochs.
    # Max Test Accuracy: 88.02 %
    # Mean Test Accuracy: 87.26 %
    # train_and_evaluate_model_on_imdb(add_attention=False)
    # 10 epochs.
    # Max Test Accuracy: 88.74 %
    # Mean Test Accuracy: 88.00 %
    train_and_evaluate_model_on_imdb(add_attention=True)


if __name__ == '__main__':
    main()
