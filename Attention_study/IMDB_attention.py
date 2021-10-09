"""
pip install attention
"""
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
        """
        回调函数
        """
        self.val_accuracies.append(logs['val_accuracy'])
        self.val_losses.append(logs['val_loss'])


def train_and_evaluate_model_on_imdb():
    """
    在 IMDB 数据集上训练并评估模型
    """
    # 加载数据，只保留词频前 n 的单词，否则置为 0（一个句子包含了许多单词，把词频很小的单词给去掉）
    # num_words: 保留词频最高的前 5000 个单词
    top_words = 5000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
    print(x_train.shape, x_train[0])
    print(y_train.shape, y_train[0])

    # pad_sequences：长度不足则填充 0
    max_review_length = 500  # 最大长度，不够就填充
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

    # 创建模型
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(0.5))
    model.add(LSTM(100, return_sequences=True))
    model.add(Attention())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # 配置训练参数
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

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
