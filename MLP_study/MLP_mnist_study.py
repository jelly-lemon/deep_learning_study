from tensorflow import keras
from tensorflow.keras import layers


def get_model():
    """
    创建一个简单模型
    """
    inputs = keras.Input(shape=(784,))
    dense_1 = layers.Dense(64, activation="relu")(inputs)
    dense_2 = layers.Dense(64, activation="relu")(dense_1)
    outputs = layers.Dense(10)(dense_2)  # 如果不指定激活函数，默认 Linear（即线性激活函数）
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()
    #
    # 绘制模型结构图
    # Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
    #
    # keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

    return model


def test_0():
    """
    用 MNIST 数据集训练

    训练两个 epoch 就能达到 acc:0.96
    """
    model = get_model()

    #
    # 加载数据
    # 首次运行会从网络下载，https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    # 图像大小：(28, 28)
    # 【注意】开个 VPN 下载更快
    #
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)

    #
    # 先将二维矩阵转一维向量
    # 再归一化 -> [0, 1]
    #
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    #
    # 配置模型
    # 【易错点】keras 一定是来自：from tensorflow import keras
    #
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    #
    # 训练
    #
    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

    #
    # 预测
    # 一个样本的输出长这样：[-6.0854545 -6.199508 -0.62811685  2.1923258 -10.526151 -2.2269807 -11.56955     10.257351 -2.9951472    0.06514306]
    # 这是因为没有使用 softmax
    # y_pred: (10000, 10)
    y_pred = model.predict(x_test)
    print(y_pred.shape)
    print(y_pred[0])

    #
    # 评估
    #
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])


def test_1():
    """
    保存模型

    保存的文件包括：
        - 模型架构
        - 模型权重值（在训练过程中得知）
        - 模型训练配置（如果有的话，如传递给 compile）
        - 优化器及其状态（如果有的话，用来从上次中断的地方重新开始训练）

    model.save() 将整个模型保存为单个文件
    """
    pass


def test_2():
    """
    使用 CNN 模型：LeNet 结构来分类 mnist 数据

    """
    inputs = keras.Input(shape=(28, 28, 1))
    conv2D_1 = layers.Conv2D(6, (3, 3), activation='relu')(inputs)
    pool2D_1 = layers.MaxPool2D((2, 2))(conv2D_1)
    conv2D_2 = layers.Conv2D(16, (3, 3), activation='relu')(pool2D_1)
    pool2D_2 = layers.MaxPool2D((2, 2))(conv2D_2)
    flatten_1 = layers.Flatten()(pool2D_2)
    dense_1 = layers.Dense(120, activation="relu")(flatten_1)
    dense_2 = layers.Dense(84, activation="relu")(dense_1)
    outputs = layers.Dense(10, activation="softmax")(dense_2)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    #
    # 训练
    #
    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
    y_pred = model.predict(x_test)
    print(y_pred[0])
    y_pred = y_pred.argmax(axis=1)
    print(y_pred[0])


    #
    # 评估
    #
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])


if __name__ == '__main__':
    test_2()
