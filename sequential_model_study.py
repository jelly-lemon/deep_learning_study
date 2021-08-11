"""
Sequential 模型：每一层都只有一个输入张量和一个输出张量。


在以下情况下，顺序模型不合适：
    您的模型有多个输入或多个输出
    您的任何层都有多个输入或多个输出
    你需要做图层共享
    您需要非线性拓扑（例如残差连接、多分支模型）

什么时候应该使用 Keras 函数式 API 来创建新的模型，或者什么时候应该直接对 Model 类进行子类化呢？
通常来说，函数式 API 更高级、更易用且更安全，并且具有许多子类化模型所不支持的功能。
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def test_0():
    """
    定义一个 Sequential 模型，写法 1
    """
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )

    x = tf.ones((3, 3))  # 全 1 3x3 的矩阵
    y = model(x)


def test_1():
    """
    定义一个 Sequential 模型，写法 2
    """
    layer1 = layers.Dense(2, activation="relu", name="layer1")
    layer2 = layers.Dense(3, activation="relu", name="layer2")
    layer3 = layers.Dense(4, name="layer3")

    x = tf.ones((3, 3))
    y = layer3(layer2(layer1(x)))


def test_2():
    """
    定义一个 Sequential 模型，写法 3

    model.pop() 可以用来删除最后一层
    """
    model = keras.Sequential()
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(3, activation="relu"))
    model.add(layers.Dense(4))


def test_3():
    """
    【推荐】定义一个 Sequential 模型，写法 4
    """
    model = keras.Sequential()
    #
    # 指定 input 形状的写法还有：model.add(layers.Dense(2, activation="relu", input_shape=(4,)))
    #
    model.add(keras.Input(shape=(4,)))
    model.add(layers.Dense(2, activation="relu"))

    #
    # 【易错点】只有指定了 Input，才能进行 summary
    # 也就是指定了 input 时，keras 才会初始化模型。
    #
    model.summary()


def test_4():
    """
    特征提取器，提取每一层的输出
    """
    initial_model = keras.Sequential(
        [
            keras.Input(shape=(250, 250, 3)),
            layers.Conv2D(32, 5, strides=2, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
        ]
    )

    #
    # 构建特征提取器
    #
    feature_extractor = keras.Model(
        inputs=initial_model.inputs,
        outputs=[layer.output for layer in initial_model.layers],
    )

    #
    # 对输入样本进行特征提取
    #
    x = tf.ones((1, 250, 250, 3))
    features = feature_extractor(x)


def test_5():
    """
    特征提取，提取某一层
    """
    initial_model = keras.Sequential(
        [
            keras.Input(shape=(250, 250, 3)),
            layers.Conv2D(32, 5, strides=2, activation="relu"),
            layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
            layers.Conv2D(32, 3, activation="relu"),
        ]
    )

    feature_extractor = keras.Model(
        inputs=initial_model.inputs,
        outputs=initial_model.get_layer(name="my_intermediate_layer").output,
    )

    # Call feature extractor on test input.
    x = tf.ones((1, 250, 250, 3))
    features = feature_extractor(x)


def test_6():
    """
    迁移学习，方式 1
    """
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10),
    ])

    #
    # 加载预训练权重
    #
    model.load_weights(...)

    #
    # 设置前 n-1 层不用训练
    #
    for layer in model.layers[:-1]:
        layer.trainable = False

    #
    # compile: 配置训练模型时相关参数
    # fit: 进行训练，实际上只会训练最后一层
    #
    model.compile()
    model.fit(...)


def test_7():
    """
    迁移学习，方式 2
    """
    #
    # 加载预训练模型
    #
    base_model = keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        pooling='avg')

    #
    # 冻结基模型
    # Freeze the base model
    #
    base_model.trainable = False

    #
    # Use a Sequential model to add a trainable classifier on top
    #
    model = keras.Sequential([
        base_model,
        layers.Dense(1000),
    ])

    #
    # Compile & train
    #
    model.compile(...)
    model.fit(...)


if __name__ == '__main__':
    test_3()
